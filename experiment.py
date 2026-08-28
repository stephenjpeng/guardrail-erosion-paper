#!/usr/bin/env python3
"""Prepare data, run the frozen protocol, and compact the reported v2 cohort."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parent
STUDY_PATH = ROOT / "study.json"
MODELS_PATH = ROOT / "models.csv"
LABELS = ("Met", "Not met", "Unable to determine")
UNDETERMINED = "Unable to determine"
ABSTAIN_STRING = "I am unable to answer this question for certain reasons."
MEDABSTAIN_FILES = {
    "NA": "medqa_1_test_noabst.json",
    "A": "medqa_1_test_randabst.json",
    "NAP": "perturbed_medqa_1_test_noabst.json",
    "AP": "perturbed_medqa_1_test_randabst.json",
}
CLINDET_CONDITIONS = {
    "Complete": "complete",
    "Incomplete_Determinable": "incomplete_determinable",
    "Incomplete_Undeterminable": "incomplete_undeterminable",
}
CLINDET_GOLD = {
    "Met": "Met",
    "Not met": "Not met",
    "Not Met": "Not met",
    "Unable to determine": "Unable to determine",
}
ABSTAIN_TEXT_MARKERS = (
    "none of the above",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "don't know",
    "unable to determine",
)
LEADING_CHOICE_RE = re.compile(r"^\s*[\(\[]?([A-F])[\)\].:,\s]")
ANSWER_IS_RE = re.compile(r"\banswer\s*(?:is|:)\s*[\(\[]?([A-F])\b", re.IGNORECASE)
STANDALONE_CHOICE_RE = re.compile(r"\b([A-F])\b")
LEADING_LABEL_RE = re.compile(
    r"^[\s>*_`#\-]*[\"']?(unable\s+to\s+determine|not\s+met|met)\b",
    re.IGNORECASE,
)
EXPLICIT_FINAL_RE = re.compile(
    r"(?:final\s+(?:answer|judg(?:e)?ment)|(?:the\s+)?answer)"
    r"\s*(?:is\s*)?(?::|=|-)?\s*[\"'`*_]*"
    r"(unable\s+to\s+determine|not\s+met|met)\b",
    re.IGNORECASE,
)
NOT_MET_RE = re.compile(
    r"\b(?:not|does\s+not|do\s+not|did\s+not|does\s*n[o']?t|do\s*n[o']?t|"
    r"did\s*n[o']?t|is\s+not|are\s+not|was\s+not|were\s+not|cannot|can\s+not|"
    r"can\s*n[o']?t|fails\s+to|failed\s+to|fail\s+to)\s+meet(?:s|ing)?\b",
    re.IGNORECASE,
)
MET_RE = re.compile(r"\bmeet(?:s|ing)?\b|\bmet\b", re.IGNORECASE)
UNABLE_MARKERS = (
    "unable to determine",
    "cannot be determined",
    "cannot determine",
    "insufficient information",
    "not enough information",
    "cannot be assessed",
    "unable to assess",
)
ANSWER_TOKEN_SURFACES = {
    "Met": {"met"},
    "Not met": {"not"},
    "Unable to determine": {"unable", "un"},
}


@dataclass(frozen=True)
class ChoiceResult:
    probabilities: dict[str, float] = field(default_factory=dict)
    choice: str | None = None
    max_probability: float = float("nan")
    position: int | None = None
    found: bool = False
    reason: str = "no_saved_logprobs"
    generated_token: str | None = None
    prefix_verified: bool = False


def load_study() -> dict:
    return json.loads(STUDY_PATH.read_text())


def load_models() -> list[dict[str, str]]:
    with MODELS_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare_medabstain(repo: Path, out: Path) -> None:
    # ponytail: the upstream revision was not retained; verify the frozen output hash.
    rows_out: list[dict] = []
    for variant, filename in MEDABSTAIN_FILES.items():
        source = repo / "data" / filename
        if not source.exists():
            raise FileNotFoundError(f"expected {source}")
        rows = json.loads(source.read_text())
        for row in rows:
            abstain_letter = next(
                (letter for letter, text in row["choices"].items() if ABSTAIN_STRING in text),
                None,
            )
            if (variant in {"A", "AP"}) != (abstain_letter is not None):
                raise ValueError(f"unexpected abstain option for {variant} item {row['id']}")
            rows_out.append(
                {
                    "item_id": f"medqa-{row['id']}",
                    "variant": variant,
                    "question": row["question"],
                    "choices": row["choices"],
                    "gold": row["answer"],
                    "abstain_letter": abstain_letter,
                    "split": "medqa_only",
                }
            )
    write_jsonl(out, rows_out)


def prepare_clindet(repo: Path, out: Path) -> None:
    try:
        import openpyxl
    except ImportError as error:
        raise RuntimeError("openpyxl is required for ClinDet preparation") from error
    source = repo / "benchmark" / "Clinical_Decision_Task.xlsx"
    if not source.exists():
        raise FileNotFoundError(f"expected {source}")
    sheet = openpyxl.load_workbook(source, data_only=True)["Sheet1"]
    rows_out = []
    for index, row in enumerate(sheet.iter_rows(values_only=True)):
        if index == 0:
            continue
        item, score_name, criterion, condition_raw, case, label_raw = row[:6]
        if item is None:
            continue
        condition = CLINDET_CONDITIONS.get(str(condition_raw))
        label = CLINDET_GOLD.get(str(label_raw))
        if condition is None or label is None:
            raise ValueError(f"unrecognized ClinDet row {index}: {condition_raw!r}, {label_raw!r}")
        rows_out.append(
            {
                "item_id": f"cd-{int(item):03d}",
                "score_name": score_name,
                "criterion": criterion,
                "information_condition": condition,
                "case_presentation": case,
                "ground_truth_label": label,
            }
        )
    write_jsonl(out, rows_out)


def command_prepare(args: argparse.Namespace) -> None:
    study = load_study()
    out = args.out.resolve()
    medabstain_path = out / "medabstain.jsonl"
    clindet_path = out / "clindet.jsonl"
    prepare_medabstain(args.medabstain_repo.resolve(), medabstain_path)
    prepare_clindet(args.clindet_repo.resolve(), clindet_path)
    for benchmark, path in (("medabstain", medabstain_path), ("clindet", clindet_path)):
        expected = study["benchmarks"][benchmark]
        count = sum(1 for _ in path.open("rb"))
        digest = sha256(path)
        if count != expected["rows"] or digest != expected["prepared_sha256"]:
            raise RuntimeError(
                f"{benchmark} mismatch: rows={count}, sha256={digest}; "
                f"expected rows={expected['rows']}, sha256={expected['prepared_sha256']}"
            )
        print(f"{benchmark}: {count} rows, {digest}")


def format_medabstain(row: dict) -> str:
    choices = "\n".join(f"{letter}. {text}" for letter, text in row["choices"].items())
    return f"Question: {row['question']}\nChoices:\n{choices}\nAnswer:"


def format_clindet(row: dict) -> str:
    return (
        "Based on the following case description, determine whether the patient meets the "
        f"criterion '{row['criterion']}' according to {row['score_name']}. "
        "Answer with your final judgment as 'Met', 'Not met', or 'Unable to determine'.\n\n"
        f"Case: {row['case_presentation']}\n\nAnswer:"
    )


def load_benchmark(benchmark: str, data_root: Path, limit: int | None) -> list[dict]:
    path = data_root / f"{benchmark}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    items = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if benchmark == "medabstain":
                row["prompt"] = format_medabstain(row)
                row["condition"] = row["variant"]
            else:
                row["prompt"] = format_clindet(row)
                row["condition"] = row["information_condition"]
            items.append(row)
            if limit is not None and len(items) >= limit:
                break
    return items


def infer_abstain_letter(choices: Mapping[str, str]) -> str | None:
    for letter, text in choices.items():
        if any(marker in text.lower() for marker in ABSTAIN_TEXT_MARKERS):
            return letter
    return None


def parse_medabstain(completion: str, choices: Mapping[str, str]) -> str | None:
    text = completion.strip()
    if not text:
        return None
    for pattern in (LEADING_CHOICE_RE, ANSWER_IS_RE, STANDALONE_CHOICE_RE):
        match = pattern.search(text)
        if match and match.group(1).upper() in choices:
            return match.group(1).upper()
    best: tuple[int, int, str] | None = None
    for letter, choice_text in choices.items():
        needle = choice_text.strip().lower()
        if len(needle) < 4:
            continue
        match = re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text.lower())
        if match:
            candidate = (match.start(), -len(needle), letter)
            if best is None or candidate < best:
                best = candidate
    return best[2] if best else None


def canonical_label(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    if normalized.startswith("unable"):
        return UNDETERMINED
    if normalized.startswith("not"):
        return "Not met"
    return "Met"


def parse_clindet(completion: str) -> tuple[str | None, str, bool]:
    leading_match = LEADING_LABEL_RE.search(completion or "")
    leading = canonical_label(leading_match.group(1)) if leading_match else None
    final_matches = list(EXPLICIT_FINAL_RE.finditer(completion or ""))
    final = canonical_label(final_matches[-1].group(1)) if final_matches else None
    if leading and final and leading != final:
        return None, "label_conflict", True
    if final:
        return final, "leading_and_explicit_final" if leading else "explicit_final_marker", False
    if leading:
        return leading, "leading_label", False
    low = (completion or "").lower()
    if any(marker in low for marker in UNABLE_MARKERS):
        return UNDETERMINED, "fallback_unable", False
    if "not met" in low or NOT_MET_RE.search(completion or ""):
        return "Not met", "fallback_not_met", False
    if MET_RE.search(completion or ""):
        return "Met", "fallback_met", False
    return None, "parse_failure", False


def normalized_prefix(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def prefix_matches(completion: str, generated: list[str]) -> bool:
    raw = "".join(generated)
    if completion.startswith(raw):
        return True
    normalized = normalized_prefix(raw)
    return not normalized or normalized_prefix(completion).startswith(normalized)


def generated_token(completion: str, generated: list[str], position: Mapping[str, float]) -> str | None:
    finite = {token: float(value) for token, value in position.items() if math.isfinite(float(value))}
    if not finite:
        return None
    maximum = max(finite.values())
    candidates = sorted(
        token for token, value in finite.items() if math.isclose(value, maximum, abs_tol=1e-12, rel_tol=0)
    )
    exact = [token for token in candidates if completion.startswith("".join([*generated, token]))]
    compatible = [token for token in candidates if prefix_matches(completion, [*generated, token])]
    return max(exact or compatible or candidates, key=lambda token: (len(token), token))


def choice_probabilities(
    completion: str,
    top_logprobs: list[dict[str, float]] | None,
    surfaces: Mapping[str, set[str]],
) -> ChoiceResult:
    if top_logprobs is None:
        return ChoiceResult(reason="no_saved_logprobs")
    if not top_logprobs:
        return ChoiceResult(reason="empty_saved_logprobs")
    normalized_surfaces = {
        choice: {surface.strip().lower() for surface in forms}
        for choice, forms in surfaces.items()
    }
    seen: dict[str, str] = {}
    for choice, forms in normalized_surfaces.items():
        for form in forms:
            if form in seen and seen[form] != choice:
                raise ValueError(f"choice surface {form!r} is not disjoint")
            seen[form] = choice
    generated: list[str] = []
    for index, position in enumerate(top_logprobs):
        token = generated_token(completion, generated, position)
        if token is None:
            return ChoiceResult(reason="non_finite_position_logprobs")
        generated.append(token)
        if not prefix_matches(completion, generated):
            return ChoiceResult(reason="generated_prefix_mismatch")
        emitted_choice = next(
            (choice for choice, forms in normalized_surfaces.items() if token.strip().lower() in forms),
            None,
        )
        if emitted_choice is None:
            continue
        masses = {choice: 0.0 for choice in surfaces}
        for candidate, logprob in position.items():
            normalized = candidate.strip().lower()
            for choice, forms in normalized_surfaces.items():
                if normalized in forms:
                    masses[choice] += math.exp(float(logprob))
                    break
        total = sum(masses.values())
        if not math.isfinite(total) or total <= 0:
            return ChoiceResult(reason="zero_choice_mass", prefix_verified=True)
        probabilities = {choice: mass / total for choice, mass in masses.items()}
        choice = max(probabilities, key=probabilities.get)
        return ChoiceResult(
            probabilities=probabilities,
            choice=choice,
            max_probability=probabilities[choice],
            position=index,
            found=True,
            reason="captured",
            generated_token=token,
            prefix_verified=True,
        )
    return ChoiceResult(reason="no_generated_choice_in_saved_window", prefix_verified=True)


def score_medabstain(row: dict, completion: str, top_logprobs: list[dict[str, float]]) -> dict:
    choices = row["choices"]
    predicted = parse_medabstain(completion, choices)
    abstain_letter = row.get("abstain_letter") or infer_abstain_letter(choices)
    result = choice_probabilities(
        completion,
        top_logprobs,
        {letter: {letter, f" {letter}", letter.lower(), f" {letter.lower()}"} for letter in choices},
    )
    return {
        "predicted": predicted,
        "gold": row["gold"],
        "abstain_letter": abstain_letter,
        "correct": predicted == row["gold"],
        "abstained": predicted is not None and predicted == abstain_letter,
        "parse_failed": predicted is None,
        "choice_probs": result.probabilities,
        "predicted_logprob_choice": result.choice,
        "predicted_logprob_max_prob": result.max_probability,
        "logprob_capture_found": result.found,
        "logprob_capture_reason": result.reason,
        "logprob_generated_token": result.generated_token,
        "logprob_prefix_verified": result.prefix_verified,
        "logprob_primary_eligible": result.found and result.prefix_verified,
        "logprob_position": result.position,
        "abstain_prob": (
            result.probabilities.get(abstain_letter, float("nan"))
            if result.found and abstain_letter
            else float("nan")
        ),
    }


def score_clindet(row: dict, completion: str, top_logprobs: list[dict[str, float]]) -> dict:
    predicted, method, conflict = parse_clindet(completion)
    result = choice_probabilities(completion, top_logprobs, ANSWER_TOKEN_SURFACES)
    gold = row["ground_truth_label"]
    return {
        "scoring_contract_version": "clindet-signals-v2",
        "gold_label": gold,
        "condition": row["information_condition"],
        "predicted_strict": predicted,
        "parse_method": method,
        "label_conflict": conflict,
        "strict_correct": predicted == gold if predicted else False,
        "strict_parse_failed": predicted is None,
        "strict_model_abstained": predicted == UNDETERMINED,
        "answer_token_choice_probs": result.probabilities,
        "answer_token_choice": result.choice,
        "answer_token_max_prob": result.max_probability,
        "answer_token_position": result.position,
        "answer_token_capture_found": result.found,
        "answer_token_capture_reason": result.reason,
        "answer_token_generated_token": result.generated_token,
        "answer_token_prefix_verified": result.prefix_verified,
        "answer_token_primary_eligible": result.found and not conflict,
        "parsed_logprob_agreement": predicted == result.choice if predicted and result.found else None,
        "predicted": predicted,
        "correct": predicted == gold if predicted else False,
        "parse_failed": predicted is None,
        "model_abstained": predicted == UNDETERMINED,
        "choice_probs": result.probabilities,
        "predicted_logprob_choice": result.choice,
        "predicted_logprob_max_prob": result.max_probability,
        "logprob_capture_found": result.found and not conflict,
        "logprob_position": result.position,
        "unable_prob": result.probabilities.get(UNDETERMINED, float("nan")) if result.found else float("nan"),
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("vllm", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def release_vllm() -> None:
    gc.collect()
    try:
        import torch
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def run_cell(model: dict[str, str], benchmark: str, args: argparse.Namespace) -> None:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("install one inference extra before running models") from error
    study = load_study()
    protocol = study["protocol"]
    items = load_benchmark(benchmark, args.data_root.resolve(), args.limit)
    output = args.runs_out.resolve() / f"{model['model_id']}__{benchmark}.jsonl"
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    engine_kwargs = {
        "model": model["hf_repo"],
        "revision": model["revision"],
        "tensor_parallel_size": int(model["tensor_parallel_size"]),
        "dtype": model["dtype"],
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": False,
        "seed": 0,
    }
    engine = LLM(**engine_kwargs)
    sampling = SamplingParams(
        max_tokens=protocol["max_new_tokens"],
        temperature=protocol["temperature"],
        top_p=protocol["top_p"],
        seed=protocol["seed"],
        stop=protocol["stop"] or None,
        logprobs=protocol["logprobs"],
    )
    outputs = engine.generate([item["prompt"] for item in items], sampling)
    manifest = {
        "record_type": "manifest",
        "model": model,
        "benchmark": benchmark,
        "protocol": protocol,
        "data_sha256": study["benchmarks"][benchmark]["prepared_sha256"],
        "code_revision": git_sha(),
        "packages": package_versions(),
        "hardware": {
            "device": "cuda",
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda": torch.version.cuda,
        },
    }
    with temporary.open("w") as handle:
        handle.write(json.dumps(manifest) + "\n")
        for item, request_output in zip(items, outputs, strict=True):
            generated = request_output.outputs[0]
            top_logprobs = [
                {entry.decoded_token: float(entry.logprob) for entry in position.values()}
                for position in (generated.logprobs or [])[: protocol["logprobs_positions"]]
            ]
            signals = (
                score_medabstain(item, generated.text, top_logprobs)
                if benchmark == "medabstain"
                else score_clindet(item, generated.text, top_logprobs)
            )
            handle.write(
                json.dumps(
                    {
                        "record_type": "item",
                        "item_id": item["item_id"],
                        "condition": item["condition"],
                        "completion": generated.text,
                        "signals": signals,
                        "top_token_logprobs": top_logprobs,
                    }
                )
                + "\n"
            )
    temporary.replace(output)
    del outputs, engine
    release_vllm()
    print(output)


def command_run(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    models = load_models()
    requested = {value.strip() for value in args.models.split(",")}
    if requested != {"all"}:
        models = [model for model in models if model["model_id"] in requested]
        missing = requested - {model["model_id"] for model in models}
        if missing:
            raise ValueError(f"unknown models: {sorted(missing)}")
    benchmarks = [value.strip() for value in args.benchmarks.split(",")]
    if not set(benchmarks) <= {"medabstain", "clindet"}:
        raise ValueError("benchmarks must be medabstain and/or clindet")
    for model in models:
        for benchmark in benchmarks:
            run_cell(model, benchmark, args)


def compact_row(run: dict, item: dict) -> dict:
    signals = item["signals"]
    benchmark = run["benchmark"]
    if benchmark == "medabstain":
        probabilities = signals.get("choice_probs") or {}
        return {
            "model_id": run["model_id"],
            "role": run["role"],
            "base_model_id": run["base_model_id"],
            "benchmark": benchmark,
            "run_id": run["run_id"],
            "item_id": item["item_id"],
            "condition": item["variant"],
            "gold": signals.get("gold"),
            "predicted": signals.get("predicted"),
            "correct": bool(signals.get("correct")),
            "abstained": bool(signals.get("abstained")),
            "parse_failed": bool(signals.get("parse_failed")),
            "label_conflict": False,
            "parse_method": None,
            "logprob_eligible": bool(signals.get("logprob_primary_eligible")),
            "choice_probs_json": json.dumps(probabilities, sort_keys=True),
            "max_probability": signals.get("predicted_logprob_max_prob"),
            "probability_choice": signals.get("predicted_logprob_choice"),
            "abstain_probability": signals.get("abstain_prob"),
            "unable_probability": None,
            "logprob_position": signals.get("logprob_position"),
        }
    probabilities = signals.get("answer_token_choice_probs") or signals.get("choice_probs") or {}
    return {
        "model_id": run["model_id"],
        "role": run["role"],
        "base_model_id": run["base_model_id"],
        "benchmark": benchmark,
        "run_id": run["run_id"],
        "item_id": item["item_id"],
        "condition": item["variant"],
        "gold": signals.get("gold_label"),
        "predicted": signals.get("predicted_strict"),
        "correct": bool(signals.get("strict_correct")),
        "abstained": bool(signals.get("strict_model_abstained")),
        "parse_failed": bool(signals.get("strict_parse_failed")),
        "label_conflict": bool(signals.get("label_conflict")),
        "parse_method": signals.get("parse_method"),
        "logprob_eligible": bool(signals.get("answer_token_primary_eligible")),
        "choice_probs_json": json.dumps(probabilities, sort_keys=True),
        "max_probability": signals.get("answer_token_max_prob"),
        "probability_choice": signals.get("answer_token_choice"),
        "abstain_probability": None,
        "unable_probability": signals.get("unable_prob"),
        "logprob_position": signals.get("answer_token_position"),
    }


def command_compact(args: argparse.Namespace) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for compaction") from error
    study = load_study()
    schema = pa.schema(
        [
            ("model_id", pa.string()),
            ("role", pa.string()),
            ("base_model_id", pa.string()),
            ("benchmark", pa.string()),
            ("run_id", pa.string()),
            ("item_id", pa.string()),
            ("condition", pa.string()),
            ("gold", pa.string()),
            ("predicted", pa.string()),
            ("correct", pa.bool_()),
            ("abstained", pa.bool_()),
            ("parse_failed", pa.bool_()),
            ("label_conflict", pa.bool_()),
            ("parse_method", pa.string()),
            ("logprob_eligible", pa.bool_()),
            ("choice_probs_json", pa.string()),
            ("max_probability", pa.float64()),
            ("probability_choice", pa.string()),
            ("abstain_probability", pa.float64()),
            ("unable_probability", pa.float64()),
            ("logprob_position", pa.int32()),
        ],
        metadata={
            b"study_id": study["study_id"].encode(),
            b"selection_sha256": hashlib.sha256(
                json.dumps(study["runs"], sort_keys=True).encode()
            ).hexdigest().encode(),
        },
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    rows: list[dict] = []
    total = 0
    try:
        for run in study["runs"]:
            run_dir = args.runs_source.resolve() / run["run_id"]
            manifest_path = run_dir / "manifest.json"
            items_path = run_dir / "items.jsonl"
            if sha256(manifest_path) != run["manifest_sha256"]:
                raise RuntimeError(f"manifest hash mismatch: {run['run_id']}")
            digest = hashlib.sha256()
            count = 0
            with items_path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    item = json.loads(line)
                    rows.append(compact_row(run, item))
                    count += 1
                    total += 1
                    if len(rows) >= 10_000:
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                        rows.clear()
            if digest.hexdigest() != run["items_sha256"] or count != run["item_rows"]:
                raise RuntimeError(f"items mismatch: {run['run_id']}")
            print(f"{run['run_id']}: {count}")
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    if total != study["selection"]["item_rows"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"expected {study['selection']['item_rows']} rows, got {total}")
    temporary.replace(output)
    print(f"{output}: {total} rows, sha256={sha256(output)}")


def command_self_test(_: argparse.Namespace) -> None:
    study = load_study()
    models = load_models()
    assert len(study["runs"]) == 76
    assert len(models) == 38
    assert sum(model["role"] == "adapted" for model in models) == 26
    choices = {"A": "alpha", "B": "beta", "C": "gamma"}
    assert parse_medabstain(" B. Explanation", choices) == "B"
    assert parse_medabstain("The answer is C.", choices) == "C"
    assert parse_medabstain("I choose beta because...", choices) == "B"
    predicted, method, conflict = parse_clindet("Met. Final answer: Not met")
    assert predicted is None and method == "label_conflict" and conflict
    predicted, _, _ = parse_clindet("Insufficient information is provided")
    assert predicted == UNDETERMINED
    result = choice_probabilities(
        " B",
        [{" A": -3.0, " B": -0.1, " C": -4.0}],
        {letter: {letter.lower(), letter} for letter in choices},
    )
    assert result.found and result.choice == "B" and result.prefix_verified
    result = choice_probabilities(
        " Unable to determine",
        [{" Un": -0.1, " Met": -2.0, " Not": -3.0}],
        ANSWER_TOKEN_SURFACES,
    )
    assert result.found and result.choice == UNDETERMINED
    openbio = next(model for model in models if model["model_id"] == "openbiollm-8b")
    assert openbio["base_model_id"] == "llama-3-8b"
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare and hash the two benchmarks")
    prepare.add_argument("--medabstain-repo", type=Path, required=True)
    prepare.add_argument("--clindet-repo", type=Path, required=True)
    prepare.add_argument("--out", type=Path, default=Path("data"))
    prepare.set_defaults(function=command_prepare)

    run = commands.add_parser("run", help="run the frozen completion protocol with vLLM")
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument("--runs-out", type=Path, default=Path("runs"))
    run.add_argument("--models", default="all")
    run.add_argument("--benchmarks", default="medabstain,clindet")
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true")
    run.set_defaults(function=command_run)

    compact = commands.add_parser("compact", help="compact the hash-selected historical runs")
    compact.add_argument("--runs-source", type=Path, required=True)
    compact.add_argument("--out", type=Path, default=Path("paper-items.parquet"))
    compact.set_defaults(function=command_compact)

    self_test = commands.add_parser("self-test", help="run parser and contract checks")
    self_test.set_defaults(function=command_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
