#!/usr/bin/env python3
"""Regenerate the audit's pair metrics, inference, sensitivities, and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
STUDY = json.loads((ROOT / "study.json").read_text())
SEED = int(STUDY["analysis"]["seed"])
N_BOOTSTRAP = int(STUDY["analysis"]["bootstrap_replicates"])
N_CLUSTER_BOOTSTRAP = int(STUDY["analysis"]["cluster_bootstrap_replicates"])
N_PERMUTATION = int(STUDY["analysis"]["permutation_replicates"])
MA_CONDITIONS = ("NA", "A", "NAP", "AP")
CD_CONDITIONS = ("complete", "incomplete_determinable", "incomplete_undeterminable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join([str(SEED), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def notebook_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def load_models() -> pd.DataFrame:
    models = pd.read_csv(ROOT / "models.csv", keep_default_na=False)
    models["param_b"] = models["param_size"].str.extract(r"([0-9.]+)")[0].astype(float)
    for column in ("strict_lineage", "ma_behavior_quality", "ma_probability_quality", "cd_behavior_quality"):
        models[column] = models[column].astype(str).str.lower().eq("true")
    models["max_effective_batch_sequences"] = pd.to_numeric(
        models["max_effective_batch_sequences"], errors="coerce"
    )
    return models


def load_items(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if len(frame) != STUDY["selection"]["item_rows"]:
        raise ValueError(f"expected {STUDY['selection']['item_rows']} rows, got {len(frame)}")
    if frame["run_id"].nunique() != STUDY["selection"]["runs"]:
        raise ValueError("run count does not match study.json")
    expected_runs = {run["run_id"] for run in STUDY["runs"]}
    if set(frame["run_id"].unique()) != expected_runs:
        raise ValueError("Parquet run selection differs from study.json")
    frame["choice_probs"] = frame["choice_probs_json"].map(
        lambda value: json.loads(value) if isinstance(value, str) and value else {}
    )
    for column in (
        "correct",
        "abstained",
        "parse_failed",
        "label_conflict",
        "logprob_eligible",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def aligned(
    frame: pd.DataFrame,
    adapted: str,
    base: str,
    benchmark: str,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = frame[frame["benchmark"].eq(benchmark) & frame["condition"].eq(condition)]
    base_rows = subset[subset["model_id"].eq(base)].sort_values("item_id").reset_index(drop=True)
    adapted_rows = subset[subset["model_id"].eq(adapted)].sort_values("item_id").reset_index(drop=True)
    if base_rows["item_id"].duplicated().any() or adapted_rows["item_id"].duplicated().any():
        raise ValueError(f"duplicate item IDs for {adapted}/{benchmark}/{condition}")
    if not base_rows["item_id"].equals(adapted_rows["item_id"]):
        raise ValueError(f"unaligned items for {adapted}/{benchmark}/{condition}")
    return base_rows, adapted_rows


def binomial_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, value) for value in range(k + 1)) / (2**n)


def mcnemar(base: np.ndarray, adapted: np.ndarray) -> tuple[int, float]:
    base = np.asarray(base, dtype=bool)
    adapted = np.asarray(adapted, dtype=bool)
    lost = int((base & ~adapted).sum())
    gained = int((~base & adapted).sum())
    discordant = lost + gained
    p_value = 1.0 if discordant == 0 else min(1.0, 2 * binomial_cdf(min(lost, gained), discordant))
    return discordant, p_value


def paired_bootstrap_ci(
    base: np.ndarray,
    adapted: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    base = np.asarray(base, dtype=float)
    adapted = np.asarray(adapted, dtype=float)
    if not len(base):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(base), size=(n_bootstrap, len(base)))
    estimates = adapted[indexes].mean(axis=1) - base[indexes].mean(axis=1)
    return tuple(map(float, np.percentile(estimates, [2.5, 97.5])))


def add_binary(row: dict, prefix: str, base: Sequence[bool], adapted: Sequence[bool]) -> None:
    base_array = np.asarray(base, dtype=bool)
    adapted_array = np.asarray(adapted, dtype=bool)
    discordant, p_value = mcnemar(base_array, adapted_array)
    row[f"{prefix}_base"] = float(base_array.mean())
    row[f"{prefix}_adapted"] = float(adapted_array.mean())
    row[f"{prefix}_delta"] = row[f"{prefix}_adapted"] - row[f"{prefix}_base"]
    row[f"{prefix}_n"] = len(base_array)
    row[f"{prefix}_n_discordant"] = discordant
    row[f"{prefix}_p_item"] = p_value
    if prefix == "ma_ap_abstention":
        low, high = paired_bootstrap_ci(
            base_array,
            adapted_array,
            seed=stable_seed(row["adapted_model_id"], "ma-AP-abstention"),
        )
        row[f"{prefix}_ci_low"] = low
        row[f"{prefix}_ci_high"] = high


def probability_arrays(frame: pd.DataFrame, labels: Sequence[str]) -> dict[str, np.ndarray]:
    matrix = np.array(
        [[probabilities.get(label, 0.0) for label in labels] for probabilities in frame["choice_probs"]],
        dtype=float,
    )
    matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
    lookup = {label: index for index, label in enumerate(labels)}
    gold = np.array([lookup.get(value, -1) for value in frame["gold"]], dtype=int)
    if (gold < 0).any():
        raise ValueError("probability row has an invalid gold label")
    prediction = matrix.argmax(axis=1)
    confidence = matrix.max(axis=1)
    correct = prediction == gold
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(len(gold)), gold] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(matrix > 0, matrix * np.log(matrix), 0.0), axis=1) / math.log(len(labels))
    return {
        "confidence": confidence,
        "correct": correct,
        "entropy_normalized": entropy,
        "brier": ((matrix - one_hot) ** 2).sum(axis=1),
        "nll": -np.log(np.clip(matrix[np.arange(len(gold)), gold], 1e-6, 1.0)),
    }


def ece10(confidence: np.ndarray, correct: np.ndarray) -> float:
    value = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence >= low) & ((confidence <= high) if high == 1 else (confidence < high))
        if selected.any():
            value += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(value)


def auroc(confidence: np.ndarray, correct: np.ndarray) -> float:
    if correct.all() or (~correct).all():
        return float("nan")
    return float(stats.mannwhitneyu(confidence[correct], confidence[~correct], alternative="two-sided").statistic / (correct.sum() * (~correct).sum()))


def add_probability_metrics(
    row: dict,
    prefix: str,
    base: pd.DataFrame,
    adapted: pd.DataFrame,
    labels: Sequence[str],
) -> None:
    joint = base["logprob_eligible"].to_numpy() & adapted["logprob_eligible"].to_numpy()
    base_arrays = probability_arrays(base[joint], labels)
    adapted_arrays = probability_arrays(adapted[joint], labels)
    row[f"{prefix}_n_joint"] = int(joint.sum())
    for metric in ("confidence", "entropy_normalized", "brier", "nll"):
        row[f"{prefix}_{metric}_base"] = float(np.mean(base_arrays[metric]))
        row[f"{prefix}_{metric}_adapted"] = float(np.mean(adapted_arrays[metric]))
        row[f"{prefix}_{metric}_delta"] = row[f"{prefix}_{metric}_adapted"] - row[f"{prefix}_{metric}_base"]
    for metric, function in (("ece10", ece10), ("auroc", auroc)):
        row[f"{prefix}_{metric}_base"] = function(base_arrays["confidence"], base_arrays["correct"])
        row[f"{prefix}_{metric}_adapted"] = function(adapted_arrays["confidence"], adapted_arrays["correct"])
        row[f"{prefix}_{metric}_delta"] = row[f"{prefix}_{metric}_adapted"] - row[f"{prefix}_{metric}_base"]
    shared_wrong = ~base_arrays["correct"] & ~adapted_arrays["correct"]
    row[f"{prefix}_shared_wrong_n"] = int(shared_wrong.sum())
    for metric in ("confidence", "entropy_normalized"):
        base_value = float(np.mean(base_arrays[metric][shared_wrong])) if shared_wrong.any() else float("nan")
        adapted_value = float(np.mean(adapted_arrays[metric][shared_wrong])) if shared_wrong.any() else float("nan")
        row[f"{prefix}_shared_wrong_{metric}_base"] = base_value
        row[f"{prefix}_shared_wrong_{metric}_adapted"] = adapted_value
        row[f"{prefix}_shared_wrong_{metric}_delta"] = adapted_value - base_value


def rate(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def add_medabstain_metrics(row: dict, frame: pd.DataFrame, adapted: str, base: str) -> None:
    slices = {}
    for condition in MA_CONDITIONS:
        base_rows, adapted_rows = aligned(frame, adapted, base, "medabstain", condition)
        slices[condition] = (base_rows, adapted_rows)
        root = f"ma_{condition.lower()}"
        add_binary(row, f"{root}_accuracy", base_rows["correct"], adapted_rows["correct"])
        add_binary(row, f"{root}_abstention", base_rows["abstained"], adapted_rows["abstained"])
        for role, rows in (("base", base_rows), ("adapted", adapted_rows)):
            answered = ~rows["abstained"].to_numpy()
            row[f"{root}_coverage_{role}"] = rate(answered)
            row[f"{root}_selective_accuracy_{role}"] = rate(rows["correct"].to_numpy()[answered])
        row[f"{root}_parse_failure_base"] = rate(base_rows["parse_failed"].to_numpy())
        row[f"{root}_parse_failure_adapted"] = rate(adapted_rows["parse_failed"].to_numpy())
        labels = list("ABCDE") if condition in {"NA", "NAP"} else list("ABCDEF")
        add_probability_metrics(row, f"{root}_cal", base_rows, adapted_rows, labels)

    for role, index in (("base", 0), ("adapted", 1)):
        na = slices["NA"][index]["correct"].to_numpy()
        nap = slices["NAP"][index]["correct"].to_numpy()
        abstain_a = slices["A"][index]["abstained"].to_numpy()
        abstain_ap = slices["AP"][index]["abstained"].to_numpy()
        key_drop = rate(na) - rate(nap)
        response = rate(abstain_ap) - rate(abstain_a)
        lost = na & ~nap
        held = na & nap
        row[f"ma_key_drop_{role}"] = key_drop
        row[f"ma_abstention_response_{role}"] = response
        row[f"ma_responsiveness_{role}"] = response / key_drop if key_drop > 0 else float("nan")
        row[f"ma_informedness_j_{role}"] = response
        row[f"ma_targeting_{role}"] = rate(abstain_ap[lost]) - rate(abstain_ap[held]) if lost.any() and held.any() else float("nan")
    for metric in ("key_drop", "abstention_response", "responsiveness", "informedness_j", "targeting"):
        row[f"ma_{metric}_delta"] = row[f"ma_{metric}_adapted"] - row[f"ma_{metric}_base"]

    base_a, adapted_a = slices["A"]
    base_abstained = base_a["abstained"].to_numpy()
    adapted_abstained = adapted_a["abstained"].to_numpy()
    base_correct = base_a["correct"].to_numpy()
    adapted_correct = adapted_a["correct"].to_numpy()
    newly_answered = base_abstained & ~adapted_abstained
    always_answered = ~base_abstained & ~adapted_abstained
    newly_abstained = ~base_abstained & adapted_abstained
    both_abstained = base_abstained & adapted_abstained
    row["ma_newly_answered_n"] = int(newly_answered.sum())
    row["ma_newly_abstained_n"] = int(newly_abstained.sum())
    row["ma_always_answered_n"] = int(always_answered.sum())
    row["ma_both_abstained_n"] = int(both_abstained.sum())
    row["ma_newly_answered_accuracy"] = rate(adapted_correct[newly_answered])
    row["ma_always_answered_adapted_accuracy"] = rate(adapted_correct[always_answered])
    row["ma_newly_answered_accuracy_gap"] = row["ma_newly_answered_accuracy"] - row["ma_always_answered_adapted_accuracy"]
    row["ma_newly_abstained_base_accuracy"] = rate(base_correct[newly_abstained])
    row["ma_always_answered_base_accuracy"] = rate(base_correct[always_answered])
    row["ma_newly_abstained_base_accuracy_gap"] = row["ma_newly_abstained_base_accuracy"] - row["ma_always_answered_base_accuracy"]
    row["ma_underconfidence_delta"] = row["ma_a_abstention_delta"]

    base_na, adapted_na = slices["NA"]
    for slice_name in ("A", "AP"):
        base_slice, adapted_slice = slices[slice_name]
        base_abstained_slice = base_slice["abstained"].to_numpy()
        adapted_abstained_slice = adapted_slice["abstained"].to_numpy()
        partitions = {
            "always_answered": ~base_abstained_slice & ~adapted_abstained_slice,
            "newly_answered": base_abstained_slice & ~adapted_abstained_slice,
            "newly_abstained": ~base_abstained_slice & adapted_abstained_slice,
            "both_abstained": base_abstained_slice & adapted_abstained_slice,
        }
        for name, selected in partitions.items():
            prefix = f"ma_{slice_name.lower()}_na_{name}"
            row[f"{prefix}_n"] = int(selected.sum())
            row[f"{prefix}_accuracy_base"] = rate(base_na["correct"].to_numpy()[selected])
            row[f"{prefix}_accuracy_adapted"] = rate(adapted_na["correct"].to_numpy()[selected])
            row[f"{prefix}_accuracy_delta"] = row[f"{prefix}_accuracy_adapted"] - row[f"{prefix}_accuracy_base"]
            if slice_name == "A":
                row[f"ma_na_{name}_n"] = row[f"{prefix}_n"]
                row[f"ma_na_{name}_accuracy_base"] = row[f"{prefix}_accuracy_base"]
                row[f"ma_na_{name}_accuracy_adapted"] = row[f"{prefix}_accuracy_adapted"]
                row[f"ma_na_{name}_accuracy_delta"] = row[f"{prefix}_accuracy_delta"]


def add_clindet_metrics(row: dict, frame: pd.DataFrame, adapted: str, base: str) -> None:
    slices = {}
    for condition in CD_CONDITIONS:
        base_rows, adapted_rows = aligned(frame, adapted, base, "clindet", condition)
        slices[condition] = (base_rows, adapted_rows)
        add_binary(row, f"cd_{condition}_accuracy", base_rows["correct"], adapted_rows["correct"])
        add_binary(row, f"cd_{condition}_unable_rate", base_rows["abstained"], adapted_rows["abstained"])
        row[f"cd_{condition}_parse_failure_base"] = rate(base_rows["parse_failed"].to_numpy())
        row[f"cd_{condition}_parse_failure_adapted"] = rate(adapted_rows["parse_failed"].to_numpy())
        add_probability_metrics(row, f"cd_{condition}_cal", base_rows, adapted_rows, list(("Met", "Not met", "Unable to determine")))
    for role, index in (("base", 0), ("adapted", 1)):
        determinable = slices["incomplete_determinable"][index]["abstained"].to_numpy()
        undeterminable = slices["incomplete_undeterminable"][index]["abstained"].to_numpy()
        row[f"cd_underconfidence_{role}"] = rate(determinable)
        row[f"cd_overconfidence_{role}"] = rate(~undeterminable)
        row[f"cd_informedness_j_{role}"] = rate(undeterminable) - rate(determinable)
    for metric in ("underconfidence", "overconfidence", "informedness_j"):
        row[f"cd_{metric}_delta"] = row[f"cd_{metric}_adapted"] - row[f"cd_{metric}_base"]


def build_pair_metrics(items: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    model_lookup = models.set_index("model_id")
    rows = []
    for adapted_row in models[models["role"].eq("adapted")].sort_values("model_id").itertuples():
        base = adapted_row.base_model_id
        row = {
            "pair_id": f"{adapted_row.model_id}__{base}",
            "adapted_model_id": adapted_row.model_id,
            "base_model_id": base,
            "param_b": adapted_row.param_b,
            "parent_alignment_state": adapted_row.parent_alignment_state,
            "post_training_methods": adapted_row.post_training_methods,
            "stage_count": adapted_row.stage_count,
            "sft_data_category": adapted_row.sft_data_category,
            "recipe_family": adapted_row.recipe_family,
            "preference_family": adapted_row.preference_family,
            "preference_objective": adapted_row.preference_objective,
            "parameterization": adapted_row.parameterization,
            "effective_batch_disclosed": adapted_row.effective_batch_disclosed,
            "max_effective_batch_sequences": adapted_row.max_effective_batch_sequences,
            "strict_lineage": adapted_row.strict_lineage,
            "ma_behavior_quality": adapted_row.ma_behavior_quality,
            "ma_probability_quality": adapted_row.ma_probability_quality,
            "cd_behavior_quality": adapted_row.cd_behavior_quality,
            "contam_medabstain": adapted_row.contam_medabstain,
            "contam_clindet": adapted_row.contam_clindet,
        }
        if base not in model_lookup.index:
            raise ValueError(f"missing base model {base}")
        add_medabstain_metrics(row, items, adapted_row.model_id, base)
        add_clindet_metrics(row, items, adapted_row.model_id, base)
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 26:
        raise ValueError(f"expected 26 pairs, got {len(result)}")
    aliases = {
        "ma_ap_minus_a_abstention_j_delta": "ma_informedness_j_delta",
        "cd_abstention_j_delta": "cd_informedness_j_delta",
        "ma_a_newly_answered_gap": "ma_newly_answered_accuracy_gap",
        "ma_a_newly_abstained_gap": "ma_newly_abstained_base_accuracy_gap",
        "ma_na_shared_wrong_confidence_delta": "ma_na_cal_shared_wrong_confidence_delta",
    }
    for public_name, internal_name in aliases.items():
        result[public_name] = result[internal_name]
    for p_column in [column for column in result if column.endswith("_p_item")]:
        result[p_column.removesuffix("_p_item") + "_q_item"] = bh_fdr(result[p_column])
    return result


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    output = np.full(len(p_values), np.nan)
    keep = np.isfinite(p_values)
    selected = p_values[keep]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(ranked))
    restored[order] = np.clip(adjusted, 0, 1)
    output[np.flatnonzero(keep)] = restored
    return output


def wilcoxon(values: Sequence[float]) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    nonzero = clean[clean != 0]
    if not len(nonzero):
        return {"n": 0, "effect": float(np.median(clean)) if len(clean) else float("nan"), "p_value": float("nan"), "rank_biserial": float("nan")}
    result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = stats.rankdata(np.abs(nonzero))
    effect_size = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    return {"n": len(nonzero), "effect": float(np.median(clean)), "p_value": float(result.pvalue), "rank_biserial": float(effect_size)}


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[str], seed: int) -> tuple[float, float, float]:
    values_array = np.asarray(values, dtype=float)
    clusters_array = np.asarray(clusters, dtype=object)
    keep = np.isfinite(values_array)
    values_array = values_array[keep]
    clusters_array = clusters_array[keep]
    unique, cluster_index = np.unique(clusters_array, return_inverse=True)
    members = [np.flatnonzero(cluster_index == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_CLUSTER_BOOTSTRAP)
    for replicate in range(N_CLUSTER_BOOTSTRAP):
        selected_clusters = rng.integers(0, len(unique), len(unique))
        selected_rows = np.concatenate(
            [rng.choice(members[index], size=len(members[index]), replace=True) for index in selected_clusters]
        )
        estimates[replicate] = np.median(values_array[selected_rows])
    low, high = np.percentile(estimates, [2.5, 97.5])
    p_value = min(1.0, 2 * min(np.mean(estimates <= 0), np.mean(estimates >= 0)))
    return float(low), float(high), float(p_value)


def permutation_pearson(x: Sequence[float], y: Sequence[float], seed: int) -> tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[keep], y_array[keep]
    observed = float(stats.pearsonr(x_array, y_array).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(N_PERMUTATION):
        permuted = float(stats.pearsonr(x_array, rng.permutation(y_array)).statistic)
        exceed += abs(permuted) >= abs(observed)
    return observed, (exceed + 1) / (N_PERMUTATION + 1)


def cochran_q(deltas: Sequence[float], standard_errors: Sequence[float]) -> tuple[float, float, float]:
    deltas_array = np.asarray(deltas, dtype=float)
    errors_array = np.asarray(standard_errors, dtype=float)
    keep = np.isfinite(deltas_array) & np.isfinite(errors_array) & (errors_array > 0)
    deltas_array, errors_array = deltas_array[keep], errors_array[keep]
    weights = 1 / errors_array**2
    weighted_mean = np.sum(weights * deltas_array) / np.sum(weights)
    statistic = float(np.sum(weights * (deltas_array - weighted_mean) ** 2))
    degrees = len(deltas_array) - 1
    p_value = float(stats.chi2.sf(statistic, degrees))
    i_squared = max(0.0, (statistic - degrees) / statistic * 100) if statistic else 0.0
    return statistic, p_value, i_squared


def claim_registry(pairs: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("ma_ap_abstention", "ma_ap_abstention_delta"),
        ("cd_undeterminable_unable", "cd_incomplete_undeterminable_unable_rate_delta"),
        ("ma_informedness", "ma_ap_minus_a_abstention_j_delta"),
        ("cd_informedness", "cd_abstention_j_delta"),
        ("newly_answered_gap", "ma_a_newly_answered_gap"),
        ("newly_abstained_gap", "ma_a_newly_abstained_gap"),
        ("ma_underconfidence", "ma_underconfidence_delta"),
        ("cd_underconfidence", "cd_underconfidence_delta"),
        ("ma_na_ece10", "ma_na_cal_ece10_delta"),
        ("ma_na_entropy", "ma_na_cal_entropy_normalized_delta"),
        ("ma_na_confidence", "ma_na_cal_confidence_delta"),
        ("ma_na_shared_wrong_confidence", "ma_na_shared_wrong_confidence_delta"),
        ("ma_na_auroc", "ma_na_cal_auroc_delta"),
    ]
    rows = []
    for claim_id, metric in definitions:
        result = wilcoxon(pairs[metric])
        low, high, cluster_p = cluster_bootstrap(
            pairs[metric], pairs["base_model_id"], notebook_seed("cluster", metric)
        )
        rows.append(
            {
                "claim_id": claim_id,
                "metric": metric,
                "test": "wilcoxon",
                **result,
                "cluster_ci_low": low,
                "cluster_ci_high": high,
                "cluster_p_value": cluster_p,
            }
        )
    correlation, correlation_p = permutation_pearson(
        pairs["ma_na_accuracy_delta"],
        pairs["ma_ap_abstention_delta"],
        notebook_seed("capcaution", "ma_na_accuracy_delta", "ma_ap_abstention_delta", "absolute"),
    )
    rows.append(
        {
            "claim_id": "ma_capability_caution_correlation",
            "metric": "ma_capability_caution_pearson_r",
            "test": "permutation_pearson",
            "n": len(pairs),
            "effect": correlation,
            "p_value": correlation_p,
            "rank_biserial": float("nan"),
            "cluster_ci_low": float("nan"),
            "cluster_ci_high": float("nan"),
            "cluster_p_value": float("nan"),
        }
    )
    standard_errors = (pairs["ma_ap_abstention_ci_high"] - pairs["ma_ap_abstention_ci_low"]) / (2 * 1.96)
    q_statistic, q_p, i_squared = cochran_q(pairs["ma_ap_abstention_delta"], standard_errors)
    rows.append(
        {
            "claim_id": "ma_heterogeneity",
            "metric": "ma_ap_abstention_i_squared",
            "test": "cochran_q",
            "n": int(np.isfinite(standard_errors).sum()),
            "effect": i_squared,
            "p_value": q_p,
            "rank_biserial": float("nan"),
            "cluster_ci_low": float("nan"),
            "cluster_ci_high": float("nan"),
            "cluster_p_value": float("nan"),
            "q_statistic": q_statistic,
        }
    )
    claims = pd.DataFrame(rows)
    claims["q_value"] = bh_fdr(claims["p_value"])
    claims["supported"] = (
        claims["q_value"].lt(0.05)
        & (claims["cluster_p_value"].lt(0.05) | claims["cluster_p_value"].isna())
    )
    return claims


def sensitivity_tables(pairs: pd.DataFrame, out: Path) -> None:
    parse_columns = [column for column in pairs if "parse_failure" in column]
    pairs[["pair_id", *parse_columns]].to_csv(out / "parse_failure_sensitivity.csv", index=False)
    key_metrics = [
        "ma_ap_abstention_delta",
        "cd_incomplete_undeterminable_unable_rate_delta",
        "ma_informedness_j_delta",
        "cd_informedness_j_delta",
    ]
    relative = pairs[["pair_id", "base_model_id", *key_metrics]].copy()
    for metric in key_metrics:
        base_column = metric.removesuffix("_delta") + "_base"
        if base_column in pairs:
            relative[f"{metric}_relative"] = pairs[metric] / pairs[base_column].replace(0, np.nan)
    relative.to_csv(out / "relative_change_sensitivity.csv", index=False)
    clean = pairs[~pairs["contam_medabstain"].eq("yes")]
    rows = []
    for metric in key_metrics:
        result = wilcoxon(clean[metric])
        rows.append({"metric": metric, "subset": "exclude_confirmed_medabstain_overlap", **result})
    pd.DataFrame(rows).to_csv(out / "contamination_sensitivity.csv", index=False)
    thin = pairs[
        pairs["ma_newly_answered_n"].lt(5) | pairs["ma_newly_abstained_n"].lt(5)
    ][["pair_id", "ma_newly_answered_n", "ma_newly_abstained_n"]]
    thin.to_csv(out / "thin_cell_sensitivity.csv", index=False)


def moderator_tables(pairs: pd.DataFrame, out: Path) -> None:
    rows = []
    for moderator in ("parent_alignment_state", "recipe_family", "preference_family", "parameterization"):
        for group, values in pairs.groupby(moderator, dropna=False)["ma_ap_abstention_delta"]:
            rows.append(
                {
                    "moderator": moderator,
                    "group": group,
                    "n": len(values),
                    "median_ma_ap_abstention_delta": float(np.median(values)),
                }
            )
    pd.DataFrame(rows).to_csv(out / "moderators.csv", index=False)
    association_rows = []
    for benchmark, capability, caution in (
        ("medabstain", "ma_na_accuracy_delta", "ma_ap_abstention_delta"),
        ("clindet", "cd_complete_accuracy_delta", "cd_incomplete_undeterminable_unable_rate_delta"),
    ):
        effect, p_value = permutation_pearson(
            pairs[capability],
            pairs[caution],
            notebook_seed("capcaution", capability, caution, "absolute"),
        )
        association_rows.append({"benchmark": benchmark, "association": "capability_caution_pearson", "effect": effect, "p_value": p_value, "n": len(pairs)})
    rho, p_value = stats.spearmanr(pairs["param_b"], pairs["ma_ap_abstention_delta"])
    association_rows.append({"benchmark": "medabstain", "association": "parameter_size_spearman", "effect": rho, "p_value": p_value, "n": len(pairs)})
    associations = pd.DataFrame(association_rows)
    associations["q_value"] = bh_fdr(associations["p_value"])
    associations.to_csv(out / "associations.csv", index=False)


def save_figure(fig: plt.Figure, out: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def identity_panel(ax: plt.Axes, pairs: pd.DataFrame, prefix: str, title: str) -> None:
    base = pairs[f"{prefix}_base"]
    adapted = pairs[f"{prefix}_adapted"]
    limits = [min(base.min(), adapted.min()), max(base.max(), adapted.max())]
    ax.plot(limits, limits, color="0.7", linewidth=1)
    ax.scatter(base, adapted, s=25, alpha=0.8)
    ax.set(title=title, xlabel="Base", ylabel="Adapted")


def dumbbell_panel(
    ax: plt.Axes,
    pairs: pd.DataFrame,
    prefix: str,
    title: str,
    *,
    q_column: str | None = None,
) -> None:
    ordered = pairs.sort_values(f"{prefix}_adapted")
    x = np.arange(len(ordered))
    for index, (_, row) in enumerate(ordered.iterrows()):
        significant = q_column is not None and row.get(q_column, 1.0) < 0.05
        ax.plot(
            [index, index],
            [row[f"{prefix}_base"], row[f"{prefix}_adapted"]],
            color="0.35",
            linewidth=1.5 if significant else 0.8,
            linestyle="-" if significant else "--",
            alpha=0.9 if significant else 0.55,
        )
    ax.scatter(x, ordered[f"{prefix}_base"], facecolors="white", edgecolors="#35608d", label="Base", zorder=3)
    ax.scatter(x, ordered[f"{prefix}_adapted"], color="#c44e52", marker="s", label="Adapted", zorder=3)
    ax.set_xticks(x, ordered["adapted_model_id"].str.replace("-", " "), rotation=90, fontsize=5)
    ax.set(title=title, ylabel="Rate")


def make_figures(pairs: pd.DataFrame, out: Path) -> None:
    labels = pairs["adapted_model_id"].str.replace("-", " ")
    y = np.arange(len(pairs))
    recipes = sorted(pairs["recipe_family"].unique())
    recipe_colors = {recipe: plt.get_cmap("tab20")(index / max(len(recipes), 1)) for index, recipe in enumerate(recipes)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    for ax, metric, title in (
        (axes[0], "ma_ap_abstention", "MedAbstain AP abstention"),
        (axes[1], "cd_incomplete_undeterminable_unable_rate", "ClinDet undeterminable Unable"),
    ):
        ax.axvline(0, color="0.7", linewidth=1)
        for index, (_, row) in enumerate(pairs.iterrows()):
            significant = row[f"{metric}_q_item"] < 0.05
            ax.scatter(
                row[f"{metric}_delta"] * 100,
                index,
                s=28,
                facecolor=recipe_colors[row["recipe_family"]] if significant else "white",
                edgecolor=recipe_colors[row["recipe_family"]],
                linewidth=1,
            )
        ax.set(title=title, xlabel="Adapted minus base (pp)")
    axes[0].set_yticks(y, labels, fontsize=7)
    save_figure(fig, out, "fig01_abstention_change_absolute_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    dumbbell_panel(axes[0], pairs, "ma_ap_abstention", "MedAbstain AP", q_column="ma_ap_abstention_q_item")
    dumbbell_panel(
        axes[1],
        pairs,
        "cd_incomplete_undeterminable_unable_rate",
        "ClinDet undeterminable",
        q_column="cd_incomplete_undeterminable_unable_rate_q_item",
    )
    axes[0].legend(frameon=False, fontsize=7)
    save_figure(fig, out, "fig03_abstention_levels_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    dumbbell_panel(axes[0], pairs, "ma_informedness_j", "MedAbstain informedness J")
    dumbbell_panel(axes[1], pairs, "cd_informedness_j", "ClinDet informedness J")
    axes[0].axhline(0, color="0.7", linewidth=0.8)
    axes[1].axhline(0, color="0.7", linewidth=0.8)
    save_figure(fig, out, "fig04_informedness_j_paper.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    limit = max(pairs["ma_key_drop_base"].max(), pairs["ma_key_drop_adapted"].max()) * 1.05
    axes[0].plot([0, limit], [0, limit], "--", color="0.6")
    axes[0].scatter(pairs["ma_key_drop_adapted"], pairs["ma_abstention_response_adapted"], label="Adapted")
    axes[0].scatter(pairs["ma_key_drop_base"], pairs["ma_abstention_response_base"], marker="x", label="Base")
    axes[0].set(xlabel="Retained-key drop", ylabel="Abstention response", title="Answer shift vs response")
    axes[0].legend(frameon=False, fontsize=7)
    dumbbell_panel(axes[1], pairs, "ma_responsiveness", "Responsiveness ratio")
    axes[1].axhline(1, color="0.6", linestyle="--")
    dumbbell_panel(axes[2], pairs, "ma_targeting", "Targeted abstention")
    axes[2].axhline(0, color="0.6", linestyle="--")
    save_figure(fig, out, "fig05_shift_response_absolute_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, new_column, always_column, count_column, title in (
        (axes[0], "ma_newly_answered_accuracy", "ma_always_answered_adapted_accuracy", "ma_newly_answered_n", "Newly answered: adapted accuracy"),
        (axes[1], "ma_newly_abstained_base_accuracy", "ma_always_answered_base_accuracy", "ma_newly_abstained_n", "Newly abstained: base accuracy"),
    ):
        ordered = pairs.sort_values(new_column)
        x = np.arange(len(ordered))
        for index, (_, row) in enumerate(ordered.iterrows()):
            ax.plot([index, index], [row[always_column] * 100, row[new_column] * 100], color="0.45", linewidth=1)
            ax.annotate(str(int(row[count_column])), (index, 2), ha="center", fontsize=4.5)
        ax.scatter(x, ordered[always_column] * 100, facecolor="white", edgecolor="#35608d", s=22, label="Always answered")
        ax.scatter(x, ordered[new_column] * 100, color="#c44e52", marker="s", s=22, label="Changed partition")
        ax.axhline(20, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, ordered["adapted_model_id"], rotation=90, fontsize=5)
        ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0].legend(frameon=False, fontsize=7)
    save_figure(fig, out, "fig06_partition_accuracy_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    partitions = ("always_answered", "newly_answered", "newly_abstained", "both_abstained")
    for ax, slice_name in zip(axes, ("a", "ap")):
        x = np.arange(len(partitions))
        for role, offset, marker, color in (("base", -0.12, "o", "#35608d"), ("adapted", 0.12, "s", "#c44e52")):
            medians = [pairs[f"ma_{slice_name}_na_{name}_accuracy_{role}"].median() * 100 for name in partitions]
            ax.plot(x + offset, medians, marker=marker, color=color, label=role.title())
        ax.axhline(20, color="0.7", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, ["Always", "New answer", "New abstain", "Both abstain"], rotation=18)
        ax.set(title=f"Partitions defined on {slice_name.upper()}", ylabel="Median NA accuracy (%)", ylim=(0, 100))
    axes[0].legend(frameon=False)
    save_figure(fig, out, "fig07_na_partition_accuracy_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    identity_panel(axes[0], pairs, "ma_na_cal_ece10", "MedAbstain NA ECE10")
    identity_panel(axes[1], pairs, "cd_complete_cal_ece10", "ClinDet complete ECE10")
    save_figure(fig, out, "fig09_ece_identity_paper.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, prefix, title in zip(
        axes.flat,
        ["ma_a_cal_ece10", "ma_nap_cal_ece10", "ma_ap_cal_ece10", "cd_incomplete_determinable_cal_ece10", "cd_incomplete_undeterminable_cal_ece10"],
        ["MA A", "MA NAP", "MA AP", "CD determinable", "CD undeterminable"],
    ):
        identity_panel(ax, pairs, prefix, title)
    axes.flat[-1].axis("off")
    save_figure(fig, out, "fig10_ece_identity_appendix_paper.pdf")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    metrics = ["confidence", "entropy_normalized", "shared_wrong_confidence", "auroc"]
    stage_colors = {1: "#2e75c9", 2: "#e02b3c", 3: "#f07d1a"}
    for ax, metric in zip(axes.flat, metrics):
        roots = (
            ("ma_na_cal", "o", "MedAbstain NA"),
            ("cd_complete_cal", "^", "ClinDet complete"),
        )
        values = []
        for root, marker, benchmark in roots:
            prefix = f"{root}_{metric}"
            base_values = pairs[f"{prefix}_base"]
            adapted_values = pairs[f"{prefix}_adapted"]
            values.extend([*base_values.dropna(), *adapted_values.dropna()])
            for _, row in pairs.iterrows():
                if pd.isna(row[f"{prefix}_base"]) or pd.isna(row[f"{prefix}_adapted"]):
                    continue
                ax.scatter(
                    row[f"{prefix}_base"],
                    row[f"{prefix}_adapted"],
                    marker=marker,
                    s=30,
                    color=stage_colors.get(int(row["stage_count"]), "0.5"),
                    alpha=0.8,
                )
        limits = [min(values), max(values)]
        ax.plot(limits, limits, "--", color="0.65", linewidth=0.8)
        ax.set(title=metric.replace("_", " ").title(), xlabel="Base", ylabel="Adapted")
    save_figure(fig, out, "fig11_confidence_grid_paper.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, x_metric, y_metric, title in (
        (axes[0], "ma_na_accuracy_delta", "ma_ap_abstention_delta", "MedAbstain"),
        (axes[1], "cd_complete_accuracy_delta", "cd_incomplete_undeterminable_unable_rate_delta", "ClinDet"),
    ):
        ax.axhline(0, color="0.8")
        ax.axvline(0, color="0.8")
        ax.scatter(pairs[x_metric] * 100, pairs[y_metric] * 100)
        ax.set(xlabel="Capability change (pp)", ylabel="Caution change (pp)", title=title)
    save_figure(fig, out, "fig12_capability_vs_caution_absolute_paper.pdf")

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [(name, group["ma_ap_abstention_delta"].dropna() * 100) for name, group in pairs.groupby("recipe_family")]
    for index, (name, values) in enumerate(groups):
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(values, np.full(len(values), index) + jitter, color=recipe_colors[name], s=30)
        ax.plot([np.median(values), np.median(values)], [index - 0.25, index + 0.25], color="0.2", linewidth=1.5)
    ax.axvline(0, color="0.7")
    ax.set_yticks(range(len(groups)), [f"{name} (n={len(values)})" for name, values in groups], fontsize=7)
    ax.set(xlabel="MA AP abstention change (pp)", title="Fine-tuning recipe families")
    save_figure(fig, out, "fig13_moderators_panel_a_paper.pdf")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0, color="0.7")
    ax.scatter(pairs["param_b"], pairs["ma_ap_abstention_delta"] * 100)
    ax.set(xscale="log", xlabel="Parameters (billions, log scale)", ylabel="MA AP change (pp)", title="Scale sensitivity")
    save_figure(fig, out, "fig14_size_scaling_absolute_paper.pdf")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(22.45, color="0.35", linestyle="--", linewidth=1, label="Clinician-judged answerable share")
    ax.scatter(pairs["ma_ap_coverage_adapted"] * 100, pairs["ma_ap_selective_accuracy_adapted"] * 100, s=32)
    ax.set(
        xlabel="Coverage on MedAbstain AP (%)",
        ylabel="Selective accuracy on answered AP items (%)",
        title="Fine-tuned coverage exceeds the answerable share",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.legend(frameon=False, fontsize=7)
    save_figure(fig, out, "fig18_ap_answerable_rebuttal_paper.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    recipe_cohort = pairs[
        pairs["strict_lineage"]
        & pairs["ma_behavior_quality"]
        & pairs["adapted_model_id"].ne("meerkat-7b")
    ]
    batch_rows = recipe_cohort.dropna(subset=["max_effective_batch_sequences"])
    x_values = batch_rows["ma_ap_abstention_delta"] * 100
    y_values = np.log2(batch_rows["max_effective_batch_sequences"])
    kendall = stats.kendalltau(x_values, y_values)
    ax.scatter(x_values, y_values, s=36)
    ax.set(
        xlabel="MA AP abstention change (pp)",
        ylabel="Effective batch size (sequences, log2)",
        title=f"Reported sequence batches: n={len(batch_rows)}, Kendall tau={kendall.statistic:.3f}, p={kendall.pvalue:.3f}",
    )
    save_figure(fig, out, "fig19_recipe_ap_batch_kendall_paper.pdf")

    recipe_cohort = pairs[
        pairs["strict_lineage"]
        & pairs["ma_behavior_quality"]
        & pairs["adapted_model_id"].ne("meerkat-7b")
    ]
    preference_labels = {
        "none": "No preferences",
        "refusal_reduction": "Refusal reduction",
        "undisclosed": "Not disclosed",
        "safety_redteam": "Other (Safety, Correctness)",
        "verifier_correctness": "Other (Safety, Correctness)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cohort, title in (
        (axes[0], recipe_cohort, "All models"),
        (axes[1], recipe_cohort[recipe_cohort["recipe_family"].isin({"clinical_text_sft", "mixed_instruction_sft", "exam_sft"})], "Non-CoT SFT only"),
    ):
        labels_series = cohort["preference_family"].map(preference_labels)
        groups = [(name, cohort.loc[labels_series.eq(name), "ma_informedness_j_delta"] * 100) for name in preference_labels.values() if labels_series.eq(name).any()]
        for index, (name, values) in enumerate(groups):
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(values, np.full(len(values), index) + jitter, s=28)
            ax.scatter(np.median(values), index, marker="x", color="#c44e52", s=65)
        ax.axvline(0, color="0.7", linestyle="--")
        ax.set_yticks(range(len(groups)), [f"{name} (n={len(values)})" for name, values in groups], fontsize=7)
        ax.set(title=title, xlabel="MA informedness change (pp)")
    save_figure(fig, out, "fig20_recipe_informedness_by_preference_paper.pdf")

    recipe3 = {
        "cpt_only": "CPT-only",
        "clinical_text_sft": "Clinical Text SFT",
        "mixed_instruction_sft": "Clinical Text SFT",
        "exam_sft": "Clinical Text SFT",
        "exam_plus_cot_sft": "Exam + CoT SFT",
    }
    recipe_rows = recipe_cohort.assign(recipe3=recipe_cohort["recipe_family"].map(recipe3)).dropna(subset=["recipe3"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, metric, title in (
        (axes[0], "ma_na_accuracy_delta", "Capability: MA NA accuracy"),
        (axes[1], "ma_ap_abstention_delta", "Caution: MA AP abstention"),
    ):
        groups = [(name, group[metric] * 100) for name, group in recipe_rows.groupby("recipe3")]
        for index, (name, values) in enumerate(groups):
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(values, np.full(len(values), index) + jitter, s=28)
            ax.scatter(np.median(values), index, marker="x", color="#c44e52", s=65)
        ax.axvline(0, color="0.7", linestyle="--")
        ax.set_yticks(range(len(groups)), [f"{name} (n={len(values)})" for name, values in groups], fontsize=7)
        ax.set(title=title, xlabel="Adapted minus base (pp)")
    save_figure(fig, out, "fig21_recipe_accuracy_caution_by_recipe3_paper.pdf")


def check_claims(claims: pd.DataFrame) -> None:
    expected = {claim["id"]: claim for claim in STUDY["expected_claims"]}
    actual = claims.set_index("claim_id")
    failures = []
    for claim_id, contract in expected.items():
        if claim_id not in actual.index:
            failures.append(f"missing {claim_id}")
            continue
        value = float(actual.loc[claim_id, "effect"])
        if not math.isclose(value, contract["expected"], abs_tol=contract["tolerance"], rel_tol=0):
            failures.append(f"{claim_id}: expected {contract['expected']}, got {value}")
    if failures:
        raise RuntimeError("claim parity failed:\n" + "\n".join(failures))


def command_analyze(args: argparse.Namespace) -> None:
    items_path = args.items.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = load_models()
    items = load_items(items_path)
    pairs = build_pair_metrics(items, models)
    claims = claim_registry(pairs)
    pairs.to_csv(out / "pair_metrics.csv", index=False)
    claims.to_csv(out / "claim_registry.csv", index=False)
    sensitivity_tables(pairs, out)
    moderator_tables(pairs, out)
    if not args.no_figures:
        make_figures(pairs, out)
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "provenance.json"
    }
    provenance = {
        "study_id": STUDY["study_id"],
        "items": {"path": str(items_path), "sha256": sha256(items_path), "rows": len(items)},
        "pair_rows": len(pairs),
        "claim_rows": len(claims),
        "outputs": output_hashes,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.check:
        check_claims(claims)
        missing = set(STUDY["required_figures"]) - set(output_hashes)
        if missing and not args.no_figures:
            raise RuntimeError(f"missing figures: {sorted(missing)}")
    print(f"wrote {len(pairs)} pairs and {len(claims)} claims to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
