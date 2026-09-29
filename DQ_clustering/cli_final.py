"""Publication-facing command-line interface for DQ clustering workflows."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .algorithms import (
    ClusterResult,
    _class1_cohesion_stats,
    _class1_purity,
    _contingency,
    _kl_margins,
    _permutation_pvalue,
    _plot_confusion_and_margins,
    _safe_sklearn_metrics,
    format_class_value,
)
from .bootstrap import BootstrapAnalyzer, BootstrapSummary
from .data_loader import (
    DEFAULT_DIR as METADATA_DEFAULT_DIR,
    PATTERN as METADATA_PATTERN,
    DatasetLoader,
    FoldxEnergyDataset,
    Metadata,
    collect_label_class_pairs,
    open_default_xlsx,
)
from .engines import ClusteringEngine, PreparedData
from .parameters import get_parameter_defaults
from .diagnostic_plots import DiagnosticPlotter
from .heatmaps import plot_value_heatmap, plot_value_heatmap_triptych


PARAM_DEFAULTS = get_parameter_defaults()

METRIC_CHOICES = (
    "stability",
    "intraclashes_A",
    "intraclashes_B",
    "interaction_AB",
    "stability_A",
    "stability_B",
)
CLUSTER_MODE_CHOICES = ("bregman", "spectral", "hierarchical", "dbscan")
MOL_PART_CHOICES = ("whole", "CT_off", "CT_TMD_off", "TMD_only")
ENERGY_SUMMARY_CHOICES = ("stats", "mode", "ensemble")


@dataclass
class AnalysisContext:
    root: Path
    metric: str
    mol_part: str
    metadata: Metadata
    datasets: list[FoldxEnergyDataset]
    classes_num: np.ndarray
    parent_labels: list[str]
    suffix_labels: list[str]
    cluster_reports_dir: Path


@dataclass
class ClusteringArtifacts:
    engine: ClusteringEngine
    prepared: PreparedData
    cluster_result: ClusterResult
    plotter: DiagnosticPlotter


def _load_special_labels_from_excel(
    row_idx: int = 0,
    *,
    default_dir: Path | None = None,
    pattern: str = METADATA_PATTERN,
) -> list[str]:
    """Load labels with class==1 from the metadata workbook."""
    try:
        df = open_default_xlsx(
            default_dir=default_dir or METADATA_DEFAULT_DIR,
            pattern=pattern,
            launch=False,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load metadata workbook: {exc}") from exc
    if df is None:
        raise SystemExit("Metadata workbook preview was requested instead of loading data.")
    _parents, _suffixes, pairs = collect_label_class_pairs(df, row_idx=row_idx)
    specials: list[str] = []
    for label, cls in pairs:
        try:
            value = float(cls)
        except Exception:
            continue
        if value == 1.0:
            specials.append(label)
    return specials


def _maybe_load_special_labels(args: argparse.Namespace) -> list[str]:
    """Best-effort helper for highlight labels used in plot-only workflows."""
    try:
        return _load_special_labels_from_excel(
            row_idx=0,
            default_dir=args.metadata_dir,
            pattern=args.metadata_pattern,
        )
    except SystemExit as exc:
        print(f"Warning: special-label highlighting disabled ({exc}).")
        return []


def _summarise_metric(values: np.ndarray, stat: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if stat == "median":
        return float(np.median(arr))
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        return float(q3 - q1)
    if stat == "mode":
        try:
            edges = np.histogram_bin_edges(arr, bins="fd")
        except Exception:
            edges = np.histogram_bin_edges(arr, bins="auto")
        if edges.size < 2:
            return float(arr[0])
        hist, edges = np.histogram(arr, bins=edges)
        idx = int(np.argmax(hist))
        return float((edges[idx] + edges[idx + 1]) / 2.0)
    raise ValueError(f"Unsupported stat: {stat!r}")


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log(sum(exp(values)))."""
    if values.size == 0:
        return float("-inf")
    vmax = float(np.max(values))
    if not np.isfinite(vmax):
        return float("-inf")
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def _ensemble_free_energy_exact(energies: np.ndarray, *, rt: float = 1.0) -> float:
    """Normalized ensemble free energy using <exp(-E_i/RT)>."""
    if not np.isfinite(rt) or rt <= 0.0:
        raise ValueError("RT must be a positive finite number.")
    energies = np.asarray(energies, dtype=np.float64)
    energies = energies[np.isfinite(energies)]
    if energies.size == 0:
        return float("nan")
    logZ = _logsumexp(-energies / float(rt)) - float(np.log(energies.size))
    if not np.isfinite(logZ):
        return float("nan")
    return float(-float(rt) * logZ)


def _ensemble_free_energy_weighted(
    energies: np.ndarray,
    weights: np.ndarray,
    *,
    rt: float = 1.0,
    normalize_weights: bool = True,
) -> float:
    """Weighted ensemble free energy using normalized weights by default."""
    if not np.isfinite(rt) or rt <= 0.0:
        raise ValueError("RT must be a positive finite number.")
    energies = np.asarray(energies, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if energies.shape != weights.shape:
        raise ValueError("energies and weights must have the same shape.")
    mask = np.isfinite(energies) & np.isfinite(weights) & (weights > 0.0)
    energies = energies[mask]
    weights = weights[mask]
    if energies.size == 0:
        return float("nan")
    log_weights = np.log(weights)
    if normalize_weights:
        log_weights = log_weights - _logsumexp(log_weights)
    logZ = _logsumexp(log_weights - (energies / float(rt)))
    if not np.isfinite(logZ):
        return float("nan")
    return float(-float(rt) * logZ)


def _ensemble_weights_from_ranks(ranks: np.ndarray, mode: str) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.float64)
    if mode == "inv_rank":
        return 1.0 / np.maximum(ranks, 1.0)
    if mode == "inv_rank_complement":
        return 1.0 / np.maximum(1001.0 - ranks, 1.0)
    raise ValueError(f"Unsupported rank weight mode: {mode!r}")


def _align_series_and_ranks(series: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-aligned series/rank arrays with matching shape."""
    series = np.asarray(series, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.float64)
    if series.shape != ranks.shape:
        n = min(series.size, ranks.size)
        series = series[:n]
        ranks = ranks[:n]
    mask = np.isfinite(series) & np.isfinite(ranks)
    return series[mask], ranks[mask]


def _parse_binary_label(value: object) -> int | None:
    """Parse class_value into a binary label (0/1) or return None."""
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if value == 0.0:
        return 0
    if value == 1.0:
        return 1
    return None


def _roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for binary labels using the trapezoidal rule."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    if y.size == 0:
        return float("nan")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores)
    y_sorted = y[order]
    score_sorted = scores[order]

    tps = 0
    fps = 0
    tpr: list[float] = [0.0]
    fpr: list[float] = [0.0]
    prev_score = None
    for yi, si in zip(y_sorted, score_sorted):
        if prev_score is None:
            prev_score = si
        elif si != prev_score:
            tpr.append(tps / n_pos)
            fpr.append(fps / n_neg)
            prev_score = si
        if yi == 1:
            tps += 1
        else:
            fps += 1
    tpr.append(tps / n_pos)
    fpr.append(fps / n_neg)

    auc = 0.0
    for idx in range(1, len(tpr)):
        auc += (fpr[idx] - fpr[idx - 1]) * (tpr[idx] + tpr[idx - 1]) * 0.5
    return float(auc)


def _best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    lower_is_positive: bool,
) -> tuple[float, float]:
    """Return the best F1 and threshold for a one-dimensional decision rule."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    y = y[mask]
    scores = scores[mask]
    if y.size == 0:
        return float("nan"), float("nan")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    best_f1 = -1.0
    best_thr = float("nan")
    for threshold in np.unique(scores):
        if lower_is_positive:
            pred = (scores <= threshold).astype(int)
        else:
            pred = (scores >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(threshold)
    return best_f1, best_thr


def _select_metric_series(
    metric: str,
    energies: np.ndarray,
    extras: dict[str, np.ndarray],
) -> np.ndarray | None:
    if metric == "stability":
        return energies
    return extras.get(metric)


def _param(name: str):
    try:
        return PARAM_DEFAULTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Parameter '{name}' is not defined in {Path(__file__).parent / 'parameters.log'}."
        ) from exc


def build_class_arrays(datasets: Sequence[FoldxEnergyDataset]) -> np.ndarray:
    """Build a numeric class array from dataset class values."""
    class_strings = [format_class_value(ds.class_value) for ds in datasets]
    try:
        return np.array([int(value) for value in class_strings], dtype=int)
    except Exception:
        return np.array([1 if value == "1" else 0 for value in class_strings], dtype=int)


def summarise_clusters(
    labels: np.ndarray,
    datasets: Sequence[FoldxEnergyDataset],
    classes_num: np.ndarray,
) -> None:
    """Print class composition summaries for inferred clusters."""
    cls_one = format_class_value(1)
    cls_zero = format_class_value(0)

    cluster_class_counts: dict[int, Counter[str]] = {}
    class_totals: Counter[str] = Counter()
    class_one_labels_by_cluster: dict[int, list[str]] = {}

    cluster_ids_present = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    for cluster_id in cluster_ids_present:
        members = [ds for lab, ds in zip(labels, datasets, strict=True) if int(lab) == cluster_id]
        counts = Counter(format_class_value(member.class_value) for member in members)
        cluster_class_counts[cluster_id] = counts
        class_totals.update(counts)
        class_one_labels_by_cluster[cluster_id] = sorted(
            member.label
            for member in members
            if format_class_value(member.class_value) == cls_one
        )

    if not class_totals:
        return

    def pct(part: int, total: int) -> float:
        return 0.0 if total == 0 else (part / total) * 100.0

    print("\nPer-cluster class composition:")
    for cluster_id in cluster_ids_present:
        counts = cluster_class_counts.get(cluster_id, Counter())
        total_members = sum(counts.values())
        class1_count = counts.get(cls_one, 0)
        class0_count = counts.get(cls_zero, 0)
        other_count = total_members - class1_count - class0_count
        line = (
            f"  Cluster {cluster_id}: total={total_members} | "
            f"class {cls_one}={pct(class1_count, total_members):5.1f}% ({class1_count}), "
            f"class {cls_zero}={pct(class0_count, total_members):5.1f}% ({class0_count})"
        )
        if other_count > 0:
            line += f", other={pct(other_count, total_members):5.1f}% ({other_count})"
        print(line)

    print("\nPer-class cluster allocation:")
    for cls_label in (cls_one, cls_zero):
        total = class_totals.get(cls_label, 0)
        if total == 0:
            continue
        distribution = ", ".join(
            f"{cluster_id}:{pct(cluster_class_counts.get(cluster_id, Counter()).get(cls_label, 0), total):5.1f}% "
            f"({cluster_class_counts.get(cluster_id, Counter()).get(cls_label, 0)})"
            for cluster_id in cluster_ids_present
        )
        print(f"  Class {cls_label}: total={total} -> {distribution}")

    dominant_cluster, dominant_counts = max(
        cluster_class_counts.items(),
        key=lambda item: item[1].get(cls_one, 0),
        default=(-1, Counter()),
    )
    dominant_size = dominant_counts.get(cls_one, 0)
    other_labels = [
        label
        for cluster_id, labels_in_cluster in sorted(class_one_labels_by_cluster.items())
        if cluster_id != dominant_cluster
        for label in labels_in_cluster
    ]

    print(
        f"\nClass {cls_one} dominant cluster: {dominant_cluster} "
        f"({dominant_size} member{'s' if dominant_size != 1 else ''})"
    )
    if other_labels:
        print(f"Class {cls_one} labels outside the dominant cluster (n={len(other_labels)}):")
        for label in other_labels:
            print(f"  {label}")
    else:
        print(f"All class {cls_one} labels fall within the dominant cluster.")


def print_evidence(
    labels: np.ndarray,
    centroids: np.ndarray,
    normalized_data: np.ndarray,
    classes_num: np.ndarray,
    output_dir: Path,
    permutations: int,
    seed: int | None,
    suffix: str = "",
) -> None:
    """Print class/cluster evidence metrics and save one diagnostic panel."""
    contingency = _contingency(labels, classes_num)
    dominant, total, purity = _class1_purity(contingency, class1_col=1)
    scores = _safe_sklearn_metrics(labels, classes_num)
    p_value = _permutation_pvalue(labels, classes_num, iters=permutations, seed=seed)
    cohesion = _class1_cohesion_stats(labels, classes_num, permutations, seed=seed)
    margins = _kl_margins(normalized_data, centroids, 1e-12)

    print("\n=== Class-cluster evidence ===")
    if total == 0:
        print("Class-1 in dominant cluster: no class-1 samples present.")
    else:
        print(f"Class-1 in dominant cluster: {dominant}/{total} ({100.0 * purity:.1f}%)")
    print(f"ARI={scores['ARI']:.3f}  NMI={scores['NMI']:.3f}  Permutation p-value={p_value:.4g}")
    if np.isnan(cohesion["observed"]):
        print("Class-1 pair cohesion: not enough class-1 members to evaluate.")
    else:
        pairs = int(cohesion["pairs"])
        plural = "s" if pairs != 1 else ""
        print(
            f"Class-1 pair cohesion ({pairs} pair{plural}): "
            f"observed={cohesion['observed']:.3f} "
            f"(random mean≈{cohesion['expected_mean']:.3f}, "
            f"95%-tile≈{cohesion['expected_q95']:.3f}, "
            f"p-value={cohesion['p_value']:.4g})"
        )
    print(
        f"Median ΔKL margin: class-1={np.median(margins[classes_num == 1]):.4f}  "
        f"class-0={np.median(margins[classes_num == 0]):.4f}"
    )

    diag_name = f"class_cluster_evidence_{suffix}.pdf" if suffix else "class_cluster_evidence.pdf"
    diag_path = output_dir / diag_name
    _plot_confusion_and_margins(contingency, margins, classes_num, diag_path)
    print(f"Saved diagnostics to {diag_path}")


def summarise_bootstrap(summary: BootstrapSummary) -> None:
    """Print compact summaries of bootstrap stability outputs."""
    print("\n=== Bootstrap stability ===")
    print(f"Replicates: {len(summary.stats)}")
    if not summary.stats:
        return

    mode_counts = Counter(str(stat["mode"]) for stat in summary.stats)
    print(
        "Modes encountered: "
        + ", ".join(f"{mode}:{count}" for mode, count in sorted(mode_counts.items()))
    )

    dominant_counter = Counter(cluster_id for cluster_id in summary.dominant_cluster_ids if cluster_id >= 0)
    if dominant_counter:
        total_defined = sum(dominant_counter.values())
        print("\nDominant class-1 cluster frequency:")
        for cluster_id, count in sorted(dominant_counter.items()):
            fraction = count / total_defined if total_defined else float("nan")
            print(f"  cluster {cluster_id}: {count}/{total_defined} ({fraction:.3f})")
        undefined = len(summary.stats) - total_defined
        if undefined > 0:
            print(f"  undefined dominant cluster in {undefined} replicates.")
    else:
        print("\nDominant class-1 cluster frequency: insufficient data.")

    share_vals = np.asarray(summary.class1_share_samples, dtype=np.float64)
    share_vals = share_vals[np.isfinite(share_vals)]
    if share_vals.size:
        print(
            "Class-1 dominant share:\n"
            f"  mean={share_vals.mean():.3f}, "
            f"std={share_vals.std(ddof=1) if share_vals.size > 1 else 0.0:.3f}, "
            f"median={np.median(share_vals):.3f}, "
            f"q05={np.quantile(share_vals, 0.05):.3f}, "
            f"q95={np.quantile(share_vals, 0.95):.3f}"
        )

    cohesion_vals = np.asarray([stat["cohesion"] for stat in summary.stats], dtype=np.float64)
    cohesion_vals = cohesion_vals[np.isfinite(cohesion_vals)]
    if cohesion_vals.size:
        print(
            "Class-1 pair cohesion:\n"
            f"  mean={cohesion_vals.mean():.3f}, "
            f"std={cohesion_vals.std(ddof=1) if cohesion_vals.size > 1 else 0.0:.3f}, "
            f"median={np.median(cohesion_vals):.3f}, "
            f"q05={np.quantile(cohesion_vals, 0.05):.3f}, "
            f"q95={np.quantile(cohesion_vals, 0.95):.3f}"
        )


def write_bootstrap_counts(
    summary: BootstrapSummary,
    datasets: Sequence[FoldxEnergyDataset],
    path: Path,
) -> None:
    """Persist per-dataset bootstrap counts for later plot-only workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "label\ttrials\tdominant_hits\tdominant_trials\tenergy_dom_hits\tenergy_dom_trials\t"
            "class1_dom_hits\tclass1_dom_trials\tclass0_same_hits\tclass0_same_trials\t"
            "dist_sum\tdist_sum_sq\tdist_trials\n"
        )
        for idx, ds in enumerate(datasets):
            handle.write(
                f"{ds.label}\t"
                f"{int(summary.trials[idx])}\t"
                f"{int(summary.dominant_hits[idx])}\t{int(summary.dominant_trials[idx])}\t"
                f"{int(summary.energy_dom_hits[idx])}\t{int(summary.energy_dom_trials[idx])}\t"
                f"{int(summary.class1_dom_hits[idx])}\t{int(summary.class1_dom_trials[idx])}\t"
                f"{int(summary.class0_same_hits[idx])}\t{int(summary.class0_same_trials[idx])}\t"
                f"{float(summary.dist_sum[idx])}\t{float(summary.dist_sum_sq[idx])}\t{int(summary.dist_trials[idx])}\n"
            )


def load_bootstrap_counts(
    counts_path: Path,
    datasets: Sequence[FoldxEnergyDataset],
) -> BootstrapSummary:
    """Load per-dataset counts and rebuild a minimal BootstrapSummary."""
    labels = [ds.label for ds in datasets]
    hits: list[int] = []
    trials: list[int] = []
    dominant_trials: list[int] = []
    class1_hits: list[int] = []
    class1_trials: list[int] = []
    class0_hits: list[int] = []
    class0_trials: list[int] = []
    energy_hits: list[int] = []
    energy_trials: list[int] = []
    dist_sum: list[float] = []
    dist_sum_sq: list[float] = []
    dist_trials: list[int] = []

    with counts_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        col_index = {name: idx for idx, name in enumerate(header)}
        rows = {line.split("\t", 1)[0]: line.strip().split("\t") for line in handle if line.strip()}

    for label in labels:
        fields = rows.get(label)
        if not fields or len(fields) < 7:
            hits.append(0)
            trials.append(0)
            dominant_trials.append(0)
            class1_hits.append(0)
            class1_trials.append(0)
            class0_hits.append(0)
            class0_trials.append(0)
            energy_hits.append(0)
            energy_trials.append(0)
            dist_sum.append(0.0)
            dist_sum_sq.append(0.0)
            dist_trials.append(0)
            continue

        def _get(name: str, default: str = "0") -> str:
            idx = col_index.get(name)
            if idx is None or idx >= len(fields):
                return default
            return fields[idx]

        dominant_hit = int(_get("dominant_hits", "0"))
        dominant_trial = int(_get("dominant_trials", "0"))
        total_trials = int(_get("trials", str(dominant_trial)))
        energy_hit = int(_get("energy_dom_hits", "0"))
        energy_trial = int(_get("energy_dom_trials", "0"))
        class1_hit = int(_get("class1_dom_hits", "0"))
        class1_trial = int(_get("class1_dom_trials", "0"))
        class0_hit = int(_get("class0_same_hits", "0"))
        class0_trial = int(_get("class0_same_trials", "0"))
        distance_sum = float(_get("dist_sum", "0"))
        distance_sum_sq = float(_get("dist_sum_sq", "0"))
        distance_trial = int(_get("dist_trials", "0"))

        hits.append(dominant_hit)
        trials.append(total_trials)
        dominant_trials.append(dominant_trial)
        energy_hits.append(energy_hit)
        energy_trials.append(energy_trial)
        class1_hits.append(class1_hit)
        class1_trials.append(class1_trial)
        class0_hits.append(class0_hit)
        class0_trials.append(class0_trial)
        dist_sum.append(distance_sum)
        dist_sum_sq.append(distance_sum_sq)
        dist_trials.append(distance_trial)

    return BootstrapSummary(
        stats=[],
        dominant_hits=np.asarray(hits, dtype=np.int64),
        dominant_trials=np.asarray(dominant_trials, dtype=np.int64),
        trials=np.asarray(trials, dtype=np.int64),
        energy_dom_hits=np.asarray(energy_hits, dtype=np.int64),
        energy_dom_trials=np.asarray(energy_trials, dtype=np.int64),
        class1_dom_hits=np.asarray(class1_hits, dtype=np.int64),
        class1_dom_trials=np.asarray(class1_trials, dtype=np.int64),
        class0_same_hits=np.asarray(class0_hits, dtype=np.int64),
        class0_same_trials=np.asarray(class0_trials, dtype=np.int64),
        dist_sum=np.asarray(dist_sum, dtype=np.float64),
        dist_sum_sq=np.asarray(dist_sum_sq, dtype=np.float64),
        dist_trials=np.asarray(dist_trials, dtype=np.int64),
        class1_share_samples=[],
        class0_share_samples=[],
        dominant_cluster_ids=[],
        log_lines=[],
    )


def _validate_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"Root path does not exist or is not a directory: {resolved}")
    return resolved


def _validate_rank_bounds(rank_min: int, rank_max: int) -> None:
    if rank_min <= 0 or rank_max <= 0:
        raise SystemExit("Rank bounds must be positive integers.")
    if rank_min > rank_max:
        raise SystemExit("--rank-min must be <= --rank-max.")


def _build_plot_suffix(*parts: object) -> str:
    return "_".join(str(part).strip() for part in parts if str(part).strip())


def _report_missing_labels(missing_labels: set[str]) -> None:
    if not missing_labels:
        return
    preview = ", ".join(sorted(missing_labels)[:10])
    if len(missing_labels) > 10:
        preview += ", ..."
    print(f"Skipped {len(missing_labels)} labels missing class metadata: {preview}")


def _filter_axis_labels(
    datasets: Sequence[FoldxEnergyDataset],
    metadata: Metadata,
) -> tuple[list[str], list[str]]:
    used_label_set = {ds.label for ds in datasets}
    used_parent_set: set[str] = set()
    used_suffix_set: set[str] = set()
    for label in used_label_set:
        if "_" in label:
            parent, suffix = label.split("_", 1)
            used_parent_set.add(parent)
            used_suffix_set.add(suffix)
        else:
            used_parent_set.add(label)
    parent_labels = [parent for parent in metadata.parents if parent in used_parent_set]
    suffix_labels = [suffix for suffix in metadata.suffixes if suffix in used_suffix_set]
    return parent_labels, suffix_labels


def _remap_metric_datasets(
    datasets: Sequence[FoldxEnergyDataset],
    metric: str,
) -> list[FoldxEnergyDataset]:
    """Project datasets onto a selected metric while keeping ranks aligned."""
    if metric == "stability":
        return list(datasets)

    remapped: list[FoldxEnergyDataset] = []
    missing_metric: list[str] = []
    for ds in datasets:
        column = ds.extra_energies.get(metric)
        if column is None:
            missing_metric.append(ds.label)
            continue
        values = np.asarray(column, dtype=np.float64)
        mask = np.isfinite(values)
        if not np.any(mask):
            missing_metric.append(ds.label)
            continue
        remapped.append(
            FoldxEnergyDataset(
                dqa_name=ds.dqa_name,
                dqb_name=ds.dqb_name,
                energies=values[mask],
                ranks=np.asarray(ds.ranks)[mask],
                extra_energies={key: np.asarray(series)[mask] for key, series in ds.extra_energies.items()},
                class_value=ds.class_value,
            )
        )
    if missing_metric:
        raise SystemExit(
            "Requested metric '" + metric + "' missing for datasets: " + ", ".join(missing_metric)
        )
    return remapped


def _load_analysis_context(args: argparse.Namespace) -> AnalysisContext:
    root = _validate_root(args.root)
    _validate_rank_bounds(args.rank_min, args.rank_max)

    print(
        f"Loading datasets: root={root} metric={args.metric} "
        f"mol_part={args.mol_part} ranks=[{args.rank_min}, {args.rank_max}]"
    )

    loader = DatasetLoader(root)
    metadata = loader.load_metadata(
        excel_dir=args.metadata_dir,
        pattern=args.metadata_pattern,
    )
    label_map = {label: value for label, value in metadata.label_classes}
    missing_labels: set[str] = set()
    datasets = loader.load_datasets(
        class_map=label_map,
        missing_labels=missing_labels,
        log_file=None,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        mol_part=args.mol_part,
    )
    _report_missing_labels(missing_labels)
    datasets = _remap_metric_datasets(datasets, args.metric)
    classes_num = build_class_arrays(datasets)
    parent_labels, suffix_labels = _filter_axis_labels(datasets, metadata)
    cluster_reports_dir = root / "reports" / "clustering" / args.metric
    return AnalysisContext(
        root=root,
        metric=args.metric,
        mol_part=args.mol_part,
        metadata=metadata,
        datasets=datasets,
        classes_num=classes_num,
        parent_labels=parent_labels,
        suffix_labels=suffix_labels,
        cluster_reports_dir=cluster_reports_dir,
    )


def _run_clustering(context: AnalysisContext, args: argparse.Namespace) -> ClusteringArtifacts:
    context.cluster_reports_dir.mkdir(parents=True, exist_ok=True)
    plot_suffix = _build_plot_suffix(context.mol_part, args.cluster_mode, args.heatmap_suffix)

    engine = ClusteringEngine(args.bins, smoothing=args.smoothing)
    prepared = engine.build_distributions(context.datasets)
    sigma_param = None if np.isnan(args.sigma) else float(args.sigma)
    dbscan_param = None if np.isnan(args.dbscan_eps) else float(args.dbscan_eps)

    print(f"Running clustering: mode={args.cluster_mode} k={args.clusters} bins={args.bins}")
    cluster_result = engine.run(
        prepared,
        mode=args.cluster_mode,
        k=args.clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        knn=args.knn,
        sigma=sigma_param,
        linkage=args.hier_linkage,
        dbscan_eps=dbscan_param,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    for message in cluster_result.messages:
        print(message)

    if cluster_result.mode == "spectral":
        sigma_report = "median" if cluster_result.sigma_used is None else f"{cluster_result.sigma_used:g}"
        print("Clustering mode: spectral (Jensen-Shannon graph)")
        print(f"  kNN={cluster_result.knn_used}  sigma={sigma_report}")
    elif cluster_result.mode == "hierarchical":
        print(f"Clustering mode: hierarchical linkage ({args.hier_linkage})")
        print(f"  Formed {np.unique(cluster_result.labels).size} cluster(s); inertia unavailable.")
    elif cluster_result.mode == "dbscan":
        eps_report = "auto" if cluster_result.sigma_used is None else f"{cluster_result.sigma_used:g}"
        print("Clustering mode: DBSCAN (JS distance)")
        print(f"  eps={eps_report}; see summary messages for cluster/noise counts.")
    else:
        print("Clustering mode: bregman (KL k-means)")
        print(
            f"  Converged in {cluster_result.iterations} iteration(s); "
            f"total within-cluster KL divergence = {cluster_result.inertia:.6f}."
        )

    summarise_clusters(cluster_result.labels, context.datasets, context.classes_num)

    if args.evidence:
        print_evidence(
            cluster_result.labels,
            cluster_result.centroids,
            prepared.normalized,
            context.classes_num,
            context.cluster_reports_dir,
            permutations=args.permutations,
            seed=args.seed,
            suffix=plot_suffix,
        )

    plotter = DiagnosticPlotter(context.cluster_reports_dir, suffix=plot_suffix)
    plotter.plot_scatters(
        prepared.normalized,
        prepared.js_distances,
        cluster_result.labels,
        context.datasets,
        bin_edges=prepared.bin_edges,
        pca_panel_label="(d)" if cluster_result.mode == "spectral" else None,
    )
    plotter.plot_centroid_profiles(
        cluster_result.centroids,
        prepared.bin_edges,
        prepared.energy_min,
        prepared.energy_max,
        cluster_result.labels,
        context.classes_num,
        cluster_result.mode,
    )
    plotter.plot_distance_heatmap(
        prepared.normalized,
        cluster_result.centroids,
        cluster_result.labels,
        context.datasets,
        context.classes_num,
        context.parent_labels,
        context.suffix_labels,
    )
    return ClusteringArtifacts(
        engine=engine,
        prepared=prepared,
        cluster_result=cluster_result,
        plotter=plotter,
    )


def _resolve_output_path(
    base_output: Path | None,
    *,
    default_dir: Path,
    default_name: str,
    part: str | None = None,
    multiple_parts: bool = False,
) -> Path:
    path = base_output.expanduser() if base_output is not None else (default_dir / default_name)
    if multiple_parts and part is not None:
        return path.with_name(f"{path.stem}_{part}{path.suffix}")
    return path


def cmd_cluster(args: argparse.Namespace) -> None:
    context = _load_analysis_context(args)
    _run_clustering(context, args)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    if args.replicates <= 0:
        raise SystemExit("--replicates must be a positive integer.")
    context = _load_analysis_context(args)
    artifacts = _run_clustering(context, args)

    metrics_path = (
        args.metrics_file.expanduser()
        if args.metrics_file is not None
        else context.cluster_reports_dir / "bootstrap_metrics.tsv"
    )
    counts_path = (
        args.counts_file.expanduser()
        if args.counts_file is not None
        else metrics_path.with_name(metrics_path.stem + "_counts.tsv")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running bootstrap: replicates={args.replicates}")
    analyzer = BootstrapAnalyzer(artifacts.engine, context.datasets, context.classes_num)
    sigma_param = None if np.isnan(args.sigma) else float(args.sigma)
    dbscan_param = None if np.isnan(args.dbscan_eps) else float(args.dbscan_eps)
    summary = analyzer.run(
        artifacts.prepared,
        mode=artifacts.cluster_result.mode,
        k=args.clusters,
        n_bootstrap=args.replicates,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        knn=args.knn,
        sigma=sigma_param,
        linkage=args.hier_linkage,
        dbscan_eps=dbscan_param,
        dbscan_min_samples=args.dbscan_min_samples,
        log_path=metrics_path,
    )
    summarise_bootstrap(summary)
    write_bootstrap_counts(summary, context.datasets, counts_path)
    artifacts.plotter.plot_bootstrap_heatmaps(
        summary,
        context.datasets,
        context.classes_num,
        context.parent_labels,
        context.suffix_labels,
    )
    print(f"Saved bootstrap metrics to {metrics_path}")
    print(f"Saved bootstrap counts to {counts_path}")


def cmd_bootstrap_plot(args: argparse.Namespace) -> None:
    context = _load_analysis_context(args)
    counts_path = args.counts_file.expanduser().resolve()
    if not counts_path.is_file():
        raise SystemExit(f"Counts file not found: {counts_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else counts_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_suffix = _build_plot_suffix(context.mol_part, args.heatmap_suffix)
    plotter = DiagnosticPlotter(output_dir, suffix=plot_suffix)
    summary = load_bootstrap_counts(counts_path, context.datasets)
    plotter.plot_bootstrap_heatmaps(
        summary,
        context.datasets,
        context.classes_num,
        context.parent_labels,
        context.suffix_labels,
    )
    print(f"Replotted bootstrap heatmaps from {counts_path}")


def _load_energy_datasets(
    root: Path,
    *,
    metadata_dir: Path | None,
    metadata_pattern: str,
    rank_min: int,
    rank_max: int,
    mol_part: str,
) -> tuple[Metadata, list[FoldxEnergyDataset]]:
    loader = DatasetLoader(root)
    metadata = loader.load_metadata(excel_dir=metadata_dir, pattern=metadata_pattern)
    label_map = {label: value for label, value in metadata.label_classes}
    missing_labels: set[str] = set()
    datasets = loader.load_datasets(
        class_map=label_map,
        missing_labels=missing_labels,
        log_file=None,
        rank_min=rank_min,
        rank_max=rank_max,
        mol_part=mol_part,
    )
    _report_missing_labels(missing_labels)
    return metadata, datasets


def _run_energy_mode_summary(
    *,
    root: Path,
    metadata: Metadata,
    datasets: Sequence[FoldxEnergyDataset],
    metric: str,
    part: str,
    args: argparse.Namespace,
) -> None:
    values_mode: dict[str, float] = {}
    for ds in datasets:
        if args.exclude_dra and ds.label.startswith("DRA_"):
            continue
        series = _select_metric_series(metric, ds.energies, ds.extra_energies)
        if series is None:
            continue
        values_mode[ds.label] = _summarise_metric(series, "mode")

    if not values_mode:
        raise SystemExit(f"No values found for metric '{metric}' with mol-part '{part}'.")

    output_path = _resolve_output_path(
        args.output,
        default_dir=root / "reports" / "foldx" / metric,
        default_name=f"{metric}_mode_heatmap_{part}.pdf",
        part=part,
        multiple_parts=len(args.parts) > 1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = None
    if args.export_tsv:
        tsv_dir = args.tsv_dir or output_path.parent
        tsv_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = tsv_dir / f"{output_path.stem}_mode.tsv"

    plot_value_heatmap(
        values_mode,
        output_path,
        parents=metadata.parents,
        subs=metadata.suffixes,
        title=f"FoldX {metric} mode ({part})",
        cbar_label=f"{metric} (mode)",
        cmap="coolwarm",
        highlight_labels=_maybe_load_special_labels(args),
        export_tsv=args.export_tsv,
        tsv_path=tsv_path,
    )
    print(f"Saved mode heatmap to {output_path}")


def _run_energy_stats_summary(
    *,
    root: Path,
    metadata: Metadata,
    datasets: Sequence[FoldxEnergyDataset],
    metric: str,
    part: str,
    args: argparse.Namespace,
) -> None:
    values_median: dict[str, float] = {}
    values_iqr: dict[str, float] = {}
    values_ratio: dict[str, float] = {}
    values_mean: dict[str, float] = {}
    for ds in datasets:
        if args.exclude_dra and ds.label.startswith("DRA_"):
            continue
        series = _select_metric_series(metric, ds.energies, ds.extra_energies)
        if series is None:
            continue
        median_val = _summarise_metric(series, "median")
        iqr_val = _summarise_metric(series, "iqr")
        values_median[ds.label] = median_val
        values_iqr[ds.label] = iqr_val
        values_ratio[ds.label] = (
            median_val / iqr_val
            if np.isfinite(median_val) and np.isfinite(iqr_val) and iqr_val > 0.0
            else float("nan")
        )
        values_mean[ds.label] = _summarise_metric(series, "mean")

    if not values_median:
        raise SystemExit(f"No values found for metric '{metric}' with mol-part '{part}'.")

    output_path = _resolve_output_path(
        args.output,
        default_dir=root / "reports" / "foldx" / metric,
        default_name=f"{metric}_stats_heatmap_{part}.pdf",
        part=part,
        multiple_parts=len(args.parts) > 1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_paths = None
    if args.export_tsv:
        tsv_dir = args.tsv_dir or output_path.parent
        tsv_dir.mkdir(parents=True, exist_ok=True)
        tsv_paths = [
            tsv_dir / f"{output_path.stem}_median.tsv",
            tsv_dir / f"{output_path.stem}_iqr.tsv",
            tsv_dir / f"{output_path.stem}_median_iqr_ratio.tsv",
            tsv_dir / f"{output_path.stem}_mean.tsv",
        ]

    plot_value_heatmap_triptych(
        [values_median, values_iqr, values_ratio, values_mean],
        output_path,
        parents=metadata.parents,
        subs=metadata.suffixes,
        cmap="coolwarm",
        titles=[
            f"FoldX {metric} median ({part})",
            f"FoldX {metric} IQR ({part})",
            f"FoldX {metric} median/IQR ({part})",
            f"FoldX {metric} mean ({part})",
        ],
        cbar_labels=[
            f"{metric} (median)",
            f"{metric} (IQR)",
            "median / IQR",
            f"{metric} (mean)",
        ],
        highlight_labels=_maybe_load_special_labels(args),
        export_tsv=args.export_tsv,
        tsv_paths=tsv_paths,
    )
    print(f"Saved summary heatmap to {output_path}")


def _run_energy_ensemble_summary(
    *,
    root: Path,
    metadata: Metadata,
    datasets: Sequence[FoldxEnergyDataset],
    metric: str,
    part: str,
    args: argparse.Namespace,
) -> None:
    weight_mode = str(args.ensemble_weight or "none")
    normalize_weights = True if weight_mode != "none" else bool(args.ensemble_normalize_weights)
    tune_rt = bool(args.ensemble_tune_rt)
    base_rt = float(args.ensemble_rt)
    rt_grid = args.ensemble_rt_grid

    if tune_rt:
        if rt_grid is None:
            rt_grid = np.logspace(np.log10(0.2), np.log10(5.0), 13).tolist()
        grid = np.asarray([float(value) for value in rt_grid if float(value) > 0.0], dtype=np.float64)
        if grid.size == 0:
            raise SystemExit("Provide a positive --ensemble-RT-grid for --ensemble-tune-RT.")

        labeled: list[tuple[np.ndarray, np.ndarray, int]] = []
        for ds in datasets:
            if args.exclude_dra and ds.label.startswith("DRA_"):
                continue
            y_val = _parse_binary_label(ds.class_value)
            if y_val is None:
                continue
            series = _select_metric_series(metric, ds.energies, ds.extra_energies)
            if series is None:
                continue
            energies, ranks = _align_series_and_ranks(np.asarray(series, dtype=np.float64), ds.ranks)
            if energies.size == 0:
                continue
            labeled.append((energies, ranks, int(y_val)))
        if not labeled:
            raise SystemExit("No usable metric/rank/binary-label data found for ensemble free-energy tuning.")

        best_auc = -1.0
        best_rt = float("nan")
        best_direction_lower = True
        best_f1 = float("nan")
        best_thr = float("nan")
        print(f"Tuning ensemble RT: metric={metric} part={part} weights={weight_mode}")
        print("RT\tauc\tbest_f1\tthr\tdirection")
        for rt_val in grid:
            y_true: list[int] = []
            f_values: list[float] = []
            for energies, ranks, y_val in labeled:
                if weight_mode != "none":
                    weights = _ensemble_weights_from_ranks(ranks, weight_mode)
                    free_energy = _ensemble_free_energy_weighted(
                        energies,
                        weights,
                        rt=float(rt_val),
                        normalize_weights=normalize_weights,
                    )
                else:
                    free_energy = _ensemble_free_energy_exact(energies, rt=float(rt_val))
                if np.isfinite(free_energy):
                    y_true.append(y_val)
                    f_values.append(float(free_energy))
            if len(set(y_true)) < 2 or not f_values:
                continue

            y_arr = np.asarray(y_true, dtype=int)
            f_arr = np.asarray(f_values, dtype=np.float64)
            auc_lower = _roc_auc_binary(y_arr, -f_arr)
            auc_upper = _roc_auc_binary(y_arr, f_arr)
            if not np.isfinite(auc_lower) and not np.isfinite(auc_upper):
                continue
            lower_is_positive = True
            auc_use = auc_lower
            if np.isfinite(auc_upper) and (not np.isfinite(auc_lower) or auc_upper > auc_lower):
                lower_is_positive = False
                auc_use = auc_upper
            f1, threshold = _best_f1_threshold(y_arr, f_arr, lower_is_positive=lower_is_positive)
            direction = "low->pos" if lower_is_positive else "high->pos"
            print(f"{rt_val:.6g}\t{auc_use:.6g}\t{f1:.6g}\t{threshold:.6g}\t{direction}")
            if np.isfinite(auc_use) and auc_use > best_auc:
                best_auc = float(auc_use)
                best_rt = float(rt_val)
                best_direction_lower = bool(lower_is_positive)
                best_f1 = float(f1)
                best_thr = float(threshold)

        if not np.isfinite(best_rt):
            raise SystemExit("Failed to tune RT: insufficient labeled data after filtering.")
        print(
            f"Best RT={best_rt:.6g} auc={best_auc:.6g} best_f1={best_f1:.6g} "
            f"thr={best_thr:.6g} direction={'low->pos' if best_direction_lower else 'high->pos'}"
        )
        rt = float(best_rt)
    else:
        rt = float(base_rt)

    R_kcal_per_mol_K = 0.00198720425864083
    temperature_K = rt / R_kcal_per_mol_K

    values_F: dict[str, float] = {}
    values_median: dict[str, float] = {}
    values_mean: dict[str, float] = {}
    report_rows: list[tuple[str, float, float, float, int]] = []
    for ds in datasets:
        if args.exclude_dra and ds.label.startswith("DRA_"):
            continue
        series = _select_metric_series(metric, ds.energies, ds.extra_energies)
        if series is None:
            continue
        series, ranks = _align_series_and_ranks(np.asarray(series, dtype=np.float64), ds.ranks)
        if series.size == 0:
            continue
        if weight_mode != "none":
            weights = _ensemble_weights_from_ranks(ranks, weight_mode)
            free_energy = _ensemble_free_energy_weighted(
                series,
                weights,
                rt=rt,
                normalize_weights=normalize_weights,
            )
        else:
            free_energy = _ensemble_free_energy_exact(series, rt=rt)
        median_val = _summarise_metric(series, "median")
        mean_val = _summarise_metric(series, "mean")
        if np.isfinite(free_energy):
            values_F[ds.label] = float(free_energy)
            values_median[ds.label] = median_val
            values_mean[ds.label] = mean_val
            report_rows.append((ds.label, float(free_energy), float(median_val), float(mean_val), int(series.size)))

    if not values_F:
        raise SystemExit(f"No usable energies found for metric '{metric}' with mol-part '{part}'.")

    print(f"Ensemble free-energy summary: metric={metric} part={part} RT={rt:.6g} (T≈{temperature_K:.1f} K)")
    print("label\tF\tmedian\tmean\tn")
    for label, free_energy, median_val, mean_val, count in sorted(report_rows, key=lambda row: row[0]):
        print(f"{label}\t{free_energy:.6g}\t{median_val:.6g}\t{mean_val:.6g}\t{count}")

    output_path = _resolve_output_path(
        args.output,
        default_dir=root / "reports" / "foldx" / metric,
        default_name=f"{metric}_ensemble_free_energy_heatmap_{part}.pdf",
        part=part,
        multiple_parts=len(args.parts) > 1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_paths = None
    if args.export_tsv:
        tsv_dir = args.tsv_dir or output_path.parent
        tsv_dir.mkdir(parents=True, exist_ok=True)
        tsv_paths = [
            tsv_dir / f"{output_path.stem}_ensemble_free_energy.tsv",
            tsv_dir / f"{output_path.stem}_median.tsv",
            tsv_dir / f"{output_path.stem}_mean.tsv",
        ]

    plot_value_heatmap_triptych(
        [values_F, values_median, values_mean],
        output_path,
        parents=metadata.parents,
        subs=metadata.suffixes,
        cmap="coolwarm",
        highlight_labels=_maybe_load_special_labels(args),
        titles=[
            rf"Normalized ensemble free energy (T={temperature_K:.0f} K)",
            f"{metric} median",
            f"{metric} mean",
        ],
        cbar_labels=[
            rf"$F$ (T={temperature_K:.0f} K)",
            f"{metric} (median)",
            f"{metric} (mean)",
        ],
        export_tsv=args.export_tsv,
        tsv_paths=tsv_paths,
    )
    print(f"Saved ensemble free-energy heatmap to {output_path}")


def cmd_energy_summary(args: argparse.Namespace) -> None:
    root = _validate_root(args.root)
    _validate_rank_bounds(args.rank_min, args.rank_max)
    for part in args.parts:
        metadata, datasets = _load_energy_datasets(
            root,
            metadata_dir=args.metadata_dir,
            metadata_pattern=args.metadata_pattern,
            rank_min=args.rank_min,
            rank_max=args.rank_max,
            mol_part=part,
        )
        if args.summary == "mode":
            _run_energy_mode_summary(
                root=root,
                metadata=metadata,
                datasets=datasets,
                metric=args.metric,
                part=part,
                args=args,
            )
        elif args.summary == "ensemble":
            _run_energy_ensemble_summary(
                root=root,
                metadata=metadata,
                datasets=datasets,
                metric=args.metric,
                part=part,
                args=args,
            )
        else:
            _run_energy_stats_summary(
                root=root,
                metadata=metadata,
                datasets=datasets,
                metric=args.metric,
                part=part,
                args=args,
            )


def _add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "root",
        nargs="?",
        default=Path("."),
        type=Path,
        help="Root directory containing DQA/DQB folders (default: current directory).",
    )


def _add_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=f"Directory containing the metadata workbook (default: {METADATA_DEFAULT_DIR}).",
    )
    parser.add_argument(
        "--metadata-pattern",
        type=str,
        default=METADATA_PATTERN,
        help=f"Glob used to discover the metadata workbook (default: {METADATA_PATTERN!r}).",
    )


def _add_metric_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="stability",
        help="FoldX metric to operate on.",
    )


def _add_rank_args(
    parser: argparse.ArgumentParser,
    *,
    rank_min_default: int,
    rank_max_default: int,
) -> None:
    parser.add_argument(
        "--rank-min",
        type=int,
        default=rank_min_default,
        help=f"Minimum AlphaFold rank to include (default: {rank_min_default}).",
    )
    parser.add_argument(
        "--rank-max",
        type=int,
        default=rank_max_default,
        help=f"Maximum AlphaFold rank to include (default: {rank_max_default}).",
    )


def _add_mol_part_arg(parser: argparse.ArgumentParser, *, default: str) -> None:
    parser.add_argument(
        "--mol-part",
        choices=MOL_PART_CHOICES,
        default=default,
        help=f"FoldX molecule-part suffix to use (default: {default}).",
    )


def _add_clustering_args(
    parser: argparse.ArgumentParser,
    *,
    clusters_default: int,
    bins_default: int,
    max_iter_default: int,
    tol_default: float,
    seed_default: int | None,
    smoothing_default: float,
    cluster_mode_default: str,
    knn_default: int,
    sigma_default: float,
    hier_linkage_default: str,
    dbscan_eps_default: float,
    dbscan_min_samples_default: int,
    permutations_default: int,
) -> None:
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=clusters_default,
        help=f"Number of clusters to infer (default: {clusters_default}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=bins_default,
        help=f"Number of histogram bins (default: {bins_default}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=max_iter_default,
        help=f"Maximum solver iterations (default: {max_iter_default}).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=tol_default,
        help=f"Convergence tolerance (default: {tol_default}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help=f"Random seed (default: {seed_default}).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=smoothing_default,
        help=f"Additive histogram smoothing (default: {smoothing_default}).",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=CLUSTER_MODE_CHOICES,
        default=cluster_mode_default,
        help="Clustering backend to use.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=knn_default,
        help=f"kNN size for spectral clustering (default: {knn_default}).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=sigma_default,
        help="Spectral affinity bandwidth (default: automatic median heuristic).",
    )
    parser.add_argument(
        "--hier-linkage",
        choices=("average", "complete", "single", "weighted"),
        default=hier_linkage_default,
        help=f"Hierarchical linkage strategy (default: {hier_linkage_default}).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=dbscan_eps_default,
        help="DBSCAN epsilon in JS distance space.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=dbscan_min_samples_default,
        help=f"DBSCAN min_samples (default: {dbscan_min_samples_default}).",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Compute evidence metrics and save the class-cluster evidence panel.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=permutations_default,
        help=f"Permutation iterations for evidence statistics (default: {permutations_default}).",
    )
    parser.add_argument(
        "--heatmap-suffix",
        type=str,
        default="",
        help="Optional suffix appended to clustering plot filenames.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the publication CLI parser and return parsed arguments."""
    knn_default = int(_param("cli_knn"))
    bins_default = int(_param("cli_bins"))
    clusters_default = int(_param("cli_clusters"))
    max_iter_default = int(_param("cli_max_iter"))
    tol_default = float(_param("cli_tol"))
    smoothing_default = float(_param("cli_smoothing"))
    cluster_mode_default = str(_param("cli_cluster_mode"))
    sigma_default = float(_param("cli_sigma"))
    hier_linkage_default = str(_param("cli_hier_linkage"))
    dbscan_eps_default = float(_param("cli_dbscan_eps"))
    dbscan_min_samples_default = int(_param("cli_dbscan_min_samples"))
    permutations_default = int(_param("cli_permutations"))
    bootstrap_default = int(_param("cli_bootstrap"))
    seed_default = _param("cli_seed")
    rank_min_default = int(_param("cli_rank_min"))
    rank_max_default = int(_param("cli_rank_max"))
    mol_part_default = str(_param("cli_mol_part"))

    parser = argparse.ArgumentParser(
        description="Publication-facing CLI for DQ clustering workflows.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run clustering, print summaries, and save diagnostics.",
    )
    _add_root_argument(cluster_parser)
    _add_metadata_args(cluster_parser)
    _add_metric_arg(cluster_parser)
    _add_rank_args(
        cluster_parser,
        rank_min_default=rank_min_default,
        rank_max_default=rank_max_default,
    )
    _add_mol_part_arg(cluster_parser, default=mol_part_default)
    _add_clustering_args(
        cluster_parser,
        clusters_default=clusters_default,
        bins_default=bins_default,
        max_iter_default=max_iter_default,
        tol_default=tol_default,
        seed_default=seed_default,
        smoothing_default=smoothing_default,
        cluster_mode_default=cluster_mode_default,
        knn_default=knn_default,
        sigma_default=sigma_default,
        hier_linkage_default=hier_linkage_default,
        dbscan_eps_default=dbscan_eps_default,
        dbscan_min_samples_default=dbscan_min_samples_default,
        permutations_default=permutations_default,
    )
    cluster_parser.set_defaults(func=cmd_cluster)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Run clustering plus bootstrap stability analysis.",
    )
    _add_root_argument(bootstrap_parser)
    _add_metadata_args(bootstrap_parser)
    _add_metric_arg(bootstrap_parser)
    _add_rank_args(
        bootstrap_parser,
        rank_min_default=rank_min_default,
        rank_max_default=rank_max_default,
    )
    _add_mol_part_arg(bootstrap_parser, default=mol_part_default)
    _add_clustering_args(
        bootstrap_parser,
        clusters_default=clusters_default,
        bins_default=bins_default,
        max_iter_default=max_iter_default,
        tol_default=tol_default,
        seed_default=seed_default,
        smoothing_default=smoothing_default,
        cluster_mode_default=cluster_mode_default,
        knn_default=knn_default,
        sigma_default=sigma_default,
        hier_linkage_default=hier_linkage_default,
        dbscan_eps_default=dbscan_eps_default,
        dbscan_min_samples_default=dbscan_min_samples_default,
        permutations_default=permutations_default,
    )
    bootstrap_parser.add_argument(
        "--replicates",
        type=int,
        default=bootstrap_default,
        help=(
            "Number of bootstrap replicates. "
            f"Current default from parameters.log is {bootstrap_default}; set this explicitly for production runs."
        ),
    )
    bootstrap_parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Optional TSV output path for per-bootstrap summary metrics.",
    )
    bootstrap_parser.add_argument(
        "--counts-file",
        type=Path,
        default=None,
        help="Optional TSV output path for per-dataset bootstrap counts.",
    )
    bootstrap_parser.set_defaults(func=cmd_bootstrap)

    bootstrap_plot_parser = subparsers.add_parser(
        "bootstrap-plot",
        help="Rebuild bootstrap heatmaps from a saved counts TSV.",
    )
    _add_root_argument(bootstrap_plot_parser)
    _add_metadata_args(bootstrap_plot_parser)
    _add_metric_arg(bootstrap_plot_parser)
    _add_rank_args(
        bootstrap_plot_parser,
        rank_min_default=rank_min_default,
        rank_max_default=rank_max_default,
    )
    _add_mol_part_arg(bootstrap_plot_parser, default=mol_part_default)
    bootstrap_plot_parser.add_argument(
        "counts_file",
        type=Path,
        help="TSV produced by the bootstrap workflow.",
    )
    bootstrap_plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write replotted heatmaps (default: alongside the counts TSV).",
    )
    bootstrap_plot_parser.add_argument(
        "--heatmap-suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames.",
    )
    bootstrap_plot_parser.set_defaults(func=cmd_bootstrap_plot)

    energy_summary_parser = subparsers.add_parser(
        "energy-summary",
        help="Generate per-heterodimer heatmaps from raw FoldX energies.",
    )
    _add_root_argument(energy_summary_parser)
    _add_metadata_args(energy_summary_parser)
    _add_metric_arg(energy_summary_parser)
    _add_rank_args(
        energy_summary_parser,
        rank_min_default=rank_min_default,
        rank_max_default=rank_max_default,
    )
    energy_summary_parser.add_argument(
        "--parts",
        nargs="+",
        default=[mol_part_default],
        metavar="PART",
        choices=MOL_PART_CHOICES,
        help=f"One or more FoldX mol-part suffixes to summarize (default: {mol_part_default}).",
    )
    energy_summary_parser.add_argument(
        "--summary",
        choices=ENERGY_SUMMARY_CHOICES,
        default="stats",
        help="Summary mode: stats, mode, or ensemble.",
    )
    energy_summary_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Base output PDF path (part suffixes are appended automatically for multi-part runs).",
    )
    energy_summary_parser.add_argument(
        "--export-tsv",
        action="store_true",
        help="Export heatmap matrices as TSV files.",
    )
    energy_summary_parser.add_argument(
        "--tsv-dir",
        type=Path,
        default=None,
        help="Directory for TSV exports (default: alongside the output PDF).",
    )
    energy_summary_parser.add_argument(
        "--exclude-dra",
        action="store_true",
        help="Exclude DRA_* labels from the summary heatmaps.",
    )
    energy_summary_parser.add_argument(
        "--ensemble-RT",
        "--ensemble-rt",
        type=float,
        default=0.596,
        dest="ensemble_rt",
        help="RT value for ensemble free-energy summaries in kcal/mol (default: 0.596).",
    )
    energy_summary_parser.add_argument(
        "--ensemble-weight",
        choices=("none", "inv_rank", "inv_rank_complement"),
        default="none",
        help="Optional rank weighting scheme for ensemble free-energy summaries.",
    )
    energy_summary_parser.add_argument(
        "--ensemble-normalize-weights",
        action="store_true",
        help="Normalize ensemble weights so their sum is one. Rank weights are normalized by default.",
    )
    energy_summary_parser.add_argument(
        "--ensemble-tune-RT",
        "--ensemble-tune-rt",
        action="store_true",
        dest="ensemble_tune_rt",
        help="Tune RT against available binary labels before generating the heatmap.",
    )
    energy_summary_parser.add_argument(
        "--ensemble-RT-grid",
        "--ensemble-rt-grid",
        nargs="+",
        type=float,
        default=None,
        dest="ensemble_rt_grid",
        help="Explicit RT grid used when tuning ensemble free-energy summaries.",
    )
    energy_summary_parser.set_defaults(func=cmd_energy_summary)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the publication CLI."""
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
