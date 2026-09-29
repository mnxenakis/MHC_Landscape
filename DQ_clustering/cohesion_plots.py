"""Cohesion and raw-seed comparison plots kept for compatibility."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .mpl_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def plot_cohesion_summary(summary_csv: Path, output_pdf: Path | None = None) -> None:
    """Plot median and IQR bands of cohesion versus number of clusters.

    Expects a CSV with rows containing per-method statistics for k, including:
      columns: class1_median/q1/q3, class0_median/q1/q3, diff_median/q1/q3.

    Inputs:
        summary_csv: Path to summary CSV produced by cohesion scan aggregator.
        output_pdf: Optional explicit output path; inferred from input when None.

    Behaviour:
        - Builds per-method series for class1, class0, and diff = s1 - s0.
        - Renders three stacked panels with shared x (k).
        - Highlights method-specific k* where diff median is maximised.
        - Adds a custom y-tick at global max(diff median) if non-overlapping.
    """
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    by_method: Dict[str, Dict[str, List[dict[str, float]]]] = defaultdict(
        lambda: {"class1": [], "class0": [], "diff": []}
    )

    with summary_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            k = int(row["k"])
            for cls in ("class1", "class0", "diff"):
                median_key = f"{cls}_median"
                q1_key = f"{cls}_q1"
                q3_key = f"{cls}_q3"
                median_val = row.get(median_key)
                q1_val = row.get(q1_key)
                q3_val = row.get(q3_key)
                if median_val is None or q1_val is None or q3_val is None:
                    continue
                by_method[method][cls].append(
                    {
                        "k": k,
                        "median": float(median_val) if median_val else float("nan"),
                        "q1": float(q1_val) if q1_val else float("nan"),
                        "q3": float(q3_val) if q3_val else float("nan"),
                    }
                )

    if not by_method:
        raise ValueError("Summary CSV contained no rows to plot.")

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    x_fmt = FuncFormatter(lambda v, p: f"${int(v)}$" if float(v).is_integer() else f"${v:.2f}$")
    y_fmt = FuncFormatter(lambda v, p: f"${v:.2f}$")
    for ax in axes:
        ax.xaxis.set_major_formatter(x_fmt)
        ax.yaxis.set_major_formatter(y_fmt)

    panels = [
        ("class1", ""),
        ("class0", ""),
        ("diff", ""),
    ]

    color_map = {"bregman": "blue", "spectral": "orange"}

    for ax, (cls, title) in zip(axes, panels):
        for method, entry_map in sorted(by_method.items()):
            entries = sorted(entry_map[cls], key=lambda item: item["k"])
            if not entries:
                continue
            ks = np.array([item["k"] for item in entries], dtype=np.int32)
            median = np.array([item["median"] for item in entries], dtype=np.float64)
            q1 = np.array([item["q1"] for item in entries], dtype=np.float64)
            q3 = np.array([item["q3"] for item in entries], dtype=np.float64)
            color = color_map.get(method, "gray")
            label = "Bregman" if method == "bregman" else ("Spectral" if method == "spectral" else method)
            ax.plot(ks, median, marker="o", label=label, color=color)
            if np.all(np.isfinite(q1)) and np.all(np.isfinite(q3)):
                ax.fill_between(ks, q1, q3, alpha=0.15, color=color)
        if cls == "diff":
            ax.set_ylabel(r"Separability", fontsize=16)
        else:
            ax.set_ylabel(r"Stable concentr." if cls == "class1" else r"Unstable concentr.", fontsize=16)
        ax.set_title(title, fontsize=20, loc="left")
        ax.grid(True, linestyle="--", alpha=0.3)
        ydata_lists = [line.get_ydata() for line in ax.get_lines()]
        if ydata_lists:
            y_min = float(np.nanmin([np.nanmin(yd) for yd in ydata_lists]))
            y_max = float(np.nanmax([np.nanmax(yd) for yd in ydata_lists]))
        else:
            y_min, y_max = 0.0, 1.0
        if y_min < 0.0:
            ax.set_ylim(y_min - 0.1, 1.0)
        else:
            ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.tick_params(axis="both", which="minor", labelsize=18)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    axes[-1].set_xlabel("Number of clusters ($k$)", fontsize=16)
    axes[0].legend(loc="best", fontsize=18)

    method_k_stars = {}
    for method, entry_map in by_method.items():
        method_diffs = {}
        for item in entry_map.get("diff", []):
            med = item.get("median")
            if med is not None and np.isfinite(med):
                method_diffs[item["k"]] = float(med)
        if method_diffs:
            method_k_stars[method] = max(method_diffs, key=method_diffs.get)

    for method, k_star in method_k_stars.items():
        color = color_map.get(method, "gray")
        axes[-1].axvspan(k_star - 0.5, k_star + 0.5, color=color, alpha=0.15, label=f"{method} max")
        axes[0].axvspan(k_star - 0.5, k_star + 0.5, color=color, alpha=0.15)
        axes[1].axvspan(k_star - 0.5, k_star + 0.5, color=color, alpha=0.15)

    fig.tight_layout()

    if output_pdf is None:
        stem = summary_csv.stem
        if stem.startswith("cohesion_scan_summary_"):
            suffix = stem.replace("cohesion_scan_summary_", "")
            output_pdf = summary_csv.with_name(f"cohesion_vs_k_{suffix}.pdf")
        else:
            output_pdf = summary_csv.with_name("cohesion_vs_k.pdf")
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote cohesion plot to {output_pdf}")


def plot_cohesion_overlay(
    summary_by_part: Mapping[str, Path],
    output_pdf: Path,
    methods_subset: Sequence[str] | None = None,
) -> None:
    """Overlay cohesion curves (s1 - s0) across molecular parts on one figure."""
    plt.figure(figsize=(8, 6))
    x_fmt = FuncFormatter(lambda v, p: f"${int(v)}$" if float(v).is_integer() else f"${v:.2f}$")
    y_fmt = FuncFormatter(lambda v, p: f"${v:.2f}$")

    for mol_part, summary_csv in summary_by_part.items():
        if not summary_csv.is_file():
            continue
        by_method: Dict[str, List[dict[str, float]]] = defaultdict(list)
        raw_groups: Dict[str, Dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        has_summary_rows = False
        with summary_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method = row.get("method", "")
                if methods_subset is not None and method not in methods_subset:
                    continue
                try:
                    k = int(row.get("k", "0"))
                except (TypeError, ValueError):
                    continue

                if "diff_median" in row:
                    has_summary_rows = True
                    median = float(row.get("diff_median", "nan"))
                    q1 = float(row.get("diff_q1", "nan"))
                    q3 = float(row.get("diff_q3", "nan"))
                    by_method[method].append({"k": k, "median": median, "q1": q1, "q3": q3})
                elif "class1_share" in row and "class0_share" in row:
                    try:
                        c1 = float(row["class1_share"])
                        c0 = float(row["class0_share"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if np.isfinite(c1) and np.isfinite(c0):
                        raw_groups[method][k].append(c1 - c0)

        if not has_summary_rows and raw_groups:
            for method, k_map in raw_groups.items():
                for k, vals in k_map.items():
                    vals_arr = np.asarray(vals, dtype=float)
                    if vals_arr.size == 0:
                        continue
                    median = float(np.nanmedian(vals_arr))
                    q1 = float(np.nanpercentile(vals_arr, 25))
                    q3 = float(np.nanpercentile(vals_arr, 75))
                    by_method[method].append({"k": k, "median": median, "q1": q1, "q3": q3})

        for method, entries in sorted(by_method.items()):
            entries = sorted(entries, key=lambda item: item["k"])
            ks = np.array([item["k"] for item in entries], dtype=np.int32)
            median = np.array([item["median"] for item in entries], dtype=np.float64)
            q1 = np.array([item["q1"] for item in entries], dtype=np.float64)
            q3 = np.array([item["q3"] for item in entries], dtype=np.float64)
            label = f"{method} ({mol_part})"
            plt.plot(ks, median, marker="o", label=label)
            if np.all(np.isfinite(q1)) and np.all(np.isfinite(q3)):
                plt.fill_between(ks, q1, q3, alpha=0.12)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(x_fmt)
    ax.yaxis.set_major_formatter(y_fmt)
    plt.xlabel("Number of clusters ($k$)", fontsize=16)
    plt.ylabel(r"Cohesion score ($s_1 - s_0$)")
    plt.title(r"Cohesion score ($s_1 - s_0$) vs k by molecular part")

    y_lower_candidates = []
    y_upper_candidates = []
    for line in ax.get_lines():
        ydata = line.get_ydata()
        y_lower_candidates.append(np.nanmin(ydata))
        y_upper_candidates.append(np.nanmax(ydata))
    if y_lower_candidates:
        y_min = float(np.nanmin(y_lower_candidates))
        y_max = float(np.nanmax(y_upper_candidates))
    else:
        y_min, y_max = 0.0, 1.0
    if y_min < 0.0:
        plt.ylim(y_min - 0.1, 1.0)
    else:
        plt.ylim(0.0, 1.0)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf")
    plt.close()
    print(f"Wrote cohesion overlay to {output_pdf}")


def plot_rank_window_raw_diff(
    raw_by_window: Mapping[str, Path],
    output_pdf: Path,
    methods_subset: Sequence[str] | None = None,
    k_filter: int | None = None,
    title: str = "Class-1 separability (raw seeds) across rank windows",
) -> None:
    """Compare class-1 separability (s1 - s0) across rank windows using raw data."""
    data: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for window_label, csv_path in raw_by_window.items():
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method = row.get("method", "")
                if methods_subset and method not in methods_subset:
                    continue
                if k_filter is not None:
                    try:
                        k_val = int(row.get("k", "0"))
                    except (TypeError, ValueError):
                        continue
                    if k_val != k_filter:
                        continue
                try:
                    c1 = float(row.get("class1_share", "nan"))
                    c0 = float(row.get("class0_share", "nan"))
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(c1) and np.isfinite(c0)):
                    continue
                data[method][window_label].append(c1 - c0)

    if not data:
        raise ValueError("No data loaded for raw rank-window comparison; check raw CSV paths/methods/k_filter.")

    methods = sorted(data.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, method in zip(axes, methods, strict=True):
        windows = sorted(data[method].keys())
        values = [data[method][w] for w in windows]
        ax.boxplot(
            values,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor="#c6dbef", edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            showfliers=True,
        )
        ax.set_xticks(range(1, len(windows) + 1))
        ax.set_xticklabels(windows, rotation=90, ha="right")
        ax.set_title(f"{method}", fontsize=16)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        flat_vals = [v for sub in values for v in sub]
        if flat_vals:
            vmin = min(flat_vals)
            ymin = vmin - 0.1 if vmin < 0.0 else 0.0
        else:
            ymin = 0.0
        ax.set_ylim(ymin, 1.0)
        ax.set_ylabel(r"$s_1 - s_0$", fontsize=18)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf")
    plt.close()


def plot_molpart_box_from_raw(
    raw_root: Path,
    metric: str,
    parts: Sequence[str],
    windows: Sequence[str],
    ks: Sequence[int] | None = None,
    combined_only: bool = False,
    per_method: bool = False,
) -> None:
    """Boxplots of class-1 separability (s1-s0) per mol-part from raw cohesion scans."""
    label_map = {
        "whole": " ",
        "CT_off": " ",
        "CT_TMD_off": " ",
        "TMD_only": " ",
    }
    display_parts = [label_map.get(p, p) for p in parts]

    def _hist_edges(all_vals: list[float]) -> np.ndarray:
        if not all_vals:
            return np.array([0.0, 1.0], dtype=float)
        if len(all_vals) == 1:
            v = float(all_vals[0])
            return np.array([v - 0.5, v + 0.5], dtype=float)
        vmin = min(all_vals)
        vmax = max(all_vals)
        if vmin == vmax:
            return np.array([vmin - 0.5, vmin + 0.5], dtype=float)
        return np.histogram_bin_edges(all_vals, bins="auto")

    def _mode_from_hist(values: np.ndarray, edges: np.ndarray) -> float:
        if values.size == 0 or edges.size < 2:
            return float("nan")
        hist, _ = np.histogram(values, bins=edges)
        if hist.size == 0:
            return float("nan")
        idx = int(np.argmax(hist))
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    def _score_parts(values: Sequence[Sequence[float]]) -> dict[str, float]:
        all_vals = [v for sub in values for v in sub]
        if not all_vals:
            return {}
        edges = _hist_edges(all_vals)
        stats: dict[str, float] = {}
        for idx, vals in enumerate(values):
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            label = display_parts[idx]
            med = float(np.nanmedian(arr))
            mean = float(np.nanmean(arr))
            mode = _mode_from_hist(arr, edges)
            if not (np.isfinite(med) and np.isfinite(mean) and np.isfinite(mode)):
                continue
            score = float(np.mean([med, mean, mode]))
            stats[label] = float(score)
        return stats

    def _select_best_part(values: Sequence[Sequence[float]]) -> tuple[str, float] | None:
        stats = _score_parts(values)
        if not stats:
            return None
        best_label = max(stats, key=stats.get)
        return best_label, stats[best_label]

    def _report_zone_best(values_by_method: dict[str, list[list[float]]], ks_label: str) -> None:
        if not combined_only:
            return
        method_scores: dict[str, dict[str, float]] = {}
        for method, vals in sorted(values_by_method.items()):
            stats = _score_parts(vals)
            if not stats:
                continue
            method_scores[method] = stats
            label, score = _select_best_part(vals)
            print(f"Zone {ks_label} | {method}: {label} (avg(median,mean,mode)={score:.4f})")

        merge_methods = [m for m in ("bregman", "spectral") if m in method_scores]
        if len(merge_methods) == 2:
            common_labels = set(method_scores[merge_methods[0]].keys())
            common_labels &= set(method_scores[merge_methods[1]].keys())
            if common_labels:
                merged_scores: list[tuple[float, str]] = []
                for label in common_labels:
                    avg_vals = [method_scores[m][label] for m in merge_methods]
                    merged_score = float(np.mean(avg_vals))
                    merged_scores.append((merged_score, label))
                merged_scores.sort(reverse=True)
                merged_score, label = merged_scores[0]
                print(f"Zone {ks_label} | merged: {label} (avg(median,mean,mode)={merged_score:.4f})")

    def load_data() -> dict[str, dict[int, list[list[float]]]]:
        out: dict[str, dict[int, list[list[float]]]] = {}
        for part in parts:
            for w in windows:
                f = raw_root / f"cohesion_scan_raw_{metric}_{part}_{w}.csv"
                if not f.is_file():
                    continue
                with f.open() as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        try:
                            k_val = int(row.get("k", "0"))
                        except ValueError:
                            continue
                        if ks is not None and k_val not in ks:
                            continue
                        try:
                            c1 = float(row["class1_share"])
                            c0 = float(row["class0_share"])
                        except (KeyError, ValueError):
                            continue
                        if not (np.isfinite(c1) and np.isfinite(c0)):
                            continue
                        method = row.get("method", "").strip() or "unknown"
                        out.setdefault(method, {})
                        out[method].setdefault(k_val, [[] for _ in parts])
                        out[method][k_val][parts.index(part)].append(c1 - c0)
        return out

    all_data = load_data()
    if not all_data:
        raise SystemExit("No data loaded; check raw paths/filenames or k filter.")

    def plot_for_method(method: str, data_by_k: dict[int, list[list[float]]]) -> None:
        if not data_by_k:
            return
        method_label = "methods_merged" if method == "all_methods" else method
        ks_to_plot = sorted(data_by_k.keys()) if ks is None else sorted(set(ks))

        if not combined_only:
            for k_val in ks_to_plot:
                values = data_by_k.get(k_val, [])
                if not any(values):
                    continue
                plt.figure(figsize=(8, 6))
                plt.boxplot(
                    values,
                    labels=display_parts,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(facecolor="#c6dbef", edgecolor="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    showfliers=True,
                )
                ax = plt.gca()
                ax.tick_params(axis="both", labelsize=17)
                ax.set_xticklabels(display_parts, fontsize=17)
                flat_vals = [v for sub in values for v in sub]
                ymin = min(flat_vals) - 0.1 if flat_vals and min(flat_vals) < 0.0 else 0.0
                plt.ylim(ymin, 1.0)
                plt.ylabel(r"Co-clustering tendency of physiologically stable heterodimers ($s_1 - s_0$)", fontsize=17)
                plt.title(
                    f"{metric} ({method_label}) class-1 separability by mol-part (k={k_val})\nwindows: {', '.join(windows)}",
                    fontsize=14,
                )
                plt.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                out = raw_root / f"molpart_box_{metric}_{method_label}_k{k_val}_subwindows.pdf"
                plt.savefig(out, format="pdf")
                plt.close()
                print(f"Wrote {out}")

        combined_values: list[list[float]] = [[] for _ in parts]
        source = {k: v for k, v in data_by_k.items() if k in ks_to_plot}
        for vals_per_part in source.values():
            for idx, vals in enumerate(vals_per_part):
                combined_values[idx].extend(vals)
        if any(combined_values):
            plt.figure(figsize=(8, 6))
            plt.boxplot(
                combined_values,
                labels=display_parts,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor="#c6dbef", edgecolor="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                showfliers=True,
            )
            ax = plt.gca()
            ax.tick_params(axis="both", labelsize=15)
            ax.set_xticklabels(display_parts, fontsize=15)
            flat_vals = [v for sub in combined_values for v in sub]
            ymin = min(flat_vals) - 0.1 if flat_vals and min(flat_vals) < 0.0 else 0.0
            plt.ylim(ymin, 1.0)
            plt.ylabel(r"Separability of physiologically stable heterodimers ($s_1 - s_0$)", fontsize=16)
            plt.title(
                f"{metric} ({method_label}) class-1 separability by mol-part (k={','.join(map(str, ks_to_plot))})\nwindows: {', '.join(windows)}",
                fontsize=14,
            )
            plt.grid(True, axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            k_tag = "k" + "".join(map(str, ks_to_plot))
            out = raw_root / f"molpart_box_{metric}_{method_label}_{k_tag}_subwindows.pdf"
            plt.savefig(out, format="pdf")
            plt.close()
            print(f"Wrote {out}")

    if per_method:
        for method, data_by_k in all_data.items():
            plot_for_method(method, data_by_k)
    else:
        merged: dict[int, list[list[float]]] = {}
        for data_by_k in all_data.values():
            for k_val, vals in data_by_k.items():
                merged.setdefault(k_val, [[] for _ in parts])
                for idx, subvals in enumerate(vals):
                    merged[k_val][idx].extend(subvals)
        plot_for_method("all_methods", merged)

    def plot_combined_methods_overlay() -> None:
        methods_present = sorted(all_data.keys())
        if len(methods_present) < 2:
            return
        ks_union = sorted({k for data_by_k in all_data.values() for k in data_by_k.keys()})
        ks_to_plot = ks_union if ks is None else sorted(set(ks))
        if not ks_to_plot:
            return
        values_by_method: dict[str, list[list[float]]] = {}
        for method, data_by_k in all_data.items():
            combined: list[list[float]] = [[] for _ in parts]
            for k_val in ks_to_plot:
                for idx, subvals in enumerate(data_by_k.get(k_val, [[] for _ in parts])):
                    combined[idx].extend(subvals)
            values_by_method[method] = combined
        if not values_by_method:
            return

        plt.figure(figsize=(9, 6))
        base_pos = np.arange(1, len(parts) + 1)
        offset = 0.15
        colors = {"bregman": "blue", "spectral": "orange"}
        alpha = 0.6
        for i, method in enumerate(methods_present[:2]):
            vals = values_by_method.get(method, [[] for _ in parts])
            pos = base_pos - offset if i == 0 else base_pos + offset
            plt.boxplot(
                vals,
                positions=pos,
                widths=0.25,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.3),
                boxprops=dict(facecolor=colors.get(method, "gray"), edgecolor="black", alpha=alpha),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                showfliers=True,
            )

        plt.xticks(base_pos, display_parts, fontsize=17)
        plt.tick_params(axis="y", labelsize=17)
        plt.ylabel(r"Separability ($k=5,6,7,8$)", fontsize=20)
        legend_handles = [
            Line2D(
                [],
                [],
                color=colors.get("bregman", "tab:blue"),
                linewidth=8,
                alpha=alpha,
                label="Bregman",
            ),
            Line2D(
                [],
                [],
                color=colors.get("spectral", "orange"),
                linewidth=8,
                alpha=alpha,
                label="Spectral",
            ),
        ]
        plt.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.35,
            edgecolor="black",
            fontsize=17,
        )
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        k_tag = "k" + "".join(map(str, ks_to_plot))
        out = raw_root / f"molpart_box_{metric}_{k_tag}_subwindows.pdf"
        plt.savefig(out, format="pdf")
        plt.close()
        print(f"Wrote {out}")
        ks_label = "k=" + ",".join(map(str, ks_to_plot))
        _report_zone_best(values_by_method, ks_label)

    plot_combined_methods_overlay()


__all__ = [
    "plot_cohesion_summary",
    "plot_cohesion_overlay",
    "plot_rank_window_raw_diff",
    "plot_molpart_box_from_raw",
]
