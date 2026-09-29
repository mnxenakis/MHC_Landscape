#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .clip_contact_register_maps import (
        DEFAULT_OUT_TSV,
        DEFAULT_PARTIAL_DIR,
        OUT_DIR,
        parse_rank_zones,
        ordered_zone_labels,
        pivot_matrix,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_contact_register_maps import (
        DEFAULT_OUT_TSV,
        DEFAULT_PARTIAL_DIR,
        OUT_DIR,
        parse_rank_zones,
        ordered_zone_labels,
        pivot_matrix,
    )


DEFAULT_GEOMETRY_TSV = OUT_DIR / "clip_groove_fluctuation_direction.tsv"
DEFAULT_OUT_PDF_REV = OUT_DIR / "clip_core_contact_register_maps_reversed_only.pdf"
DEFAULT_OUT_PDF_NONREV = OUT_DIR / "clip_core_contact_register_maps_canonical_orientation.pdf"
DEFAULT_REVERSED_ONLY_TSV = OUT_DIR / "clip_core_contact_register_highrank_reversed_only.tsv"
DEFAULT_REVERSED_ONLY_COMPARISON_ZONE = "ranks 901--1000 reversed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot simplified CLIP contact-register heatmaps for two heterodimer cohorts: "
            "reverse-bound and canonical-orientation. The reverse-bound PDF uses only "
            "high-rank CLIP-reversed structures, while the canonical-orientation PDF uses "
            "the cached canonical-orientation cohort aggregate."
        )
    )
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--geometry-tsv", type=Path, default=DEFAULT_GEOMETRY_TSV)
    parser.add_argument("--rank-zones", default="1-100,101-200,201-300,301-400,401-500,501-600,601-700,701-800,801-900,901-1000")
    parser.add_argument("--reference-zone", default="rank 1")
    parser.add_argument("--comparison-zone", default="ranks 901--1000")
    parser.add_argument("--orientation-threshold", type=float, default=0.0)
    parser.add_argument("--reversion-rank-start", type=int, default=2)
    parser.add_argument("--reversion-rank-end", type=int, default=1000)
    parser.add_argument("--out-pdf-rev", type=Path, default=DEFAULT_OUT_PDF_REV)
    parser.add_argument("--out-pdf-nonrev", type=Path, default=DEFAULT_OUT_PDF_NONREV)
    parser.add_argument("--reversed-only-tsv", type=Path, default=DEFAULT_REVERSED_ONLY_TSV)
    return parser.parse_args()


def load_reversion_cohorts(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    if not args.geometry_tsv.exists():
        raise SystemExit(f"Missing geometry TSV: {args.geometry_tsv}")
    df = pd.read_csv(args.geometry_tsv, sep="\t", usecols=["heterodimer", "rank", "clip_core_orientation_dot"])
    df["heterodimer"] = df["heterodimer"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["clip_core_orientation_dot"] = pd.to_numeric(df["clip_core_orientation_dot"], errors="coerce")
    df = df.dropna(subset=["rank", "clip_core_orientation_dot"]).copy()
    df = df[(df["rank"] >= args.reversion_rank_start) & (df["rank"] <= args.reversion_rank_end)].copy()
    if df.empty:
        raise SystemExit("No orientation rows remain after applying the requested rank window.")

    grouped = (
        df.groupby("heterodimer", as_index=False)
        .agg(reversion_possible=("clip_core_orientation_dot", lambda x: bool((np.asarray(x, dtype=float) < args.orientation_threshold).any())))
    )
    rev = set(grouped.loc[grouped["reversion_possible"], "heterodimer"].astype(str))
    nonrev = set(grouped.loc[~grouped["reversion_possible"], "heterodimer"].astype(str))
    return rev, nonrev


def aggregate_partial_subset(partial_dir: Path, heterodimers: set[str], zone_labels: list[str]) -> pd.DataFrame:
    if not partial_dir.exists():
        raise SystemExit(f"Missing partial-dir: {partial_dir}")
    if not heterodimers:
        raise SystemExit("No heterodimers were assigned to this cohort.")
    paths = [partial_dir / f"{heterodimer}.tsv" for heterodimer in sorted(heterodimers)]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise SystemExit(f"No partial TSVs found for the selected cohort in {partial_dir}")
    parts = [pd.read_csv(path, sep="\t") for path in existing]
    df = pd.concat(parts, ignore_index=True)
    group_cols = [
        "rank_zone",
        "chain_label",
        "msa_resnum",
        "axis_label",
        "sort_value",
        "clip_position",
        "clip_residue_label",
        "cutoff_angstrom",
        "contact_shell_threshold",
    ]
    agg = (
        df.groupby(group_cols, as_index=False, observed=False)
        .agg(
            n_observations=("n_observations", "sum"),
            n_contacts=("n_contacts", "sum"),
            sum_min_distance_angstrom=("sum_min_distance_angstrom", "sum"),
        )
    )
    agg["contact_probability"] = agg["n_contacts"] / agg["n_observations"]
    agg["mean_min_distance_angstrom"] = agg["sum_min_distance_angstrom"] / agg["n_observations"]
    agg["rank_zone"] = pd.Categorical(agg["rank_zone"], categories=zone_labels, ordered=True)
    agg = agg.sort_values(["rank_zone", "chain_label", "sort_value", "clip_position"]).reset_index(drop=True)
    agg["rank_zone"] = agg["rank_zone"].astype(str)
    return agg


def plot_simple_register_maps(df: pd.DataFrame, out_pdf: Path, reference_zone: str, comparison_zone: str, cohort_title: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 24,
            "ytick.labelsize": 22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 13.0), constrained_layout=True)
    chain_labels = [("alpha", "α"), ("beta", "β")]
    titles = [r"$r=1$", r"$r>900$"]
    prob_images = []

    for row_idx, (chain_label, chain_symbol) in enumerate(chain_labels):
        matrices = [
            pivot_matrix(df, reference_zone, chain_label, "contact_probability")[0],
            pivot_matrix(df, comparison_zone, chain_label, "contact_probability")[0],
        ]
        rows = list(matrices[0].index)
        cols = list(matrices[0].columns)
        for col_idx, matrix in enumerate(matrices):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="Oranges")
            prob_images.append(im)
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=40)
            if col_idx == 0:
                ax.set_ylabel(f"{chain_symbol}", fontsize=40)
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels(rows)
            else:
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels([])
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels(cols, rotation=90)
            if row_idx == 1:
                ax.set_xlabel("CLIP-core residue", fontsize=40)
            ax.tick_params(axis="x", length=0, labelsize=32)
            ax.tick_params(axis="y", length=0, labelsize=24)
            ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
            ax.grid(which="minor", color="k", linewidth=0.45, alpha=0.35)

    
    if prob_images:
        cbar = fig.colorbar(
            prob_images[0],
            ax=axes.ravel().tolist(),
            shrink=0.52,
            pad=0.01,
            fraction=0.03,
            aspect=24,
        )
        cbar.set_label(r"$p_{\mathrm{contact}}$", fontsize=40)
        cbar.ax.tick_params(labelsize=29)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def load_reversed_only_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing reversed-only TSV: {path}. "
            "Run plot_clip_contact_register_maps_reversed_only.py or its sbatch wrapper first."
        )
    return pd.read_csv(path, sep="\t")


def main() -> int:
    args = parse_args()
    zones = parse_rank_zones(args.rank_zones)
    zone_labels = ordered_zone_labels(zones)
    rev, nonrev = load_reversion_cohorts(args)

    rev_df = load_reversed_only_table(args.reversed_only_tsv)
    nonrev_df = aggregate_partial_subset(args.partial_dir, nonrev, zone_labels)

    plot_simple_register_maps(
        rev_df,
        args.out_pdf_rev,
        args.reference_zone,
        DEFAULT_REVERSED_ONLY_COMPARISON_ZONE,
        "CLIP reversed structures only",
    )
    print(f"[write] {args.out_pdf_rev}")
    plot_simple_register_maps(
        nonrev_df,
        args.out_pdf_nonrev,
        args.reference_zone,
        args.comparison_zone,
        "CLIP reversion not observed",
    )
    print(f"[write] {args.out_pdf_nonrev}")
    print(f"[summary] reversion_possible_heterodimers={len(rev)}")
    print(f"[summary] canonical_orientation_heterodimers={len(nonrev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
