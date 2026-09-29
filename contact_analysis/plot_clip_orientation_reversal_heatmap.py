#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from .clip_groove_fluctuation_direction import DEFAULT_TSV, OUT_DIR
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import DEFAULT_TSV, OUT_DIR


DEFAULT_OUT_PDF = OUT_DIR / "clip_core_orientation_reversal_probability_heatmap.pdf"
DEFAULT_SUMMARY_TSV = OUT_DIR / "clip_core_orientation_reversal_probability.tsv"
DEFAULT_MATRIX_TSV = OUT_DIR / "clip_core_orientation_reversal_probability_heatmap.tsv"


FIG_W = 10.5
FIG_H = 12.0
AXIS_FS = 35
TICK_FS = 30
TITLE_FS = 31
CBAR_LABEL_FS = 30


def canonical_allele(text: str) -> str:
    text = str(text).strip()
    for sep in (".", ":"):
        if sep in text:
            a, b = text.split(sep, 1)
            return f"{int(a)}.{int(b)}"
    return text


def canonical_pair(text: str) -> str:
    alpha, beta = str(text).strip().split("_", 1)
    return f"{canonical_allele(alpha)}_{canonical_allele(beta)}"


def allele_sort_key(text: str) -> tuple[int, int]:
    canon = canonical_allele(text)
    if "." not in canon:
        return int(canon), 0
    a, b = canon.split(".", 1)
    return int(a), int(b)


def format_allele(text: str) -> str:
    a, b = canonical_allele(text).split(".", 1)
    return f"{int(a):02d}:{int(b):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a HLA-DQ heterodimer heatmap of CLIP-core orientation-reversal "
            "probability across a high-rank AF2 window."
        )
    )
    parser.add_argument("--in-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--matrix-tsv", type=Path, default=DEFAULT_MATRIX_TSV)
    parser.add_argument("--rank-start", type=int, default=901)
    parser.add_argument("--rank-end", type=int, default=1000)
    parser.add_argument(
        "--exclude-alpha",
        nargs="*",
        default=["1.5", "3.3"],
        help="DQA1 columns to exclude from the heatmap, using 1.5 or 01:05 notation.",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=0.0,
        help="Orientation reversal threshold. Default: q_orient < 0.",
    )
    parser.add_argument("--cmap", default="magma")
    return parser.parse_args()


def load_summary(args: argparse.Namespace) -> pd.DataFrame:
    if not args.in_tsv.exists():
        raise SystemExit(f"Missing input TSV: {args.in_tsv}")
    cols = ["heterodimer", "rank", "clip_core_orientation_dot"]
    df = pd.read_csv(args.in_tsv, sep="\t", usecols=cols)
    df["heterodimer"] = df["heterodimer"].map(canonical_pair)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["clip_core_orientation_dot"] = pd.to_numeric(df["clip_core_orientation_dot"], errors="coerce")
    df = df.dropna(subset=["rank", "clip_core_orientation_dot"])
    df = df[(df["rank"] >= args.rank_start) & (df["rank"] <= args.rank_end)].copy()
    if df.empty:
        raise SystemExit(f"No rows found for ranks {args.rank_start}--{args.rank_end}.")

    grouped = (
        df.groupby("heterodimer", as_index=False)
        .agg(
            n_ranks=("clip_core_orientation_dot", "size"),
            n_reversed=("clip_core_orientation_dot", lambda x: int((x < args.q_threshold).sum())),
            median_q_orient=("clip_core_orientation_dot", "median"),
            q25_q_orient=("clip_core_orientation_dot", lambda x: x.quantile(0.25)),
            q75_q_orient=("clip_core_orientation_dot", lambda x: x.quantile(0.75)),
            min_q_orient=("clip_core_orientation_dot", "min"),
        )
        .sort_values("heterodimer")
    )
    grouped["p_orientation_reversal"] = grouped["n_reversed"] / grouped["n_ranks"]
    grouped["alpha"] = grouped["heterodimer"].str.split("_").str[0]
    grouped["beta"] = grouped["heterodimer"].str.split("_").str[1]
    excluded_alpha = {canonical_allele(value) for value in (args.exclude_alpha or [])}
    if excluded_alpha:
        grouped = grouped[~grouped["alpha"].isin(excluded_alpha)].copy()
    return grouped[
        [
            "heterodimer",
            "alpha",
            "beta",
            "n_ranks",
            "n_reversed",
            "p_orientation_reversal",
            "median_q_orient",
            "q25_q_orient",
            "q75_q_orient",
            "min_q_orient",
        ]
    ]


def summary_to_matrix(summary: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    rows = sorted(summary["beta"].unique(), key=allele_sort_key)
    cols = sorted(summary["alpha"].unique(), key=allele_sort_key)
    row_idx = {label: i for i, label in enumerate(rows)}
    col_idx = {label: j for j, label in enumerate(cols)}
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for row in summary.itertuples(index=False):
        mat[row_idx[row.beta], col_idx[row.alpha]] = float(row.p_orientation_reversal)
    return rows, cols, mat


def write_matrix(path: Path, rows: list[str], cols: list[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("DQB1/DQA1\t" + "\t".join(cols) + "\n")
        for label, values in zip(rows, mat):
            formatted = ["nan" if not np.isfinite(v) else f"{v:.6g}" for v in values]
            handle.write(label + "\t" + "\t".join(formatted) + "\n")


def plot_heatmap(
    out_pdf: Path,
    rows: list[str],
    cols: list[str],
    mat: np.ndarray,
    *,
    cmap: str,
    rank_start: int,
    rank_end: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 24,
            "axes.labelsize": AXIS_FS,
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=mpl.colormaps[cmap].with_extremes(bad="lightgray"),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([format_allele(x) for x in cols], rotation=90)
    ax.set_yticklabels([format_allele(y) for y in rows])
    ax.set_xlabel(r"DQA1*")
    ax.set_ylabel(r"DQB1*")
    ax.tick_params(axis="both", length=0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.3%", pad=0.25)
    cbar = plt.colorbar(image, cax=cax)
    cbar.set_label(r"$P(q_{\mathrm{orient}}<0)$", fontsize=CBAR_LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.outline.set_linewidth(0.8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = load_summary(args)
    rows, cols, mat = summary_to_matrix(summary)

    args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_tsv, sep="\t", index=False)
    write_matrix(args.matrix_tsv, rows, cols, mat)
    plot_heatmap(
        args.out_pdf,
        rows,
        cols,
        mat,
        cmap=args.cmap,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
    )

    print(f"[write] {args.summary_tsv}")
    print(f"[write] {args.matrix_tsv}")
    print(f"[write] {args.out_pdf}")
    print(
        "[summary] "
        f"cells={np.isfinite(mat).sum()} "
        f"median_p={np.nanmedian(mat):.3f} "
        f"max_p={np.nanmax(mat):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
