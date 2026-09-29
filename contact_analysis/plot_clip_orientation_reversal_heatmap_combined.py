#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from .clip_groove_fluctuation_direction import DEFAULT_TSV, OUT_DIR
    from .plot_clip_orientation_reversal_heatmap import (
        allele_sort_key,
        format_allele,
        load_summary,
        summary_to_matrix,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import DEFAULT_TSV, OUT_DIR
    from plot_clip_orientation_reversal_heatmap import (
        allele_sort_key,
        format_allele,
        load_summary,
        summary_to_matrix,
    )


DEFAULT_OUT_PDF = OUT_DIR / "clip_core_orientation_reversal_probability_combined_heatmap.pdf"

FIG_W = 11.0
FIG_H = 6.2
AXIS_FS = 22
TICK_FS = 20
TITLE_FS = 24
CBAR_LABEL_FS = 24
CBAR_SIZE = "4.3%"
CBAR_PAD = 0.25
SAVE_PAD_INCHES = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-rank and high-rank CLIP orientation-reversal heatmaps side by side."
    )
    parser.add_argument("--in-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3"])
    parser.add_argument("--q-threshold", type=float, default=0.0)
    parser.add_argument("--cmap", default="Oranges")
    return parser.parse_args()


def matrix_for_window(args: argparse.Namespace, rank_start: int, rank_end: int):
    window_args = SimpleNamespace(
        in_tsv=args.in_tsv,
        rank_start=rank_start,
        rank_end=rank_end,
        exclude_alpha=args.exclude_alpha,
        q_threshold=args.q_threshold,
    )
    summary = load_summary(window_args)
    rows, cols, mat = summary_to_matrix(summary)
    return rows, cols, mat, summary


def draw_heatmap(ax, mat: np.ndarray, rows: list[str], cols: list[str], *, cmap):
    image = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([format_allele(x) for x in cols], rotation=90, fontsize=TICK_FS)
    ax.set_yticklabels([format_allele(y) for y in rows], fontsize=TICK_FS)
    ax.set_xlabel(r"HLA-DQA1$^*$", fontsize=AXIS_FS)
    ax.tick_params(axis="both", length=0)
    return image


def main() -> int:
    args = parse_args()
    rows_all, cols_all, mat_all, summary_all = matrix_for_window(args, 1, 1000)
    rows_hi, cols_hi, mat_hi, summary_hi = matrix_for_window(args, 901, 1000)

    if rows_all != rows_hi or cols_all != cols_hi:
        raise SystemExit("All-rank and high-rank matrices do not have matching row/column labels.")

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": TICK_FS,
            "axes.labelsize": AXIS_FS,
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cmap = mpl.colormaps[args.cmap].with_extremes(bad="lightgray")
    fig, (ax_all, ax_hi) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)

    image = draw_heatmap(ax_all, mat_all, rows_all, cols_all, cmap=cmap)
    draw_heatmap(ax_hi, mat_hi, rows_hi, cols_hi, cmap=cmap)
    ax_all.set_title(r"$r=1$", fontsize=TITLE_FS)
    ax_hi.set_title(r"$r>900$", fontsize=TITLE_FS)
    ax_all.set_ylabel(r"HLA-DQB1$^*$", fontsize=AXIS_FS)
    ax_hi.tick_params(axis="y", left=False, labelleft=False)

    divider = make_axes_locatable(ax_hi)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$p_{\mathrm{rev}}(i)$", fontsize=CBAR_LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.outline.set_linewidth(0.8)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.90, wspace=0.28)
    fig.savefig(args.out_pdf, format="pdf", bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)

    print(f"[write] {args.out_pdf}")
    print(
        "[summary all ranks] "
        f"cells={np.isfinite(mat_all).sum()} "
        f"median_p={np.nanmedian(mat_all):.3f} "
        f"max_p={np.nanmax(mat_all):.3f}"
    )
    print(
        "[summary ranks 901-1000] "
        f"cells={np.isfinite(mat_hi).sum()} "
        f"median_p={np.nanmedian(mat_hi):.3f} "
        f"max_p={np.nanmax(mat_hi):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
