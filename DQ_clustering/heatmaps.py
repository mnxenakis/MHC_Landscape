"""General publication heatmaps for per-label scalar summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .mpl_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib.pyplot as plt
import numpy as np

from .plot_common import (
    HEATMAP_FIG_H_SCALE,
    HEATMAP_FIG_W_SCALE,
    HEATMAP_MIN_H,
    HEATMAP_MIN_W,
    _add_highlight,
    _build_axes_from_labels,
    _format_allele_label,
    _get_cmap_copy,
    _write_heatmap_tsv,
)


def plot_value_heatmap(
    values: Mapping[str, float],
    output_pdf: Path,
    *,
    parents: Sequence[str] | None = None,
    subs: Sequence[str] | None = None,
    title: str = "Value heatmap",
    cbar_label: str = "Value",
    cmap: str = "viridis",
    highlight_labels: Sequence[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    export_tsv: bool = False,
    tsv_path: Path | None = None,
) -> None:
    """Plot a single heatmap for per-label scalar values."""
    if parents is None or subs is None:
        parents, subs, parent_index, sub_index, formatted_parents, formatted_subs = _build_axes_from_labels(
            values.keys()
        )
    else:
        parent_index = {parent: idx for idx, parent in enumerate(parents)}
        sub_index = {sub: idx for idx, sub in enumerate(subs)}
        formatted_parents = [_format_allele_label(lbl) for lbl in parents]
        formatted_subs = [_format_allele_label(lbl) for lbl in subs]

    mat = np.full((len(subs), len(parents)), np.nan, dtype=np.float64)
    highlights: set[tuple[int, int]] = set()
    highlight_set = set(highlight_labels or [])

    for label, value in values.items():
        if "_" not in label:
            continue
        parent, sub = label.split("_", 1)
        if parent not in parent_index or sub not in sub_index:
            continue
        row = sub_index[sub]
        col = parent_index[parent]
        if np.isfinite(value):
            mat[row, col] = float(value)
        if label in highlight_set:
            highlights.add((row, col))

    masked = np.ma.masked_invalid(mat)
    n_par = max(1, len(parents))
    n_sub = max(1, len(subs))
    fig_w = max(HEATMAP_MIN_W, HEATMAP_FIG_W_SCALE * n_par * 1.4)
    fig_h = max(HEATMAP_MIN_H, HEATMAP_FIG_H_SCALE * n_sub * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    cmap_obj = _get_cmap_copy(cmap)
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(parents)))
    ax.set_xticklabels(formatted_parents, rotation=90, ha="center", fontsize=14)
    ax.set_yticks(np.arange(len(subs)))
    ax.set_yticklabels(formatted_subs, fontsize=14)
    ax.set_xlabel(r"HLA-DQA1$^*$", fontsize=14)
    ax.set_ylabel(r"HLA-DQB1$^*$", fontsize=14)
    ax.set_title(title, fontsize=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=12)

    for y_idx, x_idx in highlights:
        _add_highlight(ax, y_idx, x_idx, lw=0.9)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote value heatmap to {output_pdf}")

    if export_tsv:
        if tsv_path is None:
            tsv_path = output_pdf.with_suffix(".tsv")
        _write_heatmap_tsv(
            tsv_path,
            mat,
            parents,
            subs,
            formatted_parents=formatted_parents,
            formatted_subs=formatted_subs,
        )
        print(f"Wrote value heatmap TSV to {tsv_path}")


def plot_value_heatmap_triptych(
    values_list: Sequence[Mapping[str, float]],
    output_pdf: Path,
    *,
    parents: Sequence[str] | None = None,
    subs: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    cbar_labels: Sequence[str] | None = None,
    cmap: str = "viridis",
    highlight_labels: Sequence[str] | None = None,
    vmins: Sequence[float | None] | None = None,
    vmaxs: Sequence[float | None] | None = None,
    export_tsv: bool = False,
    tsv_paths: Sequence[Path] | None = None,
) -> None:
    """Plot multiple heatmaps side-by-side for per-label scalar values."""
    if not values_list:
        raise ValueError("values_list must contain at least one mapping.")

    if parents is None or subs is None:
        all_labels: list[str] = []
        for values in values_list:
            all_labels.extend(values.keys())
        parents, subs, parent_index, sub_index, formatted_parents, formatted_subs = _build_axes_from_labels(
            all_labels
        )
    else:
        parent_index = {parent: idx for idx, parent in enumerate(parents)}
        sub_index = {sub: idx for idx, sub in enumerate(subs)}
        formatted_parents = [_format_allele_label(lbl) for lbl in parents]
        formatted_subs = [_format_allele_label(lbl) for lbl in subs]

    n_panels = len(values_list)
    titles = list(titles) if titles is not None else [f"Panel {idx + 1}" for idx in range(n_panels)]
    cbar_labels = list(cbar_labels) if cbar_labels is not None else ["Value"] * n_panels
    vmins = list(vmins) if vmins is not None else [None] * n_panels
    vmaxs = list(vmaxs) if vmaxs is not None else [None] * n_panels

    if len(titles) != n_panels or len(cbar_labels) != n_panels:
        raise ValueError("titles and cbar_labels must match values_list length.")
    if len(vmins) != n_panels or len(vmaxs) != n_panels:
        raise ValueError("vmins and vmaxs must match values_list length.")

    mats: list[np.ndarray] = []
    for values in values_list:
        mat = np.full((len(subs), len(parents)), np.nan, dtype=np.float64)
        for label, value in values.items():
            if "_" not in label:
                continue
            parent, sub = label.split("_", 1)
            if parent not in parent_index or sub not in sub_index:
                continue
            row = sub_index[sub]
            col = parent_index[parent]
            if np.isfinite(value):
                mat[row, col] = float(value)
        mats.append(mat)

    highlights: set[tuple[int, int]] = set()
    if highlight_labels:
        for label in highlight_labels:
            if "_" not in label:
                continue
            parent, sub = label.split("_", 1)
            if parent not in parent_index or sub not in sub_index:
                continue
            highlights.add((sub_index[sub], parent_index[parent]))

    n_par = max(1, len(parents))
    n_sub = max(1, len(subs))
    fig_w = max(HEATMAP_MIN_W, HEATMAP_FIG_W_SCALE * n_par * 1.4)
    fig_h = max(HEATMAP_MIN_H, HEATMAP_FIG_H_SCALE * n_sub * 0.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, mat, title, cbar_label, vmin, vmax in zip(axes, mats, titles, cbar_labels, vmins, vmaxs):
        masked = np.ma.masked_invalid(mat)
        cmap_obj = _get_cmap_copy(cmap)
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(parents)))
        ax.set_xticklabels(formatted_parents, rotation=90, ha="center", fontsize=14)
        ax.set_yticks(np.arange(len(subs)))
        ax.set_yticklabels(formatted_subs, fontsize=14)
        ax.set_xlabel(r"HLA-DQA1$^*$", fontsize=14)
        ax.set_title(title, fontsize=16)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=12)

    for ax in axes:
        for y_idx, x_idx in highlights:
            _add_highlight(ax, y_idx, x_idx, lw=0.9)

    axes[0].set_ylabel(r"HLA-DQB1$^*$", fontsize=14)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    print(f"Wrote value heatmap to {output_pdf}")

    if export_tsv:
        if tsv_paths is None:
            tsv_paths = [output_pdf.with_name(f"{output_pdf.stem}_{idx + 1}.tsv") for idx in range(n_panels)]
        if len(tsv_paths) != n_panels:
            raise ValueError("tsv_paths must match values_list length.")
        for mat, tsv_path in zip(mats, tsv_paths):
            _write_heatmap_tsv(
                tsv_path,
                mat,
                parents,
                subs,
                formatted_parents=formatted_parents,
                formatted_subs=formatted_subs,
            )
        print(f"Wrote value heatmap TSV(s) to {output_pdf.parent}")


__all__ = ["plot_value_heatmap", "plot_value_heatmap_triptych"]
