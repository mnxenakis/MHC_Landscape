"""Shared helpers for publication plotting modules."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .mpl_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from .parameters import get_parameter_defaults

PARAM_DEFAULTS = get_parameter_defaults()

HEATMAP_FIG_W_SCALE = float(PARAM_DEFAULTS["plot_heatmap_fig_w_scale"])
HEATMAP_FIG_H_SCALE = float(PARAM_DEFAULTS["plot_heatmap_fig_h_scale"])
HEATMAP_MIN_W = float(PARAM_DEFAULTS["plot_heatmap_min_w"])
HEATMAP_MIN_H = float(PARAM_DEFAULTS["plot_heatmap_min_h"])


def _get_cmap_copy(name: str, *, bad_color: str = "lightgray") -> mpl.colors.Colormap:
    """Return a copy of a named colormap with the NaN color configured."""
    try:
        if hasattr(mpl, "colormaps"):
            cmap_base = mpl.colormaps.get(name)  # type: ignore[attr-defined]
        else:
            cmap_base = plt.get_cmap(name)
    except Exception:
        cmap_base = plt.get_cmap(name)
    cmap = cmap_base.copy()
    cmap.set_bad(color=bad_color)
    return cmap


def _add_highlight(
    ax: mpl.axes.Axes,
    row: int,
    col: int,
    *,
    color: str = "red",
    lw: float = 1.2,
    corner_radius: float = 0.25,
) -> None:
    """Add a thin rounded highlight around a heatmap cell."""
    try:
        patch = FancyBboxPatch(
            (col - 0.5, row - 0.5),
            1.0,
            1.0,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        )
    except Exception:
        patch = Rectangle(
            (col - 0.5, row - 0.5),
            1.0,
            1.0,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        )
    ax.add_patch(patch)


def _format_allele_label(label: str) -> str:
    """Format allele codes from 'X.Y' into '0X:0Y'."""
    parts = label.split(".")
    if len(parts) != 2:
        return label
    try:
        high = int(parts[0])
        low = int(parts[1])
    except ValueError:
        return label
    return f"{high:02d}:{low:02d}"


def _format_alpha_beta_label(label: str) -> str:
    """Convert 'alpha_beta' into 'AA:BB/CC:DD' with allele padding."""
    alpha, sep, beta = label.partition("_")
    if not sep:
        return _format_allele_label(label)
    return f"{_format_allele_label(alpha)}/{_format_allele_label(beta)}"


def _is_dra_label(label: str) -> bool:
    """Return True when a label belongs to a DRA-containing combination."""
    raw = label.strip()
    if not raw:
        return False
    alpha = raw.split("_", 1)[0].upper()
    if alpha == "DRA":
        return True
    raw_upper = raw.upper()
    return re.search(r"(?:^|[^A-Z0-9])DRA(?:[^A-Z0-9]|$)", raw_upper) is not None


def _build_axes_from_labels(
    labels: Sequence[str],
) -> tuple[list[str], list[str], dict[str, int], dict[str, int], list[str], list[str]]:
    """Build parent/sub axes and indices from labels of the form 'parent_sub'."""
    parents: list[str] = []
    subs: list[str] = []
    for label in sorted(set(labels)):
        if "_" in label:
            parent, sub = label.split("_", 1)
        else:
            parent, sub = label, ""
        if parent not in parents:
            parents.append(parent)
        if sub not in subs:
            subs.append(sub)
    formatted_parents = [_format_allele_label(lbl) for lbl in parents]
    formatted_subs = [_format_allele_label(lbl) for lbl in subs]
    parent_index = {parent: idx for idx, parent in enumerate(parents)}
    sub_index = {sub: idx for idx, sub in enumerate(subs)}
    return parents, subs, parent_index, sub_index, formatted_parents, formatted_subs


def _write_heatmap_tsv(
    output_path: Path,
    matrix: np.ndarray,
    parents: Sequence[str],
    subs: Sequence[str],
    *,
    formatted_parents: Sequence[str] | None = None,
    formatted_subs: Sequence[str] | None = None,
    header_label: str = "DQB1*",
) -> None:
    """Write a heatmap matrix to a TSV file using the provided axis labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    col_labels = list(formatted_parents) if formatted_parents is not None else list(parents)
    row_labels = list(formatted_subs) if formatted_subs is not None else list(subs)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header_label + "\t" + "\t".join(col_labels) + "\n")
        for row_idx, row_label in enumerate(row_labels):
            row_values: list[str] = []
            for col_idx in range(len(parents)):
                value = matrix[row_idx, col_idx]
                row_values.append(f"{value:.6f}" if np.isfinite(value) else "NA")
            handle.write(row_label + "\t" + "\t".join(row_values) + "\n")


__all__ = [
    "HEATMAP_FIG_H_SCALE",
    "HEATMAP_FIG_W_SCALE",
    "HEATMAP_MIN_H",
    "HEATMAP_MIN_W",
    "_add_highlight",
    "_build_axes_from_labels",
    "_format_alpha_beta_label",
    "_format_allele_label",
    "_get_cmap_copy",
    "_is_dra_label",
    "_write_heatmap_tsv",
]
