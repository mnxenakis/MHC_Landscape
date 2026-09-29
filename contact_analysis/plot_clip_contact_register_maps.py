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
    from .clip_contact_register_maps import DEFAULT_OUT_TSV, OUT_DIR
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_contact_register_maps import DEFAULT_OUT_TSV, OUT_DIR


DEFAULT_OUT_PDF = OUT_DIR / "clip_core_contact_register_rank_zone_heatmaps.pdf"
DEFAULT_ZONE_ORDER = [
    "rank 1",
    "ranks 1--100",
    "ranks 101--200",
    "ranks 201--300",
    "ranks 301--400",
    "ranks 401--500",
    "ranks 501--600",
    "ranks 601--700",
    "ranks 701--800",
    "ranks 801--900",
    "ranks 901--1000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CLIP contact-register probabilities as 11 side-by-side heatmaps "
            "covering rank 1 and the AF2 rank windows 1--100 through 901--1000."
        )
    )
    parser.add_argument("--in-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument(
        "--zone-order",
        nargs="+",
        default=DEFAULT_ZONE_ORDER,
        help="Ordered list of rank-zone labels to plot.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input TSV: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {
        "rank_zone",
        "chain_label",
        "axis_label",
        "sort_value",
        "clip_position",
        "clip_residue_label",
        "contact_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
    df["sort_value"] = pd.to_numeric(df["sort_value"], errors="coerce")
    df["clip_position"] = pd.to_numeric(df["clip_position"], errors="coerce")
    df["contact_probability"] = pd.to_numeric(df["contact_probability"], errors="coerce")
    df = df.dropna(subset=["sort_value", "clip_position", "contact_probability"]).copy()
    df["clip_position"] = df["clip_position"].astype(int)
    df["shell_label"] = df["chain_label"].astype(str).map({"alpha": r"$\alpha$", "beta": r"$\beta$"}).fillna("") + ":" + df["axis_label"].astype(str)
    return df


def shell_order(df: pd.DataFrame) -> list[str]:
    order_df = (
        df[["chain_label", "shell_label", "sort_value"]]
        .drop_duplicates()
        .sort_values(["chain_label", "sort_value", "shell_label"])
        .reset_index(drop=True)
    )
    chain_rank = {"alpha": 0, "beta": 1}
    order_df["chain_rank"] = order_df["chain_label"].map(chain_rank).fillna(99)
    order_df = order_df.sort_values(["chain_rank", "sort_value", "shell_label"])
    return order_df["shell_label"].astype(str).tolist()


def clip_order(df: pd.DataFrame) -> list[str]:
    return (
        df[["clip_residue_label", "clip_position"]]
        .drop_duplicates()
        .sort_values("clip_position")["clip_residue_label"]
        .astype(str)
        .tolist()
    )


def pivot_zone(df: pd.DataFrame, rank_zone: str, row_order: list[str], col_order: list[str]) -> pd.DataFrame:
    sub = df[df["rank_zone"] == rank_zone].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for rank zone {rank_zone!r}")
    matrix = sub.pivot_table(
        index="clip_residue_label",
        columns="shell_label",
        values="contact_probability",
        aggfunc="mean",
    )
    return matrix.reindex(index=row_order, columns=col_order)


def plot_heatmaps(df: pd.DataFrame, out_pdf: Path, zone_order: list[str]) -> None:
    present_zones = set(df["rank_zone"].astype(str).unique())
    zones = [zone for zone in zone_order if zone in present_zones]
    if not zones:
        raise SystemExit("None of the requested rank zones were found in the TSV.")

    col_order = shell_order(df)
    row_order = clip_order(df)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(zones),
        figsize=(4.3 * len(zones), 7.8),
        constrained_layout=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes)
    image = None

    for idx, (ax, zone) in enumerate(zip(axes_arr, zones, strict=True)):
        matrix = pivot_zone(df, zone, row_order, col_order)
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(zone, fontsize=18)
        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels(col_order, rotation=90)
        ax.set_yticks(np.arange(len(row_order)))
        if idx == 0:
            ax.set_yticklabels(row_order)
            ax.set_ylabel("CLIP-core residue", fontsize=18)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("HLA groove contact-shell residue", fontsize=18)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes_arr.tolist(), shrink=0.82, pad=0.012, aspect=30)
        cbar.set_label(r"$P_{\mathrm{contact}}$", fontsize=18)
        cbar.ax.tick_params(labelsize=12)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    df = load_table(args.in_tsv)
    plot_heatmaps(df, args.out_pdf, args.zone_order)
    print(f"[write] {args.out_pdf}")
    print(f"[summary] rank_zones={len([z for z in args.zone_order if z in set(df['rank_zone'].astype(str))])}")
    print(f"[summary] shell_residues={df['shell_label'].nunique()}; clip_core_residues={df['clip_residue_label'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
