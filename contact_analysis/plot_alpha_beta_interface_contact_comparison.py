#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import argparse
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contact_analysis.plot_clip_core_contact_probabilities import (
    ALPHA_BETA_RANK001_CONTACTS,
    OUT_DIR,
    filter_to_complete_panel,
    map_contacts_to_consensus_axis,
    summarize_contacts,
)
from features import explore_features as featmap


MD_SUMMARY = OUT_DIR / "md_alpha_beta_interface_contact_probability_summary.tsv"
HIGH_RANK_CONTACTS = OUT_DIR / "alpha_beta_highrank901_1000_direct_contacts.tsv"
HIGH_RANK_SUMMARY = OUT_DIR / "alpha_beta_highrank901_1000_contact_probability_summary.tsv"
OUT_MD_RANK1_PDF = OUT_DIR / "alpha_beta_md_vs_rank001_contact_probabilities_threshold0p5.pdf"
OUT_HIGH_RANK_PDF = OUT_DIR / "alpha_beta_highrank901_1000_contact_probabilities_threshold0p5.pdf"
OUT_MD_RANK1_LOW_PDF = OUT_DIR / "alpha_beta_md_vs_rank001_contact_probabilities_below0p5.pdf"
OUT_HIGH_RANK_LOW_PDF = OUT_DIR / "alpha_beta_highrank901_1000_contact_probabilities_below0p5.pdf"

CONTACT_THRESHOLD = 0.50
LOW_CONTACT_MIN = 0.20
BAR_COLOR = "green"
XTICK_FONTSIZE_ALPHA = 15
XTICK_FONTSIZE_BETA = 15
XTICK_FONTSIZE_LOW = 22
LABEL_FONTSIZE = 30
TICK_FONTSIZE = 19
ROW_SPECIFIC_ALPHA_COLOR = "red"
ROW_SPECIFIC_BETA_COLOR = "blue"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MD, rank-1 AF2, and high-rank AF2 alpha/beta interface contact summaries."
    )
    parser.add_argument("--md-summary", type=Path, default=MD_SUMMARY)
    parser.add_argument("--rank1-contacts", type=Path, default=ALPHA_BETA_RANK001_CONTACTS)
    parser.add_argument("--high-rank-contacts", type=Path, default=HIGH_RANK_CONTACTS)
    parser.add_argument("--high-rank-summary", type=Path, default=HIGH_RANK_SUMMARY)
    parser.add_argument("--out-md-rank1-pdf", type=Path, default=OUT_MD_RANK1_PDF)
    parser.add_argument("--out-high-rank-pdf", type=Path, default=OUT_HIGH_RANK_PDF)
    parser.add_argument("--out-md-rank1-low-pdf", type=Path, default=OUT_MD_RANK1_LOW_PDF)
    parser.add_argument("--out-high-rank-low-pdf", type=Path, default=OUT_HIGH_RANK_LOW_PDF)
    return parser.parse_args()


def load_af2_summary(contact_tsv: Path, out_tsv: Path | None = None) -> pd.DataFrame:
    """Map AF2 direct-contact rows to the consensus axes and summarize contacts."""
    if not contact_tsv.exists():
        raise FileNotFoundError(contact_tsv)
    df = pd.read_csv(contact_tsv, sep="\t")
    df = filter_to_complete_panel(df)
    alpha_ref = featmap.load_reference_table(featmap.DEFAULT_ALPHA_REF, "A")
    beta_ref = featmap.load_reference_table(featmap.DEFAULT_BETA_REF, "B")
    mapped = map_contacts_to_consensus_axis(df, alpha_ref, beta_ref)
    summary = summarize_contacts(mapped)
    if out_tsv is not None:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_tsv, sep="\t", index=False)
        print(f"[write] {out_tsv}")
    return summary


def load_md_summary(path: Path = MD_SUMMARY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    return df.rename(
        columns={
            "md_contact_probability": "contact_probability",
            "occupancy_q25": "contact_probability_q25",
            "occupancy_q75": "contact_probability_q75",
            "min_distance_angstrom_median": "median_min_distance_angstrom",
            "min_distance_angstrom_q25": "q25_min_distance_angstrom",
            "min_distance_angstrom_q75": "q75_min_distance_angstrom",
        }
    )


def standardize_af2_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contact_probability_q25"] = np.nan
    out["contact_probability_q75"] = np.nan
    return out


def panel_rows(
    df: pd.DataFrame,
    chain_label: str,
    mode: str = "recurrent",
    low_min: float = 0.0,
) -> pd.DataFrame:
    prob = pd.to_numeric(df["contact_probability"], errors="coerce")
    if mode == "low":
        mask = (prob >= low_min) & (prob < CONTACT_THRESHOLD)
    else:
        mask = prob >= CONTACT_THRESHOLD
    sub = df[(df["chain_label"] == chain_label) & mask].copy()
    return sub.sort_values("residue_sort")


def contact_labels(
    df: pd.DataFrame,
    chain_label: str,
    mode: str = "recurrent",
    low_min: float = 0.0,
) -> set[str]:
    return set(panel_rows(df, chain_label, mode=mode, low_min=low_min)["axis_label"].astype(str))


def row_specific_color(chain_label: str) -> str:
    return ROW_SPECIFIC_ALPHA_COLOR if chain_label == "alpha" else ROW_SPECIFIC_BETA_COLOR


def plot_contact_panel(
    ax,
    df: pd.DataFrame,
    chain_label: str,
    show_xlabel: bool = True,
    highlighted_labels: set[str] | None = None,
    mode: str = "recurrent",
    low_min: float = 0.0,
) -> None:
    sub = panel_rows(df, chain_label, mode=mode, low_min=low_min)
    if sub.empty:
        ax.text(0.5, 0.5, "No contacts", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(sub))
    ax.bar(
        x,
        sub["contact_probability"],
        color=to_rgba(BAR_COLOR, 0.50),
        edgecolor=to_rgba("black", 0.75),
        linewidth=0.45,
        zorder=2,
    )

    if sub["contact_probability_q25"].notna().any():
        y = sub["contact_probability"].to_numpy(float)
        q25 = sub["contact_probability_q25"].to_numpy(float)
        q75 = sub["contact_probability_q75"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=[y - q25, q75 - y],
            fmt="none",
            ecolor=to_rgba("black", 0.35),
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )

    ax.set_ylim(low_min, CONTACT_THRESHOLD) if mode == "low" else ax.set_ylim(CONTACT_THRESHOLD, 1.04)
    ax.set_ylabel("Contact probability")
    chain_symbol = r"$\alpha$" if chain_label == "alpha" else r"$\beta$"
    ax.set_xlabel(f"Consensus residue (chain {chain_symbol})" if show_xlabel else "")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["axis_label"].astype(str), rotation=90, ha="center")
    for label in ax.get_xticklabels():
        if highlighted_labels and label.get_text() in highlighted_labels:
            label.set_fontweight("bold")
            label.set_color(row_specific_color(chain_label))
    xtick_size = XTICK_FONTSIZE_LOW if mode == "low" else (
        XTICK_FONTSIZE_ALPHA if chain_label == "alpha" else XTICK_FONTSIZE_BETA
    )
    ax.tick_params(axis="x", labelsize=xtick_size)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", color="black", alpha=0.25, linestyle="--", lw=0.8)
    ax.set_xlim(-0.4, len(x) - 0.6)

    dist_ax = ax.twinx()
    dist_y = sub["median_min_distance_angstrom"].to_numpy(float)
    dist_q25 = sub["q25_min_distance_angstrom"].to_numpy(float)
    dist_q75 = sub["q75_min_distance_angstrom"].to_numpy(float)
    dist_ax.fill_between(x, dist_q25, dist_q75, color="black", alpha=0.20, linewidth=0, zorder=1)
    dist_ax.plot(
        x,
        dist_y,
        color="black",
        linewidth=1.5,
        marker="o",
        markerfacecolor=to_rgba("black", 0.30),
        markeredgecolor="black",
        markersize=4.2,
        zorder=4,
    )
    dist_ax.set_ylim(0.0, 5.2)
    dist_ax.set_ylabel("Minimum distance [Å]")
    dist_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def write_counts(label: str, df: pd.DataFrame, mode: str = "recurrent", low_min: float = 0.0) -> None:
    prob = pd.to_numeric(df["contact_probability"], errors="coerce")
    sub = df[(prob >= low_min) & (prob < CONTACT_THRESHOLD)] if mode == "low" else df[prob >= CONTACT_THRESHOLD]
    counts = sub.groupby("chain_label").size().to_dict()
    threshold_text = (
        f"{low_min:.2f}<=threshold<{CONTACT_THRESHOLD:.2f}"
        if mode == "low"
        else f"threshold>={CONTACT_THRESHOLD:.2f}"
    )
    print(f"[count] {label} {threshold_text}: {counts} total={len(sub)}")


def plot_md_vs_rank1(
    md: pd.DataFrame,
    rank1: pd.DataFrame,
    mode: str,
    out_pdf: Path,
    md_low_min: float = 0.0,
    af2_low_min: float = 0.0,
) -> None:
    md_only = {
        chain: contact_labels(md, chain, mode=mode, low_min=md_low_min)
        - contact_labels(rank1, chain, mode=mode, low_min=af2_low_min)
        for chain in ("alpha", "beta")
    }
    rank1_only = {
        chain: contact_labels(rank1, chain, mode=mode, low_min=af2_low_min)
        - contact_labels(md, chain, mode=mode, low_min=md_low_min)
        for chain in ("alpha", "beta")
    }
    for chain in ("alpha", "beta"):
        print(
            f"[unique] {chain} MD-only={len(md_only[chain])} "
            f"AF2-rank1-only={len(rank1_only[chain])}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(28.0, 11.5), constrained_layout=True)
    plot_contact_panel(
        axes[0, 0],
        md,
        "alpha",
        show_xlabel=False,
        highlighted_labels=md_only["alpha"],
        mode=mode,
        low_min=md_low_min,
    )
    plot_contact_panel(
        axes[0, 1],
        md,
        "beta",
        show_xlabel=False,
        highlighted_labels=md_only["beta"],
        mode=mode,
        low_min=md_low_min,
    )
    plot_contact_panel(
        axes[1, 0],
        rank1,
        "alpha",
        show_xlabel=True,
        highlighted_labels=rank1_only["alpha"],
        mode=mode,
        low_min=af2_low_min,
    )
    plot_contact_panel(
        axes[1, 1],
        rank1,
        "beta",
        show_xlabel=True,
        highlighted_labels=rank1_only["beta"],
        mode=mode,
        low_min=af2_low_min,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_pdf}")


def plot_high_rank(high_rank: pd.DataFrame, rank1: pd.DataFrame, mode: str, out_pdf: Path) -> None:
    high_only = {
        chain: contact_labels(high_rank, chain, mode=mode) - contact_labels(rank1, chain, mode=mode)
        for chain in ("alpha", "beta")
    }
    for chain in ("alpha", "beta"):
        print(f"[unique] {chain} high-rank-only={len(high_only[chain])}")

    fig, axes = plt.subplots(1, 2, figsize=(28.0, 6.2), constrained_layout=True)
    plot_contact_panel(
        axes[0],
        high_rank,
        "alpha",
        show_xlabel=True,
        highlighted_labels=high_only["alpha"],
        mode=mode,
    )
    plot_contact_panel(
        axes[1],
        high_rank,
        "beta",
        show_xlabel=True,
        highlighted_labels=high_only["beta"],
        mode=mode,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_pdf}")


def main() -> int:
    args = parse_args()
    md = load_md_summary(args.md_summary)
    rank1 = standardize_af2_summary(load_af2_summary(args.rank1_contacts))
    write_counts("MD alpha/beta", md)
    write_counts("AF2 rank 1 alpha/beta", rank1)
    plot_md_vs_rank1(md, rank1, mode="recurrent", out_pdf=args.out_md_rank1_pdf)
    write_counts("MD alpha/beta", md, mode="low", low_min=LOW_CONTACT_MIN)
    write_counts("AF2 rank 1 alpha/beta", rank1, mode="low", low_min=0.0)
    plot_md_vs_rank1(
        md,
        rank1,
        mode="low",
        out_pdf=args.out_md_rank1_low_pdf,
        md_low_min=LOW_CONTACT_MIN,
        af2_low_min=0.0,
    )

    if args.high_rank_contacts.exists():
        high_rank = standardize_af2_summary(load_af2_summary(args.high_rank_contacts, args.high_rank_summary))
        write_counts("AF2 ranks 901-1000 alpha/beta", high_rank)
        plot_high_rank(high_rank, rank1, mode="recurrent", out_pdf=args.out_high_rank_pdf)
        write_counts("AF2 ranks 901-1000 alpha/beta", high_rank, mode="low", low_min=0.0)
        plot_high_rank(high_rank, rank1, mode="low", out_pdf=args.out_high_rank_low_pdf)
    else:
        print(f"[skip] Missing {args.high_rank_contacts}")
        print("[hint] Generate high-rank contacts with extract_alpha_beta_contacts_by_rank.py, then rerun this plotter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
