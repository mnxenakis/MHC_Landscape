#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
        DEFAULT_ALPHA_REF,
        DEFAULT_BETA_REF,
        DEFAULT_HIGH_RANK_SUMMARY,
        DEFAULT_RANK1_SUMMARY,
        ROOT,
        ShellResidue,
        aggregate_partial_tables,
        clip_core_labels,
        counters_to_rows,
        init_counters,
        load_contact_shell,
        min_distance,
        residue_mapping_for_heterodimer,
    )
    from .clip_groove_fluctuation_direction import iter_pair_dirs, parse_pdb
    from .extract_clip_core_contacts_by_rank import find_rank_pdb
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_contact_register_maps import (
        DEFAULT_ALPHA_REF,
        DEFAULT_BETA_REF,
        DEFAULT_HIGH_RANK_SUMMARY,
        DEFAULT_RANK1_SUMMARY,
        ROOT,
        ShellResidue,
        aggregate_partial_tables,
        clip_core_labels,
        counters_to_rows,
        init_counters,
        load_contact_shell,
        min_distance,
        residue_mapping_for_heterodimer,
    )
    from clip_groove_fluctuation_direction import iter_pair_dirs, parse_pdb
    from extract_clip_core_contacts_by_rank import find_rank_pdb
from features import explore_features as featmap


DEFAULT_GEOMETRY_TSV = ROOT / "reports" / "RMSD_figures" / "clip_groove_fluctuation_direction.tsv"
DEFAULT_OUT_PDF = ROOT / "reports" / "RMSD_figures" / "clip_core_contact_register_maps_highrank_reversed_only.pdf"
DEFAULT_FILTERED_TSV = ROOT / "reports" / "RMSD_figures" / "clip_core_contact_register_highrank_reversed_only.tsv"
DEFAULT_FILTERED_PARTIAL_DIR = ROOT / "reports" / "RMSD_figures" / "clip_core_contact_register_highrank_reversed_only_partials"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot simplified alpha/beta CLIP-register heatmaps comparing rank 1 "
            "with only those high-rank 901--1000 structures that are orientation-reversed."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--geometry-tsv", type=Path, default=DEFAULT_GEOMETRY_TSV)
    parser.add_argument("--rank1-summary", type=Path, default=DEFAULT_RANK1_SUMMARY)
    parser.add_argument("--high-rank-summary", type=Path, default=DEFAULT_HIGH_RANK_SUMMARY)
    parser.add_argument("--alpha-ref", type=Path, default=DEFAULT_ALPHA_REF)
    parser.add_argument("--beta-ref", type=Path, default=DEFAULT_BETA_REF)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_FILTERED_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_FILTERED_PARTIAL_DIR)
    parser.add_argument("--heterodimer", nargs="*", default=None)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--clip-core-start", type=int, default=5)
    parser.add_argument("--clip-core-end", type=int, default=13)
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument("--contact-shell-threshold", type=float, default=0.2)
    parser.add_argument("--comparison-rank-start", type=int, default=901)
    parser.add_argument("--comparison-rank-end", type=int, default=1000)
    parser.add_argument("--orientation-threshold", type=float, default=0.0)
    parser.add_argument("--reuse-tsv", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force-recompute-partials", action="store_true")
    return parser.parse_args()


def load_reversed_rank_lookup(args: argparse.Namespace) -> dict[tuple[str, int], bool]:
    if not args.geometry_tsv.exists():
        raise SystemExit(f"Missing geometry TSV: {args.geometry_tsv}")
    df = pd.read_csv(args.geometry_tsv, sep="\t", usecols=["heterodimer", "rank", "clip_core_orientation_dot"])
    df["heterodimer"] = df["heterodimer"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["clip_core_orientation_dot"] = pd.to_numeric(df["clip_core_orientation_dot"], errors="coerce")
    df = df.dropna(subset=["rank", "clip_core_orientation_dot"]).copy()
    df = df[(df["rank"] >= args.comparison_rank_start) & (df["rank"] <= args.comparison_rank_end)].copy()
    return {
        (str(row.heterodimer), int(row.rank)): bool(float(row.clip_core_orientation_dot) < args.orientation_threshold)
        for row in df.itertuples(index=False)
    }


def reversed_cohort(lookup: dict[tuple[str, int], bool]) -> set[str]:
    out: set[str] = set()
    for (heterodimer, _rank), is_reversed in lookup.items():
        if is_reversed:
            out.add(heterodimer)
    return out


def update_counter_block(
    counters,
    structure,
    shell_mapping: dict[ShellResidue, int],
    shell: list[ShellResidue],
    clip_core_start: int,
    clip_core_end: int,
    cutoff: float,
    zone_label: str,
) -> None:
    clip_keys, clip_labels = clip_core_labels(structure, clip_core_start, clip_core_end)
    clip_residues = [structure.by_key.get(key) for key in clip_keys]

    for shell_residue, source_resnum in shell_mapping.items():
        hla_residue = structure.by_key.get((shell_residue.chain_id, str(source_resnum)))
        if hla_residue is None or not hla_residue.heavy:
            continue
        for clip_position, (clip_label, clip_residue) in enumerate(zip(clip_labels, clip_residues, strict=True), start=1):
            if clip_residue is None or not clip_residue.heavy:
                continue
            dist = min_distance(hla_residue.heavy, clip_residue.heavy)
            is_contact = math.isfinite(dist) and dist <= cutoff
            key = (
                zone_label,
                shell_residue.chain_label,
                shell_residue.msa_resnum,
                shell_residue.axis_label,
                shell_residue.sort_value,
                clip_position,
                clip_label,
            )
            counter = counters[key]
            counter.n_observations += 1
            counter.sum_min_distance += dist
            if is_contact:
                counter.n_contacts += 1


def compute_state_conditioned_table(args: argparse.Namespace) -> pd.DataFrame:
    zone_labels = ["rank 1", f"ranks {args.comparison_rank_start}--{args.comparison_rank_end} reversed"]
    lookup = load_reversed_rank_lookup(args)
    include = reversed_cohort(lookup)
    if args.heterodimer:
        include &= set(args.heterodimer)
    if not include:
        raise SystemExit("No heterodimers have reversed states in the selected high-rank window.")

    alpha_ref = featmap.load_reference_table(args.alpha_ref, "A")
    beta_ref = featmap.load_reference_table(args.beta_ref, "B")
    shell = load_contact_shell(args)
    clip_core = "PVSKMRMATPLLMQA"[args.clip_core_start - 1 : args.clip_core_end]
    clip_labels = [
        f"{aa}{position}"
        for aa, position in zip(clip_core, range(args.clip_core_start, args.clip_core_end + 1), strict=True)
    ]

    pair_dirs = list(iter_pair_dirs(args.root, include, set(args.exclude_alpha or [])))
    total = len(pair_dirs)
    args.partial_dir.mkdir(parents=True, exist_ok=True)

    for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
        partial_path = args.partial_dir / f"{heterodimer}.tsv"
        if partial_path.exists() and not args.force_recompute_partials:
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"[skip] {idx}/{total}: {heterodimer}", flush=True)
            continue
        if idx == 1 or idx % 20 == 0 or idx == total:
            print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)
        counters = init_counters(shell, clip_labels, zone_labels)
        shell_mapping = residue_mapping_for_heterodimer(heterodimer, shell, alpha_ref, beta_ref)

        rank1_pdb = find_rank_pdb(results_dir, 1)
        if rank1_pdb is not None:
            structure = parse_pdb(rank1_pdb)
            update_counter_block(
                counters,
                structure,
                shell_mapping,
                shell,
                args.clip_core_start,
                args.clip_core_end,
                args.cutoff_angstrom,
                "rank 1",
            )

        for rank in range(args.comparison_rank_start, args.comparison_rank_end + 1):
            if not lookup.get((heterodimer, rank), False):
                continue
            pdb_path = find_rank_pdb(results_dir, rank)
            if pdb_path is None:
                continue
            structure = parse_pdb(pdb_path)
            update_counter_block(
                counters,
                structure,
                shell_mapping,
                shell,
                args.clip_core_start,
                args.clip_core_end,
                args.cutoff_angstrom,
                f"ranks {args.comparison_rank_start}--{args.comparison_rank_end} reversed",
            )

        partial_df = pd.DataFrame(
            counters_to_rows(
                counters,
                clip_core_start=args.clip_core_start,
                cutoff_angstrom=args.cutoff_angstrom,
                contact_shell_threshold=args.contact_shell_threshold,
                heterodimer=heterodimer,
            )
        )
        tmp_path = partial_path.with_suffix(".tmp")
        partial_df.to_csv(tmp_path, sep="\t", index=False)
        tmp_path.replace(partial_path)

    return aggregate_partial_tables(args.partial_dir, zone_labels)


def pivot_matrix(df: pd.DataFrame, rank_zone: str, chain_label: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    sub = df[(df["rank_zone"] == rank_zone) & (df["chain_label"] == chain_label)].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for rank_zone={rank_zone!r}, chain_label={chain_label!r}")
    row_order = (
        sub[["axis_label", "sort_value"]]
        .drop_duplicates()
        .sort_values("sort_value")["axis_label"]
        .astype(str)
        .tolist()
    )
    col_order = (
        sub[["clip_residue_label", "clip_position"]]
        .drop_duplicates()
        .sort_values("clip_position")["clip_residue_label"]
        .astype(str)
        .tolist()
    )
    matrix = sub.pivot_table(index="axis_label", columns="clip_residue_label", values="contact_probability", aggfunc="mean")
    return matrix.reindex(index=row_order, columns=col_order), row_order, col_order


def plot_register_maps(df: pd.DataFrame, out_pdf: Path, comparison_zone: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 13.0), constrained_layout=True)
    chain_labels = [("alpha", r"$\alpha$"), ("beta", r"$\beta$")]
    titles = ["rank 1", comparison_zone]
    images = []

    for row_idx, (chain_label, chain_symbol) in enumerate(chain_labels):
        ref, rows, cols = pivot_matrix(df, "rank 1", chain_label)
        comp, _rows, _cols = pivot_matrix(df, comparison_zone, chain_label)
        for col_idx, matrix in enumerate([ref, comp]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            images.append(im)
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=21)
            if col_idx == 0:
                ax.set_ylabel(f"chain {chain_symbol}\nHLA contact-shell residue", fontsize=20)
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels(rows)
            else:
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels([])
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels(cols, rotation=90)
            ax.set_xlabel("CLIP-core residue", fontsize=20)
            ax.tick_params(axis="both", length=0)
            ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.65)

    if images:
        cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
        cbar.set_label(r"$P_{\mathrm{contact}}$", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    comparison_zone = f"ranks {args.comparison_rank_start}--{args.comparison_rank_end} reversed"
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.partial_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_tsv:
        if not args.out_tsv.exists():
            raise SystemExit(f"Missing cached TSV: {args.out_tsv}")
        df = pd.read_csv(args.out_tsv, sep="\t")
    elif args.aggregate_only:
        df = aggregate_partial_tables(args.partial_dir, ["rank 1", comparison_zone])
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")
    else:
        df = compute_state_conditioned_table(args)
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")
    plot_register_maps(df, args.out_pdf, comparison_zone)
    print(f"[write] {args.out_pdf}")
    print(f"[summary] comparison_zone={comparison_zone}")
    for chain_label in ("alpha", "beta"):
        n_shell = df[df["chain_label"] == chain_label]["axis_label"].nunique()
        print(f"[summary] {chain_label}: contact-shell residues={n_shell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
