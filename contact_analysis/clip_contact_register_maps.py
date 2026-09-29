#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
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
    from .clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
        parse_pdb,
        sequence_and_keys,
    )
    from .extract_clip_core_contacts_by_rank import find_rank_pdb
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
        parse_pdb,
        sequence_and_keys,
    )
    from extract_clip_core_contacts_by_rank import find_rank_pdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import explore_features as featmap


DEFAULT_RANK1_SUMMARY = OUT_DIR / "clip_core_rank001_contact_probability_summary.tsv"
DEFAULT_HIGH_RANK_SUMMARY = OUT_DIR / "clip_core_highrank901_1000_contact_probability_summary.tsv"
DEFAULT_OUT_TSV = OUT_DIR / "clip_core_contact_register_rank_zones.tsv"
DEFAULT_OUT_PDF = OUT_DIR / "clip_core_contact_register_maps.pdf"
DEFAULT_PARTIAL_DIR = OUT_DIR / "clip_core_contact_register_partials"
DEFAULT_ALPHA_REF = featmap.DEFAULT_ALPHA_REF
DEFAULT_BETA_REF = featmap.DEFAULT_BETA_REF


@dataclass(frozen=True)
class ShellResidue:
    chain_label: str
    chain_id: str
    msa_resnum: str
    axis_label: str
    sort_value: float


@dataclass
class Counter:
    n_observations: int = 0
    n_contacts: int = 0
    sum_min_distance: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the CLIP-register specificity of recurrent HLA-DQ "
            "CLIP-core contact residues. The script first defines the HLA-side "
            "contact shell from existing P>=threshold CLIP-core summaries, then "
            "computes which individual CLIP-core residues contact each shell "
            "residue across rank zones."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rank1-summary", type=Path, default=DEFAULT_RANK1_SUMMARY)
    parser.add_argument("--high-rank-summary", type=Path, default=DEFAULT_HIGH_RANK_SUMMARY)
    parser.add_argument("--alpha-ref", type=Path, default=DEFAULT_ALPHA_REF)
    parser.add_argument("--beta-ref", type=Path, default=DEFAULT_BETA_REF)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--heterodimer", nargs="*", default=None)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--clip-core-start", type=int, default=5)
    parser.add_argument("--clip-core-end", type=int, default=13)
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument("--contact-shell-threshold", type=float, default=0.2)
    parser.add_argument(
        "--rank-zones",
        default="1-100,101-200,201-300,301-400,401-500,501-600,601-700,701-800,801-900,901-1000",
        help="Comma-separated rank zones, for example '1-100,401-500,901-1000'.",
    )
    parser.add_argument("--reference-zone", default="rank 1")
    parser.add_argument("--comparison-zone", default="ranks 901--1000")
    parser.add_argument("--reuse-tsv", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force-recompute-partials", action="store_true")
    return parser.parse_args()


def parse_rank_zones(text: str) -> list[tuple[str, range]]:
    zones: list[tuple[str, range]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
        else:
            start = end = int(item)
        if start > end:
            raise SystemExit(f"Invalid rank zone {item!r}: start > end")
        label = f"rank {start}" if start == end else f"ranks {start}--{end}"
        zones.append((label, range(start, end + 1)))
    if not zones:
        raise SystemExit("No rank zones were provided.")
    return zones


def load_contact_shell(args: argparse.Namespace) -> list[ShellResidue]:
    parts = []
    for source_name, path in (("rank1", args.rank1_summary), ("high_rank", args.high_rank_summary)):
        if not path.exists():
            raise SystemExit(f"Missing {source_name} contact summary: {path}")
        df = pd.read_csv(path, sep="\t")
        required = {"chain_label", "msa_resnum", "axis_label", "sort_value", "contact_probability"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
        df["contact_probability"] = pd.to_numeric(df["contact_probability"], errors="coerce")
        df = df[df["contact_probability"] >= args.contact_shell_threshold].copy()
        df["source_summary"] = source_name
        parts.append(df)

    shell_df = pd.concat(parts, ignore_index=True)
    shell_df["msa_resnum"] = shell_df["msa_resnum"].astype(str)
    shell_df["axis_label"] = shell_df["axis_label"].astype(str)
    shell_df["sort_value"] = pd.to_numeric(shell_df["sort_value"], errors="coerce")
    shell_df = (
        shell_df.sort_values(["chain_label", "sort_value"])
        .drop_duplicates(["chain_label", "msa_resnum"])
        .reset_index(drop=True)
    )

    shell: list[ShellResidue] = []
    for row in shell_df.itertuples(index=False):
        chain_label = str(row.chain_label)
        if chain_label == "alpha":
            chain_id = CHAIN_ALPHA
        elif chain_label == "beta":
            chain_id = CHAIN_BETA
        else:
            continue
        shell.append(
            ShellResidue(
                chain_label=chain_label,
                chain_id=chain_id,
                msa_resnum=str(row.msa_resnum),
                axis_label=str(row.axis_label),
                sort_value=float(row.sort_value),
            )
        )
    if not shell:
        raise SystemExit("Contact shell is empty after thresholding.")
    return shell


def rank_to_zone_labels(zones: list[tuple[str, range]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {1: ["rank 1"]}
    for label, ranks in zones:
        for rank in ranks:
            labels = out.setdefault(rank, [])
            if label not in labels:
                labels.append(label)
    return out


def ordered_zone_labels(zones: list[tuple[str, range]]) -> list[str]:
    labels = ["rank 1"]
    for label, _zone in zones:
        if label not in labels:
            labels.append(label)
    return labels


def required_ranks(zones: list[tuple[str, range]]) -> list[int]:
    ranks = {1}
    for _label, zone in zones:
        ranks.update(zone)
    return sorted(ranks)


def residue_mapping_for_heterodimer(
    heterodimer: str,
    shell: list[ShellResidue],
    alpha_ref: featmap.ChainReference,
    beta_ref: featmap.ChainReference,
) -> dict[ShellResidue, int]:
    codes = featmap.parse_heterodimer_codes(heterodimer, alpha_ref, beta_ref)
    out: dict[ShellResidue, int] = {}
    for residue in shell:
        if residue.chain_label == "alpha":
            ref = alpha_ref
            allele = codes.alpha
        else:
            ref = beta_ref
            allele = codes.beta
        seq = ref.allele_sequences[allele]
        match = seq[seq["msa_resnum"].astype(str) == residue.msa_resnum]
        if match.empty:
            continue
        out[residue] = int(match.iloc[0]["source_resnum"])
    return out


def min_distance(residue_atoms: list[np.ndarray], clip_atoms: list[np.ndarray]) -> float:
    if not residue_atoms or not clip_atoms:
        return math.nan
    a = np.array(residue_atoms, dtype=float)
    b = np.array(clip_atoms, dtype=float)
    delta = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=2))))


def init_counters(shell: list[ShellResidue], clip_labels: list[str], zone_labels: list[str]) -> dict[tuple, Counter]:
    counters: dict[tuple, Counter] = {}
    for zone in zone_labels:
        for residue in shell:
            for clip_position, clip_label in enumerate(clip_labels, start=1):
                key = (
                    zone,
                    residue.chain_label,
                    residue.msa_resnum,
                    residue.axis_label,
                    residue.sort_value,
                    clip_position,
                    clip_label,
                )
                counters[key] = Counter()
    return counters


def counters_to_rows(
    counters: dict[tuple, Counter],
    *,
    clip_core_start: int,
    cutoff_angstrom: float,
    contact_shell_threshold: float,
    heterodimer: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, counter in counters.items():
        (
            rank_zone,
            chain_label,
            msa_resnum,
            axis_label,
            sort_value,
            clip_position,
            clip_label,
        ) = key
        prob = counter.n_contacts / counter.n_observations if counter.n_observations else math.nan
        mean_dist = counter.sum_min_distance / counter.n_observations if counter.n_observations else math.nan
        row = {
            "rank_zone": rank_zone,
            "chain_label": chain_label,
            "msa_resnum": msa_resnum,
            "axis_label": axis_label,
            "sort_value": sort_value,
            "clip_position": clip_position + clip_core_start - 1,
            "clip_residue_label": clip_label,
            "n_observations": counter.n_observations,
            "n_contacts": counter.n_contacts,
            "sum_min_distance_angstrom": counter.sum_min_distance,
            "contact_probability": prob,
            "mean_min_distance_angstrom": mean_dist,
            "cutoff_angstrom": cutoff_angstrom,
            "contact_shell_threshold": contact_shell_threshold,
        }
        if heterodimer is not None:
            row = {"heterodimer": heterodimer, **row}
        rows.append(row)
    return rows


def aggregate_partial_tables(partial_dir: Path, zone_labels: list[str]) -> pd.DataFrame:
    paths = sorted(partial_dir.glob("*.tsv"))
    if not paths:
        raise SystemExit(f"No partial TSVs found in {partial_dir}")
    parts = [pd.read_csv(path, sep="\t") for path in paths]
    df = pd.concat(parts, ignore_index=True)
    required = {
        "rank_zone",
        "chain_label",
        "msa_resnum",
        "axis_label",
        "sort_value",
        "clip_position",
        "clip_residue_label",
        "n_observations",
        "n_contacts",
        "sum_min_distance_angstrom",
        "cutoff_angstrom",
        "contact_shell_threshold",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Partial register-map TSVs are missing columns: {sorted(missing)}")

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


def clip_core_labels(structure, clip_core_start: int, clip_core_end: int) -> tuple[list[tuple[str, str]], list[str]]:
    clip_seq, clip_keys = sequence_and_keys(structure, CHAIN_CLIP)
    if clip_core_start < 1 or clip_core_end > len(clip_keys) or clip_core_start > clip_core_end:
        raise ValueError(
            f"Invalid CLIP core range {clip_core_start}-{clip_core_end} for {len(clip_keys)} CLIP residues."
        )
    keys = clip_keys[clip_core_start - 1 : clip_core_end]
    labels = [
        f"{aa}{position}"
        for aa, position in zip(
            clip_seq[clip_core_start - 1 : clip_core_end],
            range(clip_core_start, clip_core_end + 1),
            strict=True,
        )
    ]
    return keys, labels


def update_counters_for_pdb(
    counters: dict[tuple, Counter],
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    zone_labels: list[str],
    shell_mapping: dict[ShellResidue, int],
    clip_core_start: int,
    clip_core_end: int,
    cutoff: float,
) -> None:
    structure = parse_pdb(pdb_path)
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
            for zone in zone_labels:
                key = (
                    zone,
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


def compute_register_maps(args: argparse.Namespace) -> pd.DataFrame:
    zones = parse_rank_zones(args.rank_zones)
    rank_zones = rank_to_zone_labels(zones)
    ranks = required_ranks(zones)
    zone_labels = ordered_zone_labels(zones)

    alpha_ref = featmap.load_reference_table(args.alpha_ref, "A")
    beta_ref = featmap.load_reference_table(args.beta_ref, "B")
    shell = load_contact_shell(args)

    # Use the expected CLIP1 core labels for initialization. Each parsed PDB is
    # still checked against the requested residue range before counters update.
    clip_core = "PVSKMRMATPLLMQA"[args.clip_core_start - 1 : args.clip_core_end]
    clip_labels = [
        f"{aa}{position}"
        for aa, position in zip(
            clip_core,
            range(args.clip_core_start, args.clip_core_end + 1),
            strict=True,
        )
    ]
    include = set(args.heterodimer) if args.heterodimer else None
    exclude_alpha = set(args.exclude_alpha or [])
    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    total = len(pair_dirs)
    args.partial_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
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
        for rank in ranks:
            pdb_path = find_rank_pdb(results_dir, rank)
            if pdb_path is None:
                missing += 1
                continue
            update_counters_for_pdb(
                counters,
                heterodimer,
                pdb_path,
                rank,
                rank_zones.get(rank, []),
                shell_mapping,
                args.clip_core_start,
                args.clip_core_end,
                args.cutoff_angstrom,
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

    if missing:
        print(f"[warn] missing {missing} heterodimer-rank PDB(s)")

    return aggregate_partial_tables(args.partial_dir, zone_labels)


def pivot_matrix(df: pd.DataFrame, rank_zone: str, chain_label: str, value_col: str) -> tuple[pd.DataFrame, list[str], list[str]]:
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
    matrix = sub.pivot_table(
        index="axis_label",
        columns="clip_residue_label",
        values=value_col,
        aggfunc="mean",
    ).reindex(index=row_order, columns=col_order)
    return matrix, row_order, col_order


def plot_register_maps(df: pd.DataFrame, out_pdf: Path, reference_zone: str, comparison_zone: str) -> None:
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

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18.5, 13.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )
    prob_images = []
    delta_images = []
    chain_labels = [("alpha", r"$\alpha$"), ("beta", r"$\beta$")]
    titles = [reference_zone, comparison_zone, f"{comparison_zone} $-$ {reference_zone}"]

    for row_idx, (chain_label, chain_symbol) in enumerate(chain_labels):
        ref, rows, cols = pivot_matrix(df, reference_zone, chain_label, "contact_probability")
        comp, _rows, _cols = pivot_matrix(df, comparison_zone, chain_label, "contact_probability")
        delta = comp - ref
        matrices = [ref, comp, delta]
        for col_idx, matrix in enumerate(matrices):
            ax = axes[row_idx, col_idx]
            if col_idx < 2:
                im = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
                prob_images.append(im)
            else:
                im = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=-1.0, vmax=1.0, cmap="RdBu_r")
                delta_images.append(im)
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

    if prob_images:
        cbar = fig.colorbar(prob_images[0], ax=axes[:, :2], shrink=0.78, pad=0.01)
        cbar.set_label(r"$P_{\mathrm{contact}}$", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
    if delta_images:
        cbar = fig.colorbar(delta_images[0], ax=axes[:, 2], shrink=0.78, pad=0.01)
        cbar.set_label(r"$\Delta P_{\mathrm{contact}}$", fontsize=20)
        cbar.ax.tick_params(labelsize=15)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.partial_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_tsv:
        if not args.out_tsv.exists():
            raise SystemExit(f"Missing cached TSV: {args.out_tsv}")
        df = pd.read_csv(args.out_tsv, sep="\t")
    elif args.aggregate_only:
        zones = parse_rank_zones(args.rank_zones)
        zone_labels = ordered_zone_labels(zones)
        df = aggregate_partial_tables(args.partial_dir, zone_labels)
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")
    else:
        df = compute_register_maps(args)
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")

    plot_register_maps(df, args.out_pdf, args.reference_zone, args.comparison_zone)
    print(f"[write] {args.out_pdf}")
    for chain_label in ("alpha", "beta"):
        n_shell = df[df["chain_label"] == chain_label]["axis_label"].nunique()
        print(f"[summary] {chain_label}: contact-shell residues={n_shell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
