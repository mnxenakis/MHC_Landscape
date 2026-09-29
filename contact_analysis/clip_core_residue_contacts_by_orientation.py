#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from .clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
    )


DEFAULT_GEOMETRY_TSV = OUT_DIR / "clip_groove_fluctuation_direction.tsv"
DEFAULT_OUT_TSV = OUT_DIR / "clip_core_residue_contacts_by_orientation.tsv"
DEFAULT_SUMMARY_TSV = OUT_DIR / "clip_core_residue_contact_probability_by_orientation.tsv"
DEFAULT_PDF = OUT_DIR / "clip_core_residue_contact_probability_by_orientation.pdf"

DEFAULT_RANK_START = 901
DEFAULT_RANK_END = 1000

GROUP_ORDER = ["rank1", "high-rank preserved", "high-rank reversed"]
GROUP_STYLE = {
    "rank1": ("black", "o"),
    "high-rank preserved": ("#0072b2", "s"),
    "high-rank reversed": ("#d55e00", "^"),
}
RANK_RE = re.compile(r"rank_(1000|[0-9]{3})(?:[^0-9]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CLIP-residue-centered contacts to the alpha and beta "
            "groove sides, split by high-rank CLIP-core orientation."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--geometry-tsv", type=Path, default=DEFAULT_GEOMETRY_TSV)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--heterodimer", nargs="*", default=None)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--clip-core-start", type=int, default=5)
    parser.add_argument("--clip-core-end", type=int, default=13)
    parser.add_argument("--rank-start", type=int, default=DEFAULT_RANK_START)
    parser.add_argument("--rank-end", type=int, default=DEFAULT_RANK_END)
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument(
        "--preserved-min-q",
        type=float,
        default=0.5,
        help="High-rank models with q_orient above this value are treated as orientation-preserved.",
    )
    parser.add_argument(
        "--reversed-max-q",
        type=float,
        default=0.0,
        help="High-rank models with q_orient below this value are treated as orientation-reversed.",
    )
    parser.add_argument("--reuse-tsv", action="store_true")
    return parser.parse_args()


def atom_is_hydrogen(atom_name: str, element: str) -> bool:
    element = element.strip().upper()
    if element:
        return element in {"H", "D"}
    return atom_name.strip().upper().startswith(("H", "D"))


def fast_read_targets_and_clip_core(
    pdb_path: Path,
    clip_core_start: int,
    clip_core_end: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    alpha_atoms: list[np.ndarray] = []
    beta_atoms: list[np.ndarray] = []
    clip_residues: dict[str, dict[str, object]] = {}
    clip_order: list[str] = []

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip()
            if chain not in {CHAIN_ALPHA, CHAIN_BETA, CHAIN_CLIP}:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if atom_is_hydrogen(atom_name, element):
                continue
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            if chain == CHAIN_ALPHA:
                alpha_atoms.append(coord)
                continue
            if chain == CHAIN_BETA:
                beta_atoms.append(coord)
                continue

            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = f"{resseq}{icode}"
            if key not in clip_residues:
                clip_residues[key] = {
                    "key": key,
                    "resname": line[17:20].strip().upper(),
                    "has_ca": False,
                    "atoms": [],
                }
                clip_order.append(key)
            if atom_name == "CA":
                clip_residues[key]["has_ca"] = True
            clip_residues[key]["atoms"].append(coord)

    clip_ca_order = [key for key in clip_order if bool(clip_residues[key]["has_ca"])]
    if clip_core_start < 1 or clip_core_end > len(clip_ca_order) or clip_core_start > clip_core_end:
        raise ValueError(
            f"{pdb_path.name}: invalid CLIP core range {clip_core_start}-{clip_core_end} "
            f"for {len(clip_ca_order)} chain-C residues."
        )
    core_keys = set(clip_ca_order[clip_core_start - 1 : clip_core_end])
    clip_core = [clip_residues[key] for key in clip_ca_order if key in core_keys]
    targets = {
        "alpha": np.array(alpha_atoms, dtype=float) if alpha_atoms else np.empty((0, 3), dtype=float),
        "beta": np.array(beta_atoms, dtype=float) if beta_atoms else np.empty((0, 3), dtype=float),
    }
    return targets, clip_core


def min_distance_to_atoms(residue_atoms: list[np.ndarray], target_atoms: np.ndarray, tree: cKDTree | None) -> float:
    if not residue_atoms or target_atoms.size == 0:
        return math.nan
    residue_atoms = np.array(residue_atoms, dtype=float)
    if tree is not None:
        distances, _idx = tree.query(residue_atoms, k=1, workers=1)
        return float(np.min(distances))
    delta = residue_atoms[:, None, :] - target_atoms[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=2))))


def contact_rows_for_structure(
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    group: str,
    clip_core_start: int,
    clip_core_end: int,
    cutoff_angstrom: float,
) -> list[dict[str, object]]:
    target_atoms, clip_core = fast_read_targets_and_clip_core(pdb_path, clip_core_start, clip_core_end)
    target_trees = {
        side: cKDTree(atoms) if atoms.size else None
        for side, atoms in target_atoms.items()
    }

    rows: list[dict[str, object]] = []
    for offset, residue in enumerate(clip_core, start=clip_core_start):
        resname = str(residue["resname"])
        one_letter = THREE_TO_ONE.get(resname, "X")
        for target_side, atoms in target_atoms.items():
            min_dist = min_distance_to_atoms(residue["atoms"], atoms, target_trees[target_side])
            rows.append(
                {
                    "heterodimer": heterodimer,
                    "rank": rank,
                    "orientation_group": group,
                    "pdb_file": pdb_path.name,
                    "clip_position": offset,
                    "clip_residue_id": residue["key"],
                    "clip_residue_name": resname,
                    "clip_residue_one_letter": one_letter,
                    "clip_residue_label": f"{one_letter}{offset}",
                    "target_side": target_side,
                    "min_distance_angstrom": min_dist,
                    "contact": bool(math.isfinite(min_dist) and min_dist <= cutoff_angstrom),
                    "cutoff_angstrom": cutoff_angstrom,
                }
            )
    return rows


def load_orientation_lookup(args: argparse.Namespace) -> dict[tuple[str, int], str]:
    if not args.geometry_tsv.exists():
        raise SystemExit(f"Missing geometry TSV with q_orient values: {args.geometry_tsv}")
    usecols = ["heterodimer", "rank", "clip_core_orientation_dot"]
    df = pd.read_csv(args.geometry_tsv, sep="\t", usecols=usecols)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["clip_core_orientation_dot"] = pd.to_numeric(df["clip_core_orientation_dot"], errors="coerce")
    df = df.dropna(subset=["rank", "clip_core_orientation_dot"])
    df = df[(df["rank"] >= args.rank_start) & (df["rank"] <= args.rank_end)].copy()

    lookup: dict[tuple[str, int], str] = {}
    for row in df.itertuples(index=False):
        q = float(row.clip_core_orientation_dot)
        if q > args.preserved_min_q:
            group = "high-rank preserved"
        elif q < args.reversed_max_q:
            group = "high-rank reversed"
        else:
            group = "high-rank tilted"
        lookup[(str(row.heterodimer), int(row.rank))] = group
    return lookup


def build_rank_pdb_index(results_dir: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in sorted(results_dir.glob("*_Repair_chained.pdb")):
        match = RANK_RE.search(path.name)
        if not match:
            continue
        rank = int(match.group(1))
        index.setdefault(rank, path)
    return index


def compute_contacts(args: argparse.Namespace) -> pd.DataFrame:
    include = set(args.heterodimer) if args.heterodimer else None
    exclude_alpha = set(args.exclude_alpha or [])
    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    orientation_lookup = load_orientation_lookup(args)

    rows: list[dict[str, object]] = []
    total = len(pair_dirs)
    ranks = range(args.rank_start, args.rank_end + 1)
    for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)

        rank_index = build_rank_pdb_index(results_dir)
        rank1_pdb = rank_index.get(1)
        if rank1_pdb is not None:
            rows.extend(
                contact_rows_for_structure(
                    heterodimer,
                    rank1_pdb,
                    1,
                    "rank1",
                    args.clip_core_start,
                    args.clip_core_end,
                    args.cutoff_angstrom,
                )
            )

        for rank in ranks:
            group = orientation_lookup.get((heterodimer, rank))
            if group is None:
                continue
            pdb_path = rank_index.get(rank)
            if pdb_path is None:
                continue
            rows.extend(
                contact_rows_for_structure(
                    heterodimer,
                    pdb_path,
                    rank,
                    group,
                    args.clip_core_start,
                    args.clip_core_end,
                    args.cutoff_angstrom,
                )
            )

    if not rows:
        raise SystemExit("No CLIP-residue contact rows were computed.")
    return pd.DataFrame(rows).sort_values(
        ["orientation_group", "heterodimer", "rank", "target_side", "clip_position"]
    )


def summarize_contacts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "orientation_group",
                "target_side",
                "clip_position",
                "clip_residue_label",
                "clip_residue_one_letter",
            ],
            as_index=False,
        )
        .agg(
            n_observations=("contact", "size"),
            n_contacts=("contact", "sum"),
            median_min_distance_angstrom=("min_distance_angstrom", "median"),
            q25_min_distance_angstrom=("min_distance_angstrom", lambda x: x.quantile(0.25)),
            q75_min_distance_angstrom=("min_distance_angstrom", lambda x: x.quantile(0.75)),
        )
        .sort_values(["orientation_group", "target_side", "clip_position"])
    )
    grouped["contact_probability"] = grouped["n_contacts"] / grouped["n_observations"]
    return grouped


def plot_summary(summary: pd.DataFrame, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 18,
            "axes.labelsize": 30,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = (
        summary[["clip_position", "clip_residue_label"]]
        .drop_duplicates()
        .sort_values("clip_position")["clip_residue_label"]
        .tolist()
    )
    label_to_x = {label: idx for idx, label in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True, sharey=True)
    for ax, target_side in zip(axes, ["alpha", "beta"]):
        side = summary[summary["target_side"] == target_side]
        for group in GROUP_ORDER:
            sub = side[side["orientation_group"] == group].sort_values("clip_position")
            if sub.empty:
                continue
            color, marker = GROUP_STYLE[group]
            x = [label_to_x[label] for label in sub["clip_residue_label"]]
            ax.plot(
                x,
                sub["contact_probability"],
                color=color,
                lw=2.8,
                marker=marker,
                markersize=8,
                label=group,
            )
        side_label = r"$\alpha$" if target_side == "alpha" else r"$\beta$"
        ax.set_title(f"contacts to {side_label} side", fontsize=28)
        ax.set_xlabel("CLIP-core residue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="0.9", lw=0.8)
    axes[0].set_ylabel(r"$P_{\mathrm{contact}}$")
    axes[1].legend(loc="lower center", frameon=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    print("[summary] contact probabilities for M5 and M13")
    for group in GROUP_ORDER:
        sub = summary[
            (summary["orientation_group"] == group)
            & (summary["clip_residue_label"].isin(["M5", "M13"]))
        ]
        if sub.empty:
            continue
        print(f"\n{group}")
        for row in sub.sort_values(["target_side", "clip_position"]).itertuples(index=False):
            print(
                f"  {row.target_side} {row.clip_residue_label}: "
                f"P={row.contact_probability:.3f}, "
                f"median_d={row.median_min_distance_angstrom:.2f} Å"
            )


def main() -> int:
    args = parse_args()
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_tsv:
        if not args.out_tsv.exists():
            raise SystemExit(f"Missing cached TSV: {args.out_tsv}")
        df = pd.read_csv(args.out_tsv, sep="\t")
    else:
        df = compute_contacts(args)
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")

    summary = summarize_contacts(df)
    summary.to_csv(args.summary_tsv, sep="\t", index=False)
    print(f"[write] {args.summary_tsv}")
    plot_summary(summary, args.out_pdf)
    print_summary(summary)
    print(f"[write] {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
