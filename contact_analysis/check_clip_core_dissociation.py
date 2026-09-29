#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(os.environ.get("HLA_DQ_ROOT", ".")).resolve()
OUT_DIR = ROOT / "reports" / "RMSD_figures"
CHAIN_ALPHA = "A"
CHAIN_BETA = "B"
CHAIN_CLIP = "C"
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


DEFAULT_OUT_TSV = OUT_DIR / "clip_core_residue_boundness_by_rank.tsv"
DEFAULT_SUMMARY_TSV = OUT_DIR / "clip_core_residue_boundness_summary.tsv"
BOUNDNESS_COLUMNS = [
    "heterodimer",
    "rank",
    "pdb_file",
    "clip_position",
    "clip_residue_label",
    "clip_residue_id",
    "clip_residue_name",
    "min_distance_to_alpha_beta_angstrom",
    "is_bound",
    "cutoff_angstrom",
]


def rank_tag(rank: int) -> str:
    return f"rank_{rank:03d}" if rank < 1000 else f"rank_{rank}"


def find_rank_pdb(results_dir: Path, rank: int) -> Path | None:
    matches = sorted(results_dir.glob(f"*{rank_tag(rank)}*_Repair_chained.pdb"))
    return matches[0] if matches else None


def iter_pair_dirs(root: Path, include: set[str] | None, exclude_alpha: set[str]):
    for pair_dir in sorted(root.glob("*.*/*_*")):
        if pair_dir.parent.name in exclude_alpha:
            continue
        if include is not None and pair_dir.name not in include:
            continue
        results_dir = pair_dir / f"{pair_dir.name}_results"
        if results_dir.is_dir():
            yield pair_dir.name, results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each CLIP-core residue remains in contact with the "
            "alpha/beta groove across AF2 ranks."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--heterodimer", nargs="*", default=None)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--clip-core-start", type=int, default=5)
    parser.add_argument("--clip-core-end", type=int, default=13)
    parser.add_argument("--rank-start", type=int, default=1)
    parser.add_argument("--rank-end", type=int, default=1000)
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="Number of heterodimers to process in parallel.",
    )
    parser.add_argument(
        "--write-per-observation-tsv",
        action="store_true",
        help=(
            "Write one row per heterodimer/rank/CLIP-core residue. By default, "
            "only compact summaries are written to avoid large memory and I/O cost."
        ),
    )
    return parser.parse_args()


def atom_is_hydrogen(atom_name: str, element: str) -> bool:
    element = element.strip().upper()
    if element:
        return element in {"H", "D"}
    return atom_name.strip().upper().startswith(("H", "D"))


def fast_read_groove_and_clip_core(
    pdb_path: Path,
    clip_core_start: int,
    clip_core_end: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Read only alpha/beta heavy atoms and chain-C CLIP-core heavy atoms."""
    groove_atoms: list[np.ndarray] = []
    clip_residues: dict[str, dict[str, object]] = {}
    clip_order: list[str] = []

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if atom_is_hydrogen(atom_name, element):
                continue
            chain = line[21].strip()
            if chain not in {CHAIN_ALPHA, CHAIN_BETA, CHAIN_CLIP}:
                continue
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            if chain in {CHAIN_ALPHA, CHAIN_BETA}:
                groove_atoms.append(coord)
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
            f"{pdb_path.name}: invalid CLIP core {clip_core_start}-{clip_core_end} "
            f"for {len(clip_ca_order)} chain-C residues."
        )
    core_keys = set(clip_ca_order[clip_core_start - 1 : clip_core_end])
    core = [
        clip_residues[key]
        for key in clip_ca_order
        if key in core_keys
    ]
    groove = np.array(groove_atoms, dtype=float) if groove_atoms else np.empty((0, 3), dtype=float)
    return groove, core


def min_distance_to_atoms(residue_atoms: list[np.ndarray], target_atoms: np.ndarray, tree: cKDTree | None = None) -> float:
    if not residue_atoms or target_atoms.size == 0:
        return math.nan
    residue_atoms = np.array(residue_atoms, dtype=float)
    if tree is not None:
        distances, _idx = tree.query(residue_atoms, k=1, workers=1)
        return float(np.min(distances))
    delta = residue_atoms[:, None, :] - target_atoms[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=2))))


def boundness_rows(
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    clip_core_start: int,
    clip_core_end: int,
    cutoff: float,
    keep_rows: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groove_atoms, clip_core = fast_read_groove_and_clip_core(
        pdb_path,
        clip_core_start,
        clip_core_end,
    )
    groove_tree = cKDTree(groove_atoms) if groove_atoms.size else None
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for clip_position, residue in zip(
        range(clip_core_start, clip_core_end + 1),
        clip_core,
        strict=True,
    ):
        dist = min_distance_to_atoms(residue["atoms"], groove_atoms, groove_tree)
        resname = str(residue["resname"])
        one = THREE_TO_ONE.get(resname, "X")
        is_bound = bool(math.isfinite(dist) and dist <= cutoff)
        row = {
            "heterodimer": heterodimer,
            "rank": rank,
            "pdb_file": pdb_path.name,
            "clip_position": clip_position,
            "clip_residue_label": f"{one}{clip_position}",
            "clip_residue_id": residue["key"],
            "clip_residue_name": resname,
            "min_distance_to_alpha_beta_angstrom": dist,
            "is_bound": is_bound,
            "cutoff_angstrom": cutoff,
        }
        event_rows.append(row)
        if keep_rows:
            rows.append(row)
    return rows, event_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["clip_position", "clip_residue_label"], as_index=False)
        .agg(
            n_observations=("is_bound", "size"),
            n_bound=("is_bound", "sum"),
            bound_probability=("is_bound", "mean"),
            median_min_distance_angstrom=("min_distance_to_alpha_beta_angstrom", "median"),
            max_min_distance_angstrom=("min_distance_to_alpha_beta_angstrom", "max"),
        )
        .sort_values("clip_position")
    )
    by_rank = (
        df.groupby("rank", as_index=False)
        .agg(
            n_observations=("is_bound", "size"),
            n_bound=("is_bound", "sum"),
            bound_probability=("is_bound", "mean"),
            max_min_distance_angstrom=("min_distance_to_alpha_beta_angstrom", "max"),
        )
        .sort_values("rank")
    )
    grouped["summary_level"] = "clip_residue"
    by_rank["clip_position"] = np.nan
    by_rank["clip_residue_label"] = "all_core_residues"
    by_rank["median_min_distance_angstrom"] = np.nan
    by_rank["summary_level"] = "rank"
    return pd.concat(
        [
            grouped[
                [
                    "summary_level",
                    "clip_position",
                    "clip_residue_label",
                    "n_observations",
                    "n_bound",
                    "bound_probability",
                    "median_min_distance_angstrom",
                    "max_min_distance_angstrom",
                ]
            ],
            by_rank[
                [
                    "summary_level",
                    "clip_position",
                    "clip_residue_label",
                    "n_observations",
                    "n_bound",
                    "bound_probability",
                    "median_min_distance_angstrom",
                    "max_min_distance_angstrom",
                ]
            ],
        ],
        ignore_index=True,
    )


def summarize_from_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "summary_level",
                "clip_position",
                "clip_residue_label",
                "n_observations",
                "n_bound",
                "bound_probability",
                "median_min_distance_angstrom",
                "max_min_distance_angstrom",
            ]
        )
    return summarize(pd.DataFrame(rows))


def process_pair(
    heterodimer: str,
    results_dir: Path,
    ranks: list[int],
    clip_core_start: int,
    clip_core_end: int,
    cutoff: float,
    keep_rows: bool,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    unbound_rows: list[dict[str, object]] = []
    missing = 0
    for rank in ranks:
        pdb_path = find_rank_pdb(results_dir, rank)
        if pdb_path is None:
            missing += 1
            continue
        stored_rows, event_rows = boundness_rows(
            heterodimer,
            pdb_path,
            rank,
            clip_core_start,
            clip_core_end,
            cutoff,
            keep_rows=keep_rows,
        )
        rows.extend(stored_rows)
        all_summary_rows.extend(event_rows)
        unbound_rows.extend([row for row in event_rows if not row["is_bound"]])
    return heterodimer, rows, all_summary_rows, unbound_rows, missing


def main() -> int:
    args = parse_args()
    include = set(args.heterodimer) if args.heterodimer else None
    exclude_alpha = set(args.exclude_alpha or [])
    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    ranks = list(range(args.rank_start, args.rank_end + 1))

    rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    unbound_rows: list[dict[str, object]] = []
    missing = 0
    total = len(pair_dirs)
    print(f"[info] heterodimers={total} ranks={args.rank_start}-{args.rank_end} workers={args.workers}", flush=True)
    if args.workers <= 1:
        for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)
            _heterodimer, stored, summary_part, unbound_part, missing_part = process_pair(
                heterodimer,
                results_dir,
                ranks,
                args.clip_core_start,
                args.clip_core_end,
                args.cutoff_angstrom,
                args.write_per_observation_tsv,
            )
            rows.extend(stored)
            all_summary_rows.extend(summary_part)
            unbound_rows.extend(unbound_part)
            missing += missing_part
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_pair,
                    heterodimer,
                    results_dir,
                    ranks,
                    args.clip_core_start,
                    args.clip_core_end,
                    args.cutoff_angstrom,
                    args.write_per_observation_tsv,
                )
                for heterodimer, results_dir in pair_dirs
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                heterodimer, stored, summary_part, unbound_part, missing_part = future.result()
                rows.extend(stored)
                all_summary_rows.extend(summary_part)
                unbound_rows.extend(unbound_part)
                missing += missing_part
                if idx == 1 or idx % 20 == 0 or idx == total:
                    print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)
    if not all_summary_rows:
        raise SystemExit("No CLIP-core boundness rows were computed.")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    if args.write_per_observation_tsv:
        out_df = pd.DataFrame(rows, columns=BOUNDNESS_COLUMNS)
    else:
        out_df = pd.DataFrame(unbound_rows, columns=BOUNDNESS_COLUMNS)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    summary = summarize_from_rows(all_summary_rows)
    summary.to_csv(args.summary_tsv, sep="\t", index=False)
    print(f"[write] {args.out_tsv}")
    print(f"[write] {args.summary_tsv}")
    total_obs = len(all_summary_rows)
    total_unbound = len(unbound_rows)
    max_min_distance = max(float(row["min_distance_to_alpha_beta_angstrom"]) for row in all_summary_rows)
    print(
        "[summary] "
        f"observations={total_obs} bound_fraction={(1.0 - total_unbound / total_obs):.6f} "
        f"unbound_observations={total_unbound} "
        f"max_min_distance={max_min_distance:.3f} Å"
    )
    if missing:
        print(f"[warn] missing {missing} heterodimer-rank PDB(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
