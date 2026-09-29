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

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from .clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
        parse_pdb,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        OUT_DIR,
        ROOT,
        THREE_TO_ONE,
        iter_pair_dirs,
        parse_pdb,
    )


DEFAULT_RANK_START = 1000
DEFAULT_RANK_END = 1000
DEFAULT_OUT_TSV = OUT_DIR / "alpha_beta_rank1000_direct_contacts.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract alpha/beta scaffold residue contacts for one AF2 rank or "
            "a rank window across the HLA-DQ panel."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Single AF2 rank to extract. Overrides --rank-start/--rank-end when set.",
    )
    parser.add_argument("--rank-start", type=int, default=DEFAULT_RANK_START)
    parser.add_argument("--rank-end", type=int, default=DEFAULT_RANK_END)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    parser.add_argument("--heterodimer", nargs="*", default=None, help="Optional heterodimer ids such as 5.1_2.1.")
    parser.add_argument("--exclude-alpha", nargs="*", default=["5.2"], help="Alpha-chain parent dirs to skip.")
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    return parser.parse_args()


def rank_tag(rank: int) -> str:
    return f"rank_{rank:03d}" if rank < 1000 else f"rank_{rank}"


def find_rank_pdb(results_dir: Path, rank: int) -> Path | None:
    matches = sorted(results_dir.glob(f"*{rank_tag(rank)}*_Repair_chained.pdb"))
    return matches[0] if matches else None


def residue_contacts_to_chain_with_distances(
    structure,
    source_chain: str,
    target_chain: str,
    cutoff: float,
) -> list[tuple[object, float]]:
    """Return source-chain residues whose heavy atoms approach target-chain heavy atoms."""
    target_atoms = [
        atom
        for residue in structure.chains.get(target_chain, [])
        for atom in residue.heavy
    ]
    if not target_atoms:
        return []

    target_tree = cKDTree(np.asarray(target_atoms, dtype=float))
    contacts: list[tuple[object, float]] = []
    for residue in structure.chains.get(source_chain, []):
        if not residue.heavy:
            continue
        dists, _idx = target_tree.query(np.asarray(residue.heavy, dtype=float), k=1)
        min_dist = float(np.min(dists))
        if math.isfinite(min_dist) and min_dist <= cutoff:
            contacts.append((residue, min_dist))
    return contacts


def rows_for_contacts(
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    chain_label: str,
    target_label: str,
    contacts: list[tuple[object, float]],
    cutoff: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for residue, min_distance in contacts:
        rows.append(
            {
                "heterodimer": heterodimer,
                "rank": rank,
                "reference_pdb": pdb_path.name,
                "chain": residue.chain,
                "chain_label": chain_label,
                "target_chain_label": target_label,
                "residue_id": residue.key,
                "residue_name": residue.resname,
                "residue_one_letter": THREE_TO_ONE.get(residue.resname, "X"),
                "min_distance_angstrom": min_distance,
                "cutoff_angstrom": cutoff,
            }
        )
    return rows


def contact_rows_for_rank(
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    cutoff: float,
) -> list[dict[str, object]]:
    structure = parse_pdb(pdb_path)
    return (
        rows_for_contacts(
            heterodimer,
            pdb_path,
            rank,
            "alpha",
            "beta",
            residue_contacts_to_chain_with_distances(structure, CHAIN_ALPHA, CHAIN_BETA, cutoff),
            cutoff,
        )
        + rows_for_contacts(
            heterodimer,
            pdb_path,
            rank,
            "beta",
            "alpha",
            residue_contacts_to_chain_with_distances(structure, CHAIN_BETA, CHAIN_ALPHA, cutoff),
            cutoff,
        )
    )


def main() -> int:
    args = parse_args()
    if args.rank is not None:
        ranks = [args.rank]
    else:
        if args.rank_start > args.rank_end:
            raise SystemExit("--rank-start cannot be larger than --rank-end")
        ranks = list(range(args.rank_start, args.rank_end + 1))

    include = set(args.heterodimer) if args.heterodimer else None
    exclude_alpha = set(args.exclude_alpha or [])
    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    all_rows: list[dict[str, object]] = []
    missing: list[str] = []

    total = len(pair_dirs)
    for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)
        for rank in ranks:
            pdb = find_rank_pdb(results_dir, rank)
            if pdb is None:
                missing.append(f"{heterodimer}:rank_{rank}")
                continue
            all_rows.extend(contact_rows_for_rank(heterodimer, pdb, rank, args.cutoff_angstrom))

    if not all_rows:
        rank_desc = str(ranks[0]) if len(ranks) == 1 else f"{ranks[0]}-{ranks[-1]}"
        raise SystemExit(f"No rank-{rank_desc} alpha/beta contacts were extracted.")

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["heterodimer", "rank", "chain", "residue_id"]).reset_index(drop=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[write] {args.out_tsv}")
    print(
        "[summary] "
        f"rank_start={min(ranks)} rank_end={max(ranks)} ranks={len(ranks)} "
        f"heterodimers={out['heterodimer'].nunique()} contact_rows={len(out)}"
    )
    if missing:
        print(f"[warn] missing {len(missing)} heterodimer-rank PDB(s): {', '.join(missing[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
