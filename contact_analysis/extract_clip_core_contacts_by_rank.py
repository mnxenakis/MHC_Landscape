#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import pandas as pd

try:
    from .clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        contact_residues_to_clip_core_with_distances,
        contact_rows,
        iter_pair_dirs,
        parse_pdb,
        sequence_and_keys,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import (
        CHAIN_ALPHA,
        CHAIN_BETA,
        CHAIN_CLIP,
        OUT_DIR,
        ROOT,
        contact_residues_to_clip_core_with_distances,
        contact_rows,
        iter_pair_dirs,
        parse_pdb,
        sequence_and_keys,
    )


DEFAULT_RANK_START = 901
DEFAULT_RANK_END = 1000
DEFAULT_OUT_TSV = OUT_DIR / "clip_core_highrank901_1000_direct_contacts.tsv"


def rank_tag(rank: int) -> str:
    return f"rank_{rank:03d}" if rank < 1000 else f"rank_{rank}"


def find_rank_pdb(results_dir: Path, rank: int) -> Path | None:
    matches = sorted(results_dir.glob(f"*{rank_tag(rank)}*_Repair_chained.pdb"))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract alpha/beta residues contacting the canonical CLIP core for "
            "one AF2 rank or a rank window across the HLA-DQ panel."
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
    parser.add_argument("--clip-core-start", type=int, default=5, help="1-based CLIP residue position for core start.")
    parser.add_argument("--clip-core-end", type=int, default=13, help="1-based CLIP residue position for core end.")
    parser.add_argument("--pocket-cutoff-angstrom", type=float, default=5.0)
    return parser.parse_args()


def contact_rows_for_rank(
    heterodimer: str,
    pdb_path: Path,
    rank: int,
    clip_core_start: int,
    clip_core_end: int,
    cutoff_angstrom: float,
) -> list[dict[str, object]]:
    structure = parse_pdb(pdb_path)
    clip_seq, clip_keys = sequence_and_keys(structure, CHAIN_CLIP)
    if not clip_keys:
        raise ValueError(f"{pdb_path.name}: no CLIP chain C C-alpha atoms found.")
    if clip_core_start < 1 or clip_core_end > len(clip_keys) or clip_core_start > clip_core_end:
        raise ValueError(
            f"{pdb_path.name}: invalid CLIP core range {clip_core_start}-{clip_core_end} "
            f"for {len(clip_keys)} chain-C residues."
        )

    clip_core_keys = clip_keys[clip_core_start - 1 : clip_core_end]
    clip_core_sequence = clip_seq[clip_core_start - 1 : clip_core_end]
    rows = (
        contact_rows(
            heterodimer,
            pdb_path,
            "alpha",
            contact_residues_to_clip_core_with_distances(
                structure, CHAIN_ALPHA, clip_core_keys, cutoff_angstrom
            ),
            cutoff_angstrom,
            clip_core_start,
            clip_core_end,
            clip_core_sequence,
        )
        + contact_rows(
            heterodimer,
            pdb_path,
            "beta",
            contact_residues_to_clip_core_with_distances(
                structure, CHAIN_BETA, clip_core_keys, cutoff_angstrom
            ),
            cutoff_angstrom,
            clip_core_start,
            clip_core_end,
            clip_core_sequence,
        )
    )
    for row in rows:
        row["rank"] = rank
    return rows


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
    all_rows: list[dict[str, object]] = []
    missing: list[str] = []

    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    total = len(pair_dirs)
    for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[proc] {idx}/{total}: {heterodimer}", flush=True)
        for rank in ranks:
            pdb = find_rank_pdb(results_dir, rank)
            if pdb is None:
                missing.append(f"{heterodimer}:rank_{rank}")
                continue
            all_rows.extend(
                contact_rows_for_rank(
                    heterodimer,
                    pdb,
                    rank,
                    args.clip_core_start,
                    args.clip_core_end,
                    args.pocket_cutoff_angstrom,
                )
            )

    if not all_rows:
        rank_desc = str(ranks[0]) if len(ranks) == 1 else f"{ranks[0]}-{ranks[-1]}"
        raise SystemExit(f"No rank-{rank_desc} CLIP-core contacts were extracted.")

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
