#!/usr/bin/env python3
"""Normalize repaired DQ complex chains to A/B/C.

This helper rewrites one repaired trimer PDB so that:

- chain A = HLA-DQ alpha
- chain B = HLA-DQ beta
- chain C = peptide

The intended input is a full-length DQ alpha/beta/peptide model produced by
ColabFold/AlphaFold2 and optionally repaired by FoldX. The script writes the
normalized file next to the input as ``*_chained.pdb``.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


TMD_ALPHA = "TVVCALGLSVGLVGIVVGTVFII"
TMD_BETA = "MLSGIGGFVLGLIFLGLGLII"
MAX_MISMATCH = 2

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


@dataclass(frozen=True)
class ChainInfo:
    chain_id: str
    sequence: str
    residue_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename one repaired trimer PDB to the A/B/C chain convention."
    )
    parser.add_argument("pdb", type=Path, help="Input repaired PDB.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Default: <input>_chained.pdb",
    )
    return parser.parse_args()


def _best_motif_match(sequence: str, motif: str) -> int | None:
    if len(sequence) < len(motif):
        return None
    best: int | None = None
    for start in range(len(sequence) - len(motif) + 1):
        window = sequence[start : start + len(motif)]
        mismatches = sum(1 for a, b in zip(window, motif) if a != b)
        if best is None or mismatches < best:
            best = mismatches
    return best


def _chain_infos(pdb_path: Path) -> list[ChainInfo]:
    residues_by_chain: dict[str, list[str]] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            resname = line[17:20].strip().upper()
            residues_by_chain.setdefault(chain_id, []).append(THREE_TO_ONE.get(resname, "X"))
    infos = [
        ChainInfo(chain_id=chain_id, sequence="".join(seq), residue_count=len(seq))
        for chain_id, seq in sorted(residues_by_chain.items())
        if seq
    ]
    if len(infos) < 3:
        raise SystemExit(f"[error] expected at least 3 non-empty chains in {pdb_path}")
    return infos


def choose_mapping(chain_infos: list[ChainInfo]) -> dict[str, str]:
    peptide = min(chain_infos, key=lambda info: (info.residue_count, info.chain_id))
    non_peptide = [info for info in chain_infos if info.chain_id != peptide.chain_id]
    if len(non_peptide) < 2:
        raise SystemExit("[error] could not identify two non-peptide chains.")

    alpha_candidates: list[tuple[int, int, str]] = []
    beta_candidates: list[tuple[int, int, str]] = []
    for info in non_peptide:
        alpha_mm = _best_motif_match(info.sequence, TMD_ALPHA)
        beta_mm = _best_motif_match(info.sequence, TMD_BETA)
        alpha_candidates.append((alpha_mm if alpha_mm is not None else 999, -info.residue_count, info.chain_id))
        beta_candidates.append((beta_mm if beta_mm is not None else 999, -info.residue_count, info.chain_id))

    alpha_chain = min(alpha_candidates)[2]
    remaining = [info for info in non_peptide if info.chain_id != alpha_chain]
    if not remaining:
        raise SystemExit("[error] alpha/beta chain assignment collapsed to one chain.")

    beta_scores = []
    for info in remaining:
        beta_mm = _best_motif_match(info.sequence, TMD_BETA)
        beta_scores.append((beta_mm if beta_mm is not None else 999, -info.residue_count, info.chain_id))
    beta_chain = min(beta_scores)[2]

    alpha_mm = next(score for score, _neg_len, chain_id in alpha_candidates if chain_id == alpha_chain)
    beta_mm = next(score for score, _neg_len, chain_id in beta_scores if chain_id == beta_chain)
    if alpha_mm > MAX_MISMATCH:
        raise SystemExit(
            f"[error] could not confidently identify alpha chain (best TMD mismatch={alpha_mm})."
        )
    if beta_mm > MAX_MISMATCH:
        raise SystemExit(
            f"[error] could not confidently identify beta chain (best TMD mismatch={beta_mm})."
        )

    return {
        alpha_chain: "A",
        beta_chain: "B",
        peptide.chain_id: "C",
    }


def rewrite_chains(input_pdb: Path, output_pdb: Path, mapping: dict[str, str]) -> None:
    with input_pdb.open("r", encoding="utf-8", errors="replace") as src, output_pdb.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith(("ATOM  ", "HETATM", "ANISOU", "TER   ")):
                old_chain = line[21].strip() or "_"
                new_chain = mapping.get(old_chain)
                if new_chain is not None:
                    line = f"{line[:21]}{new_chain}{line[22:]}"
            dst.write(line)


def main() -> int:
    args = parse_args()
    input_pdb = args.pdb.expanduser().resolve()
    if not input_pdb.is_file():
        raise SystemExit(f"[error] input PDB not found: {input_pdb}")

    output_pdb = args.out.expanduser().resolve() if args.out else input_pdb.with_name(f"{input_pdb.stem}_chained.pdb")
    infos = _chain_infos(input_pdb)
    mapping = choose_mapping(infos)

    desired = {"A", "B", "C"}
    if set(mapping.values()) == desired and mapping == {key: key for key in mapping}:
        shutil.copyfile(input_pdb, output_pdb)
    else:
        rewrite_chains(input_pdb, output_pdb, mapping)

    print(f"[write] {output_pdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())