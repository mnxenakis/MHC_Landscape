#!/usr/bin/env python3
"""Rename chain IDs for residue ranges in a PDB file.

This helper supports the publication FoldX pipeline, which moves CT or TMD
residues out of chains A/B into spare chain IDs before running AnalyseComplex.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename PDB chains over residue ranges.")
    parser.add_argument("--pdb", type=Path, required=True, help="PDB file to edit in place.")
    parser.add_argument("--chain-A-start", type=int, required=True)
    parser.add_argument("--chain-A-end", type=int, required=True)
    parser.add_argument("--chain-B-start", type=int, required=True)
    parser.add_argument("--chain-B-end", type=int, required=True)
    parser.add_argument("--replace_A", required=True, help="Replacement chain ID for matching A residues.")
    parser.add_argument("--replace_B", required=True, help="Replacement chain ID for matching B residues.")
    return parser.parse_args()


def _rewrite_line(line: str, args: argparse.Namespace) -> str:
    if not line.startswith(("ATOM  ", "HETATM", "ANISOU", "TER   ")):
        return line
    chain_id = line[21].strip()
    resseq_raw = line[22:26].strip()
    try:
        resseq = int(resseq_raw)
    except ValueError:
        return line

    new_chain: str | None = None
    if chain_id == "A" and args.chain_A_start <= resseq <= args.chain_A_end:
        new_chain = args.replace_A
    elif chain_id == "B" and args.chain_B_start <= resseq <= args.chain_B_end:
        new_chain = args.replace_B
    if new_chain is None:
        return line
    return f"{line[:21]}{new_chain[0]}{line[22:]}"


def main() -> int:
    args = parse_args()
    pdb_path = args.pdb.expanduser().resolve()
    if not pdb_path.is_file():
        raise SystemExit(f"[error] PDB not found: {pdb_path}")
    text = pdb_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    rewritten = [_rewrite_line(line, args) for line in text]
    pdb_path.write_text("".join(rewritten), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())