#!/usr/bin/env python3
"""Infer DQ ectodomain/TMD/CT segment boundaries from a normalized PDB.

The publication FoldX workflow needs chain-local residue ranges for the TMD and
CT portions of chains A and B. This helper scans a chained PDB, finds the known
alpha and beta TMD motifs with a small mismatch tolerance, and writes a compact
JSON file.
"""

from __future__ import annotations

import argparse
import json
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
class ChainResidues:
    chain_id: str
    sequence: str
    residue_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer A/B chain TMD and CT residue ranges from one chained PDB."
    )
    parser.add_argument("--pdb", type=Path, required=True, help="Normalized chained PDB.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--DRalpha",
        action="store_true",
        help="Compatibility flag retained for older workflows; ignored for DQ release data.",
    )
    return parser.parse_args()


def _find_motif(sequence: str, motif: str) -> tuple[int | None, int | None]:
    if len(sequence) < len(motif):
        return None, None
    best_pos: int | None = None
    best_mm: int | None = None
    for start in range(len(sequence) - len(motif) + 1):
        window = sequence[start : start + len(motif)]
        mismatches = sum(1 for a, b in zip(window, motif) if a != b)
        if best_mm is None or mismatches < best_mm:
            best_mm = mismatches
            best_pos = start
    return best_pos, best_mm


def _chain_residues(pdb_path: Path) -> dict[str, ChainResidues]:
    residues_by_chain: dict[str, list[tuple[int, str]]] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain_id = line[21].strip()
            resseq_raw = line[22:26].strip()
            icode = line[26].strip()
            try:
                resseq = int(resseq_raw)
            except ValueError:
                raise SystemExit(f"[error] non-integer residue number {resseq_raw!r} in {pdb_path}")
            key = (chain_id, resseq_raw, icode)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            resname = line[17:20].strip().upper()
            residues_by_chain.setdefault(chain_id, []).append((resseq, THREE_TO_ONE.get(resname, "X")))

    out: dict[str, ChainResidues] = {}
    for chain_id, residues in residues_by_chain.items():
        residues.sort(key=lambda item: item[0])
        residue_ids = [resseq for resseq, _aa in residues]
        sequence = "".join(aa for _resseq, aa in residues)
        out[chain_id] = ChainResidues(chain_id=chain_id, sequence=sequence, residue_ids=residue_ids)
    return out


def _resolve_alpha_beta(chains: dict[str, ChainResidues]) -> tuple[ChainResidues, ChainResidues]:
    if "A" in chains and "B" in chains:
        return chains["A"], chains["B"]

    candidates = list(chains.values())
    if len(candidates) < 2:
        raise SystemExit("[error] expected at least two protein chains in the PDB.")

    alpha_scores: list[tuple[int, int, str]] = []
    beta_scores: list[tuple[int, int, str]] = []
    for chain in candidates:
        _pos_a, mm_a = _find_motif(chain.sequence, TMD_ALPHA)
        _pos_b, mm_b = _find_motif(chain.sequence, TMD_BETA)
        alpha_scores.append((mm_a if mm_a is not None else 999, -len(chain.residue_ids), chain.chain_id))
        beta_scores.append((mm_b if mm_b is not None else 999, -len(chain.residue_ids), chain.chain_id))

    alpha_id = min(alpha_scores)[2]
    remaining = [chain for chain in candidates if chain.chain_id != alpha_id]
    if not remaining:
        raise SystemExit("[error] could not identify distinct alpha and beta chains.")

    beta_id = min(
        (mm if mm is not None else 999, -len(chain.residue_ids), chain.chain_id)
        for chain in remaining
        for _pos, mm in [_find_motif(chain.sequence, TMD_BETA)]
    )[2]

    alpha = chains[alpha_id]
    beta = chains[beta_id]
    _alpha_pos, alpha_mm = _find_motif(alpha.sequence, TMD_ALPHA)
    _beta_pos, beta_mm = _find_motif(beta.sequence, TMD_BETA)
    if alpha_mm is None or alpha_mm > MAX_MISMATCH:
        raise SystemExit(
            f"[error] could not identify alpha-chain TMD motif in {alpha.chain_id} "
            f"(best mismatch={alpha_mm})."
        )
    if beta_mm is None or beta_mm > MAX_MISMATCH:
        raise SystemExit(
            f"[error] could not identify beta-chain TMD motif in {beta.chain_id} "
            f"(best mismatch={beta_mm})."
        )
    return alpha, beta


def _bounds(chain: ChainResidues, motif: str, chain_label: str) -> dict[str, list[int]]:
    start_idx, mismatches = _find_motif(chain.sequence, motif)
    if start_idx is None or mismatches is None or mismatches > MAX_MISMATCH:
        raise SystemExit(
            f"[error] could not identify {chain_label} TMD motif in chained PDB "
            f"(best mismatch={mismatches})."
        )
    end_idx = start_idx + len(motif) - 1
    tmd_start = chain.residue_ids[start_idx]
    tmd_end = chain.residue_ids[end_idx]
    ct_start = chain.residue_ids[end_idx + 1] if end_idx + 1 < len(chain.residue_ids) else tmd_end + 1
    ct_end = chain.residue_ids[-1]
    es_start = chain.residue_ids[0]
    es_end = chain.residue_ids[start_idx - 1] if start_idx > 0 else chain.residue_ids[0] - 1
    return {
        "ES": [es_start, es_end],
        "TMD": [tmd_start, tmd_end],
        "CT": [ct_start, ct_end],
    }


def main() -> int:
    args = parse_args()
    pdb_path = args.pdb.expanduser().resolve()
    if not pdb_path.is_file():
        raise SystemExit(f"[error] chained PDB not found: {pdb_path}")
    residues = _chain_residues(pdb_path)
    alpha_chain, beta_chain = _resolve_alpha_beta(residues)

    data = {
        "A": _bounds(alpha_chain, TMD_ALPHA, f"alpha chain {alpha_chain.chain_id}"),
        "B": _bounds(beta_chain, TMD_BETA, f"beta chain {beta_chain.chain_id}"),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[write] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())