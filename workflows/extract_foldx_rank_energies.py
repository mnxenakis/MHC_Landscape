#!/usr/bin/env python3
"""Extract per-rank FoldX energies from one *_results directory.

Run this script inside an ``X.Y_Z.W_results`` directory, or pass the directory
with ``--results-dir``.  The output TSV is written one level up, in ``X.Y_Z.W``.

The script is designed for ``FoldXPipeline.sh`` output names, but it also keeps
small legacy fallbacks for older component names where the meaning is
unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

KCAL_TO_KJ = 4.184
RANK_RE = re.compile(r"rank_(1000|\d{3})(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write one parent-level TSV containing FoldX energies per AF2 rank."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing rank-ordered PDBs and FoldX outputs.",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Optional output path. Default: ../<parent-dir-name>_FoldXEnergies.tsv",
    )
    return parser.parse_args()


def rank_from_name(name: str) -> int | None:
    match = RANK_RE.search(name)
    if not match:
        return None
    return int(match.group(1))


def is_raw_pdb(path: Path) -> bool:
    """Return True only for original AF2/ColabFold model PDBs.

    FoldXPipeline.sh creates many derived PDB files in the same directory.  We
    ignore those here so each row corresponds to exactly one rank-ordered input
    model.
    """

    name = path.name
    if "_Repair" in name:
        return False
    derived_suffixes = (
        "_Repair.pdb",
        "_Repair_chained.pdb",
        "_chained.pdb",
        "_alpha_whole.pdb",
        "_beta_whole.pdb",
        "_alpha_chainA.pdb",
        "_beta_chainB.pdb",
        "_CLIP_chainC.pdb",
        "_AB_CT_off.pdb",
        "_AB_CT_TMD_off.pdb",
    )
    return path.suffix == ".pdb" and not name.endswith(derived_suffixes)


def find_ranked_raw_pdbs(results_dir: Path) -> dict[int, Path]:
    """Map AF2 rank -> raw input PDB and fail on missing/duplicate rank tokens."""

    ranked: dict[int, Path] = {}
    bad_names: list[str] = []
    duplicates: list[str] = []

    for pdb in sorted(results_dir.glob("*.pdb")):
        if not is_raw_pdb(pdb):
            continue
        rank = rank_from_name(pdb.name)
        if rank is None:
            bad_names.append(pdb.name)
            continue
        if rank in ranked:
            duplicates.append(f"rank_{rank:03d}: {ranked[rank].name} | {pdb.name}")
            continue
        ranked[rank] = pdb

    if bad_names:
        joined = "\n  ".join(bad_names[:20])
        raise SystemExit(
            "[error] raw PDB files without rank_001...rank_1000 token:\n  "
            + joined
        )
    if duplicates:
        joined = "\n  ".join(duplicates[:20])
        raise SystemExit("[error] duplicate raw PDB ranks:\n  " + joined)
    if not ranked:
        raise SystemExit(f"[error] no rank-ordered raw PDB files found in {results_dir}")

    return ranked


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def find_single_glob(results_dir: Path, patterns: list[str]) -> Path | None:
    """Return a single glob match, or None if no unique match exists."""

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(results_dir.glob(pattern)))
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def parse_stability(path: Path | None) -> float:
    """Read FoldX Stability energy from the first numeric column after PDB name."""

    if path is None:
        return math.nan
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                return float(parts[1])
            except ValueError:
                continue
    return math.nan


def parse_interaction_summary(path: Path | None) -> float:
    """Read the ``Interaction Energy`` column from a FoldX Summary_*.fxout file."""

    if path is None:
        return math.nan

    header: list[str] | None = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if "Interaction Energy" in parts:
                header = parts
                continue
            if header is None:
                continue
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts, strict=False))
            try:
                return float(row["Interaction Energy"])
            except (KeyError, ValueError):
                return math.nan

    return math.nan


def add_units(row: dict[str, object], name: str, kcal_value: float) -> None:
    """Store both native FoldX kcal/mol and converted kJ/mol values."""

    row[f"{name}_kcal_mol"] = kcal_value
    row[f"{name}_kj_mol"] = kcal_value * KCAL_TO_KJ if math.isfinite(kcal_value) else math.nan


def rank_token(rank: int) -> str:
    return f"rank_{rank:03d}" if rank < 1000 else "rank_1000"


def collect_row(results_dir: Path, rank: int, raw_pdb: Path) -> dict[str, object]:
    """Collect all energies for one rank from predictable FoldXPipeline outputs."""

    raw_base = raw_pdb.stem
    tag_base = f"{raw_base}_Repair"
    rank_tag = rank_token(rank)

    row: dict[str, object] = {
        "rank": rank,
        "rank_label": rank_tag,
        "raw_pdb": raw_pdb.name,
    }

    stability_files = {
        "total_stability": [
            results_dir / f"{tag_base}_chained_0_ST.fxout",
            results_dir / f"{tag_base}_0_ST.fxout",
        ],
        "alpha_stability_norepair": [
            results_dir / f"{tag_base}_alpha_chainA_NoRepair_0_ST.fxout",
            results_dir / f"{tag_base}_alpha_whole_NoRepair_0_ST.fxout",
        ],
        "beta_stability_norepair": [
            results_dir / f"{tag_base}_beta_chainB_NoRepair_0_ST.fxout",
            results_dir / f"{tag_base}_beta_whole_NoRepair_0_ST.fxout",
        ],
        "clip_stability_norepair": [
            results_dir / f"{tag_base}_CLIP_chainC_NoRepair_0_ST.fxout",
        ],
        "alpha_stability_repaired": [
            results_dir / f"{tag_base}_alpha_chainA_Repair_0_ST.fxout",
            results_dir / f"{tag_base}_alpha_whole_Repair_0_ST.fxout",
        ],
        "beta_stability_repaired": [
            results_dir / f"{tag_base}_beta_chainB_Repair_0_ST.fxout",
            results_dir / f"{tag_base}_beta_whole_Repair_0_ST.fxout",
        ],
        "clip_stability_repaired": [
            results_dir / f"{tag_base}_CLIP_chainC_Repair_0_ST.fxout",
        ],
    }

    for name, paths in stability_files.items():
        # The glob fallbacks cover older quick-test outputs named only by rank.
        if "clip_stability" in name:
            mode = "NoRepair" if name.endswith("norepair") else "Repair"
            path = first_existing(paths) or find_single_glob(
                results_dir, [f"{rank_tag}_CLIP_chainC_{mode}_0_ST.fxout"]
            )
        else:
            path = first_existing(paths)
        add_units(row, name, parse_stability(path))

    interaction_files = {
        "ab_interaction_whole": [
            results_dir / f"{tag_base}_AB_whole_Summary.fxout",
        ],
        "ab_interaction_ct_off": [
            results_dir / f"{tag_base}_AB_CT_off_Summary.fxout",
        ],
        "ac_interaction_whole": [
            results_dir / f"{tag_base}_AC_whole_Summary.fxout",
        ],
        "bc_interaction_whole": [
            results_dir / f"{tag_base}_BC_whole_Summary.fxout",
        ],
        "ab_interaction_ct_tmd_off": [
            results_dir / f"{tag_base}_AB_CT_TMD_off_Summary.fxout",
        ],
        "ab_interaction_tmd_only": [
            results_dir / f"{tag_base}_AB_TMD_only_Summary.fxout",
        ],
    }

    legacy_globs = {
        "ab_interaction_whole": [f"{rank_tag}_A_B_Summary.fxout"],
        "ac_interaction_whole": [f"{rank_tag}_A_C_Summary.fxout"],
        "bc_interaction_whole": [f"{rank_tag}_B_C_Summary.fxout"],
    }
    for name, paths in interaction_files.items():
        path = first_existing(paths)
        if path is None:
            path = find_single_glob(results_dir, legacy_globs.get(name, []))
        add_units(row, name, parse_interaction_summary(path))

    return row


def write_tsv(rows: list[dict[str, object]], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    fixed = ["rank", "rank_label", "raw_pdb"]
    energy_names = [
        "total_stability",
        "alpha_stability_norepair",
        "beta_stability_norepair",
        "clip_stability_norepair",
        "alpha_stability_repaired",
        "beta_stability_repaired",
        "clip_stability_repaired",
        "ab_interaction_whole",
        "ab_interaction_ct_off",
        "ac_interaction_whole",
        "bc_interaction_whole",
        "ab_interaction_ct_tmd_off",
        "ab_interaction_tmd_only",
    ]
    columns = fixed + [
        f"{name}_{unit}" for name in energy_names for unit in ("kcal_mol", "kj_mol")
    ]

    with out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"[error] not a directory: {results_dir}")

    heterodimer = results_dir.name.removesuffix("_results")
    out_tsv = args.out_tsv
    if out_tsv is None:
        out_tsv = results_dir.parent / f"{results_dir.parent.name}_FoldXEnergies.tsv"

    ranked = find_ranked_raw_pdbs(results_dir)
    rows = [collect_row(results_dir, rank, ranked[rank]) for rank in sorted(ranked)]
    write_tsv(rows, out_tsv)

    print(f"[write] {out_tsv}")
    print(f"[summary] ranks={len(rows)} first={min(ranked)} last={max(ranked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
