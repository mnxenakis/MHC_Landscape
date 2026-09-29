#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HLA_DQ_ROOT", ".")).resolve()
OUT_DIR = ROOT / "reports" / "RMSD_figures"
DEFAULT_TSV = OUT_DIR / "clip_groove_fluctuation_direction.tsv"
DEFAULT_PDF = OUT_DIR / "clip_groove_fluctuation_direction.pdf"
DEFAULT_ORIENTATION_PDF = OUT_DIR / "clip_groove_orientation_reversal.pdf"
DEFAULT_CONTACTS_TSV = OUT_DIR / "clip_core_rank001_direct_contacts.tsv"

CHAIN_ALPHA = "A"
CHAIN_BETA = "B"
CHAIN_CLIP = "C"

TMD_ALPHA = "TVVCALGLSVGLVGIVVGTVFII"
TMD_BETA = "MLSGIGGFVLGLIFLGLGLII"
MAX_TMD_MISMATCHES = 2

RANK_RE = re.compile(r"rank_(\d{3,4})")
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


@dataclass
class Residue:
    chain: str
    key: str
    resname: str
    atoms: dict[str, np.ndarray]
    heavy: list[np.ndarray]


@dataclass
class Structure:
    chains: dict[str, list[Residue]]
    by_key: dict[tuple[str, str], Residue]


@dataclass
class ReferenceFrame:
    heterodimer: str
    ref_pdb: Path
    fit_keys: list[tuple[str, str]]
    clip_keys: list[tuple[str, str]]
    clip_core_keys: list[tuple[str, str]]
    ref_fit: np.ndarray
    ref_clip: dict[tuple[str, str], np.ndarray]
    ref_clip_core: dict[tuple[str, str], np.ndarray]
    ref_clip_core_centroid: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_pocket_count: int
    beta_pocket_count: int
    rank001_contact_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify the direction of rank-dependent CLIP fluctuations after "
            "rank-001 alpha/beta scaffold alignment."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--orientation-pdf", type=Path, default=DEFAULT_ORIENTATION_PDF)
    parser.add_argument("--rank001-contacts-tsv", type=Path, default=DEFAULT_CONTACTS_TSV)
    parser.add_argument("--heterodimer", nargs="*", default=None, help="Optional heterodimer ids such as 3.2_2.2.")
    parser.add_argument("--exclude-alpha", nargs="*", default=["5.2"], help="Alpha-chain parent dirs to skip.")
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--rank-stride", type=int, default=1)
    parser.add_argument("--max-heterodimers", type=int, default=None)
    parser.add_argument("--clip-core-start", type=int, default=5, help="1-based CLIP residue position for core start.")
    parser.add_argument("--clip-core-end", type=int, default=13, help="1-based CLIP residue position for core end.")
    parser.add_argument("--pocket-cutoff-angstrom", type=float, default=5.0)
    parser.add_argument("--reuse-tsv", action="store_true", help="Reuse --out-tsv and only regenerate the figure.")
    return parser.parse_args()


def rank_from_name(path: Path) -> int | None:
    match = RANK_RE.search(path.name)
    return int(match.group(1)) if match else None


def iter_pair_dirs(root: Path, include: set[str] | None, exclude_alpha: set[str]):
    for pair_dir in sorted(root.glob("*.*/*_*")):
        if pair_dir.parent.name in exclude_alpha:
            continue
        if include is not None and pair_dir.name not in include:
            continue
        results_dir = pair_dir / f"{pair_dir.name}_results"
        if results_dir.is_dir():
            yield pair_dir.name, results_dir


def rank_pdbs(results_dir: Path) -> dict[int, Path]:
    pdbs: dict[int, Path] = {}
    for path in sorted(results_dir.glob("*_Repair_chained.pdb")):
        rank = rank_from_name(path)
        if rank is not None:
            pdbs[rank] = path
    return pdbs


def atom_is_hydrogen(atom_name: str, element: str) -> bool:
    element = element.strip().upper()
    if element:
        return element in {"H", "D"}
    return atom_name.strip().upper().startswith(("H", "D"))


def parse_pdb(path: Path) -> Structure:
    residues: dict[tuple[str, str], Residue] = {}
    order: list[tuple[str, str]] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            if not chain:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = f"{resseq}{icode}"
            element = line[76:78].strip() if len(line) >= 78 else ""
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            rkey = (chain, key)
            if rkey not in residues:
                residues[rkey] = Residue(chain=chain, key=key, resname=resname, atoms={}, heavy=[])
                order.append(rkey)
            residues[rkey].atoms.setdefault(atom_name, coord)
            if not atom_is_hydrogen(atom_name, element):
                residues[rkey].heavy.append(coord)

    chains: dict[str, list[Residue]] = {}
    for chain, key in order:
        chains.setdefault(chain, []).append(residues[(chain, key)])
    return Structure(chains=chains, by_key=residues)


def sequence_and_keys(structure: Structure, chain: str) -> tuple[str, list[tuple[str, str]]]:
    seq: list[str] = []
    keys: list[tuple[str, str]] = []
    for residue in structure.chains.get(chain, []):
        if "CA" not in residue.atoms:
            continue
        seq.append(THREE_TO_ONE.get(residue.resname, "X"))
        keys.append((chain, residue.key))
    return "".join(seq), keys


def find_motif(seq: str, motif: str, max_mismatches: int) -> tuple[int | None, int | None]:
    best_pos = None
    best_mm = len(motif) + 1
    for pos in range(0, len(seq) - len(motif) + 1):
        window = seq[pos : pos + len(motif)]
        mismatches = sum(a != b for a, b in zip(window, motif))
        if mismatches < best_mm:
            best_pos = pos
            best_mm = mismatches
    if best_pos is not None and best_mm <= max_mismatches:
        return best_pos, best_mm
    return None, None


def non_ct_fit_keys(structure: Structure) -> list[tuple[str, str]]:
    fit_keys: list[tuple[str, str]] = []
    for chain, motif in ((CHAIN_ALPHA, TMD_ALPHA), (CHAIN_BETA, TMD_BETA)):
        seq, keys = sequence_and_keys(structure, chain)
        start, mismatches = find_motif(seq, motif, MAX_TMD_MISMATCHES)
        if start is None:
            raise ValueError(f"Could not identify TMD motif in chain {chain}.")
        end = start + len(motif) - 1
        fit_keys.extend(keys[: end + 1])
    return fit_keys


def ca_coord_map(structure: Structure, keys: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for key in keys:
        residue = structure.by_key.get(key)
        if residue is not None and "CA" in residue.atoms:
            out[key] = residue.atoms["CA"]
    return out


def coords_for_keys(coord_map: dict[tuple[str, str], np.ndarray], keys: list[tuple[str, str]]) -> np.ndarray:
    return np.array([coord_map[key] for key in keys], dtype=float)


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return v / norm


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mob_centroid = mobile.mean(axis=0)
    ref_centroid = reference.mean(axis=0)
    mob_centered = mobile - mob_centroid
    ref_centered = reference - ref_centroid
    covariance = mob_centered.T @ ref_centered
    u, _s, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = mob_centered @ rotation + ref_centroid
    diff = aligned - reference
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    translation = ref_centroid - mob_centroid @ rotation
    return rotation, translation, rmsd


def transform_coords(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation + translation


def contact_residues_to_clip_core(
    structure: Structure,
    chain: str,
    clip_core_keys: list[tuple[str, str]],
    cutoff: float,
) -> list[Residue]:
    return [
        residue
        for residue, _dist in contact_residues_to_clip_core_with_distances(
            structure, chain, clip_core_keys, cutoff
        )
    ]


def contact_residues_to_clip_core_with_distances(
    structure: Structure,
    chain: str,
    clip_core_keys: list[tuple[str, str]],
    cutoff: float,
) -> list[tuple[Residue, float]]:
    clip_atoms = [
        atom
        for key in clip_core_keys
        if key in structure.by_key
        for atom in structure.by_key[key].heavy
    ]
    if not clip_atoms:
        return []
    clip_array = np.array(clip_atoms, dtype=float)
    cutoff2 = cutoff * cutoff
    contacts: list[tuple[Residue, float]] = []
    for residue in structure.chains.get(chain, []):
        if not residue.heavy:
            continue
        res_atoms = np.array(residue.heavy, dtype=float)
        delta = res_atoms[:, None, :] - clip_array[None, :, :]
        min_dist2 = float(np.min(np.sum(delta * delta, axis=2)))
        if min_dist2 <= cutoff2:
            contacts.append((residue, math.sqrt(min_dist2)))
    return contacts


def contact_rows(
    heterodimer: str,
    ref_pdb: Path,
    chain_label: str,
    contacts: list[tuple[Residue, float]],
    cutoff: float,
    clip_core_start: int,
    clip_core_end: int,
    clip_core_sequence: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for residue, min_distance in contacts:
        rows.append(
            {
                "heterodimer": heterodimer,
                "rank": 1,
                "reference_pdb": ref_pdb.name,
                "chain": residue.chain,
                "chain_label": chain_label,
                "residue_id": residue.key,
                "residue_name": residue.resname,
                "residue_one_letter": THREE_TO_ONE.get(residue.resname, "X"),
                "min_distance_angstrom": min_distance,
                "cutoff_angstrom": cutoff,
                "clip_core_start": clip_core_start,
                "clip_core_end": clip_core_end,
                "clip_core_sequence": clip_core_sequence,
            }
        )
    return rows


def residue_centroid(residues: list[Residue]) -> np.ndarray | None:
    coords = [atom for residue in residues for atom in residue.heavy]
    if not coords:
        return None
    return np.mean(np.array(coords, dtype=float), axis=0)


def chain_ca_centroid(structure: Structure, chain: str) -> np.ndarray:
    coords = [residue.atoms["CA"] for residue in structure.chains.get(chain, []) if "CA" in residue.atoms]
    if not coords:
        raise ValueError(f"No CA atoms found for chain {chain}.")
    return np.mean(np.array(coords, dtype=float), axis=0)


def build_reference_frame(
    heterodimer: str,
    ref_pdb: Path,
    clip_core_start: int,
    clip_core_end: int,
    pocket_cutoff: float,
) -> ReferenceFrame:
    structure = parse_pdb(ref_pdb)
    fit_keys = non_ct_fit_keys(structure)
    clip_seq, clip_keys = sequence_and_keys(structure, CHAIN_CLIP)
    if not clip_keys:
        raise ValueError("No CLIP chain C CA atoms found.")
    if clip_core_start < 1 or clip_core_end > len(clip_keys) or clip_core_start > clip_core_end:
        raise ValueError(
            f"Invalid CLIP core range {clip_core_start}-{clip_core_end} for {len(clip_keys)} chain-C residues."
        )
    clip_core_keys = clip_keys[clip_core_start - 1 : clip_core_end]
    clip_core_sequence = clip_seq[clip_core_start - 1 : clip_core_end]

    ref_ca = ca_coord_map(structure, fit_keys + clip_keys)
    ref_fit = coords_for_keys(ref_ca, fit_keys)
    ref_clip = {key: ref_ca[key] for key in clip_keys}
    ref_clip_core = {key: ref_ca[key] for key in clip_core_keys}
    core_coords = coords_for_keys(ref_clip_core, clip_core_keys)
    core_centroid = core_coords.mean(axis=0)

    x_axis = unit_vector(core_coords[-1] - core_coords[0])

    alpha_contacts_with_dist = contact_residues_to_clip_core_with_distances(
        structure, CHAIN_ALPHA, clip_core_keys, pocket_cutoff
    )
    beta_contacts_with_dist = contact_residues_to_clip_core_with_distances(
        structure, CHAIN_BETA, clip_core_keys, pocket_cutoff
    )
    alpha_contacts = [residue for residue, _dist in alpha_contacts_with_dist]
    beta_contacts = [residue for residue, _dist in beta_contacts_with_dist]
    rank001_contact_rows = (
        contact_rows(
            heterodimer,
            ref_pdb,
            "alpha",
            alpha_contacts_with_dist,
            pocket_cutoff,
            clip_core_start,
            clip_core_end,
            clip_core_sequence,
        )
        + contact_rows(
            heterodimer,
            ref_pdb,
            "beta",
            beta_contacts_with_dist,
            pocket_cutoff,
            clip_core_start,
            clip_core_end,
            clip_core_sequence,
        )
    )
    alpha_centroid = residue_centroid(alpha_contacts)
    beta_centroid = residue_centroid(beta_contacts)
    if alpha_centroid is None:
        alpha_centroid = chain_ca_centroid(structure, CHAIN_ALPHA)
    if beta_centroid is None:
        beta_centroid = chain_ca_centroid(structure, CHAIN_BETA)

    y_raw = beta_centroid - alpha_centroid
    y_axis = unit_vector(y_raw - np.dot(y_raw, x_axis) * x_axis)
    z_axis = unit_vector(np.cross(x_axis, y_axis))

    return ReferenceFrame(
        heterodimer=heterodimer,
        ref_pdb=ref_pdb,
        fit_keys=fit_keys,
        clip_keys=clip_keys,
        clip_core_keys=clip_core_keys,
        ref_fit=ref_fit,
        ref_clip=ref_clip,
        ref_clip_core=ref_clip_core,
        ref_clip_core_centroid=core_centroid,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        alpha_pocket_count=len(alpha_contacts),
        beta_pocket_count=len(beta_contacts),
        rank001_contact_rows=rank001_contact_rows,
    )


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def process_rank(ref: ReferenceFrame, pdb_path: Path, rank: int) -> dict[str, object] | None:
    structure = parse_pdb(pdb_path)
    mob_ca = ca_coord_map(structure, ref.fit_keys + ref.clip_keys)
    common_fit_keys = [key for key in ref.fit_keys if key in mob_ca]
    if len(common_fit_keys) < 3:
        return None

    ref_fit = coords_for_keys({key: coord for key, coord in zip(ref.fit_keys, ref.ref_fit)}, common_fit_keys)
    mob_fit = coords_for_keys(mob_ca, common_fit_keys)
    rotation, translation, fit_rmsd_ang = kabsch_align(mob_fit, ref_fit)

    common_clip_keys = [key for key in ref.clip_keys if key in mob_ca and key in ref.ref_clip]
    common_core_keys = [key for key in ref.clip_core_keys if key in mob_ca and key in ref.ref_clip_core]
    if not common_core_keys:
        return None

    mob_clip = transform_coords(coords_for_keys(mob_ca, common_clip_keys), rotation, translation)
    ref_clip = coords_for_keys(ref.ref_clip, common_clip_keys)
    mob_core = transform_coords(coords_for_keys(mob_ca, common_core_keys), rotation, translation)
    ref_core = coords_for_keys(ref.ref_clip_core, common_core_keys)

    core_centroid = mob_core.mean(axis=0)
    ref_core_centroid = ref_core.mean(axis=0)
    displacement = core_centroid - ref_core_centroid
    dx = float(np.dot(displacement, ref.x_axis))
    dy = float(np.dot(displacement, ref.y_axis))
    dz = float(np.dot(displacement, ref.z_axis))
    centroid_shift = float(np.linalg.norm(displacement))
    denom = dx * dx + dy * dy + dz * dz
    if denom > 1.0e-20:
        fx, fy, fz = dx * dx / denom, dy * dy / denom, dz * dz / denom
    else:
        fx, fy, fz = 0.0, 0.0, 0.0

    orientation_dot = np.nan
    orientation_angle_deg = np.nan
    endpoint_same_nm = np.nan
    endpoint_flip_nm = np.nan
    endpoint_flip_minus_same_nm = np.nan
    if len(common_core_keys) >= 2:
        ref_axis = unit_vector(ref_core[-1] - ref_core[0])
        mob_axis = unit_vector(mob_core[-1] - mob_core[0])
        orientation_dot = float(np.clip(np.dot(mob_axis, ref_axis), -1.0, 1.0))
        orientation_angle_deg = float(np.degrees(np.arccos(orientation_dot)))
        endpoint_same_nm = (
            0.5
            * (
                np.linalg.norm(mob_core[0] - ref_core[0])
                + np.linalg.norm(mob_core[-1] - ref_core[-1])
            )
            * 0.1
        )
        endpoint_flip_nm = (
            0.5
            * (
                np.linalg.norm(mob_core[0] - ref_core[-1])
                + np.linalg.norm(mob_core[-1] - ref_core[0])
            )
            * 0.1
        )
        endpoint_flip_minus_same_nm = endpoint_flip_nm - endpoint_same_nm

    return {
        "heterodimer": ref.heterodimer,
        "rank": rank,
        "pdb_file": pdb_path.name,
        "fit_rmsd_angstrom": fit_rmsd_ang,
        "clip_rmsd_nm": rmsd(mob_clip, ref_clip) * 0.1,
        "clip_core_rmsd_nm": rmsd(mob_core, ref_core) * 0.1,
        "clip_core_centroid_shift_nm": centroid_shift * 0.1,
        "clip_shift_along_groove_nm": dx * 0.1,
        "clip_shift_alpha_to_beta_nm": dy * 0.1,
        "clip_shift_groove_normal_nm": dz * 0.1,
        "clip_abs_shift_along_groove_nm": abs(dx) * 0.1,
        "clip_abs_shift_alpha_to_beta_nm": abs(dy) * 0.1,
        "clip_abs_shift_groove_normal_nm": abs(dz) * 0.1,
        "frac_along_groove": fx,
        "frac_alpha_to_beta": fy,
        "frac_groove_normal": fz,
        "clip_core_orientation_dot": orientation_dot,
        "clip_core_orientation_angle_deg": orientation_angle_deg,
        "clip_core_endpoint_same_nm": endpoint_same_nm,
        "clip_core_endpoint_flip_nm": endpoint_flip_nm,
        "clip_core_endpoint_flip_minus_same_nm": endpoint_flip_minus_same_nm,
        "n_fit_ca": len(common_fit_keys),
        "n_clip_ca": len(common_clip_keys),
        "n_clip_core_ca": len(common_core_keys),
        "rank001_alpha_pocket_residues": ref.alpha_pocket_count,
        "rank001_beta_pocket_residues": ref.beta_pocket_count,
    }


def compute_table(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    include = set(args.heterodimer) if args.heterodimer else None
    exclude_alpha = set(args.exclude_alpha or [])
    pair_dirs = list(iter_pair_dirs(args.root, include, exclude_alpha))
    if args.max_heterodimers is not None:
        pair_dirs = pair_dirs[: args.max_heterodimers]
    if not pair_dirs:
        raise SystemExit("No heterodimer results directories found.")

    rows: list[dict[str, object]] = []
    rank001_contact_rows: list[dict[str, object]] = []
    for idx, (heterodimer, results_dir) in enumerate(pair_dirs, start=1):
        pdb_by_rank = rank_pdbs(results_dir)
        if 1 not in pdb_by_rank:
            print(f"[skip] {heterodimer}: missing rank_001 *_Repair_chained.pdb")
            continue
        print(f"[proc] {idx}/{len(pair_dirs)} {heterodimer}", flush=True)
        try:
            ref = build_reference_frame(
                heterodimer,
                pdb_by_rank[1],
                args.clip_core_start,
                args.clip_core_end,
                args.pocket_cutoff_angstrom,
            )
        except Exception as exc:
            print(f"[skip] {heterodimer}: {exc}")
            continue

        rank001_contact_rows.extend(ref.rank001_contact_rows)

        for rank, pdb_path in sorted(pdb_by_rank.items()):
            if args.max_rank is not None and rank > args.max_rank:
                continue
            if args.rank_stride > 1 and (rank - 1) % args.rank_stride != 0:
                continue
            row = process_rank(ref, pdb_path, rank)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit("No CLIP fluctuation rows were computed.")
    geometry_df = pd.DataFrame(rows).sort_values(["heterodimer", "rank"])
    contacts_df = pd.DataFrame(rank001_contact_rows)
    if not contacts_df.empty:
        contacts_df = contacts_df.sort_values(["heterodimer", "chain", "residue_id"])
    return geometry_df, contacts_df


def summarize_by_rank(df: pd.DataFrame, value: str) -> pd.DataFrame:
    clean = df[["rank", value]].copy()
    clean["rank"] = pd.to_numeric(clean["rank"], errors="coerce")
    clean[value] = pd.to_numeric(clean[value], errors="coerce")
    clean = clean.dropna()
    clean = clean.loc[clean["rank"] >= 2]
    out = (
        clean.groupby("rank", as_index=False)[value]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            std="std",
        )
        .sort_values("rank")
    )
    return out


def plot_iqr(ax, summary: pd.DataFrame, label: str, color: str, lw: float = 2.5) -> None:
    x = summary["rank"].to_numpy(dtype=float)
    y = summary["median"].to_numpy(dtype=float)
    q25 = summary["q25"].to_numpy(dtype=float)
    q75 = summary["q75"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, lw=lw, label=label)
    ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)


def plot_figure(df: pd.DataFrame, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 20,
            "axes.labelsize": 30,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_df = df.copy()
    plot_df["rank"] = pd.to_numeric(plot_df["rank"], errors="coerce")
    plot_df = plot_df.loc[plot_df["rank"] >= 2].dropna(subset=["rank"])
    if plot_df.empty:
        raise ValueError("No ranks >= 2 are available for plotting.")
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 10.2), sharex=True, constrained_layout=True)
    ax_b, ax_d = axes.ravel()

    components = [
        ("clip_shift_along_groove_nm", r"along groove ($s_x(r)$)", "#d55e00"),
        ("clip_shift_alpha_to_beta_nm", r"$\alpha\rightarrow\beta$ ($s_y(r)$)", "#0072b2"),
        ("clip_shift_groove_normal_nm", r"groove normal ($s_z(r)$)", "#009e73"),
    ]
    for col, label, color in components:
        plot_iqr(ax_b, summarize_by_rank(plot_df, col), label, color)
    ax_b.axhline(0.0, color="0.25", lw=1.0, ls="--", alpha=0.7)
    ax_b.set_ylim(-0.15, 0.05)
    ax_b.set_ylabel(r"$s_x(r),\,s_y(r),\,s_z(r)$ [nm]")
    ax_b.legend(loc="best", frameon=False)

    fractions = [
        ("frac_along_groove", r"along groove ($f_x(r)$)", "#d55e00"),
        ("frac_alpha_to_beta", r"$\alpha\rightarrow\beta$ ($f_y(r)$)", "#0072b2"),
        ("frac_groove_normal", r"groove normal ($f_z(r)$)", "#009e73"),
    ]
    for col, label, color in fractions:
        plot_iqr(ax_d, summarize_by_rank(plot_df, col), label, color)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.set_ylabel(r"$f_x(r),\,f_y(r),\,f_z(r)$")

    for ax in (ax_b, ax_d):
        ax.set_xlim(0, 1000)
        ax.set_xticks([2, *range(100, 1001, 100)])
        ax.grid(True, color="0.9", lw=0.8)
    ax_d.set_xlabel("rank $r$")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_orientation_figure(df: pd.DataFrame, out_pdf: Path) -> None:
    required = {
        "clip_core_orientation_dot",
        "clip_core_orientation_angle_deg",
        "clip_core_endpoint_flip_minus_same_nm",
    }
    missing = required.difference(df.columns)
    if missing:
        print(f"[skip] orientation plot requires recomputed TSV columns: {', '.join(sorted(missing))}")
        return

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 20,
            "axes.labelsize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_df = df.copy()
    plot_df["rank"] = pd.to_numeric(plot_df["rank"], errors="coerce")
    plot_df = plot_df.loc[plot_df["rank"] >= 2].dropna(subset=["rank"])
    if plot_df.empty:
        raise ValueError("No ranks >= 2 are available for orientation plotting.")

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.6), constrained_layout=True)
    ax_q, ax_d = axes.ravel()

    plot_iqr(
        ax_q,
        summarize_by_rank(plot_df, "clip_core_orientation_dot"),
        r"$q_{\mathrm{orient}}(r)$",
        "#6a3d9a",
        lw=2.8,
    )
    ax_q.axhline(1.0, color="0.25", lw=1.2, ls="--", alpha=0.5)
    ax_q.axhline(0.0, color="0.25", lw=1.2, ls="--", alpha=0.5)
    ax_q.set_ylim(-1.05, 1.05)
    ax_q.set_ylabel(r"$q_{\mathrm{orient}}(r)$")
    ax_q.legend(loc="lower left", frameon=False)

    plot_iqr(
        ax_d,
        summarize_by_rank(plot_df, "clip_core_endpoint_flip_minus_same_nm"),
        r"$D_{\mathrm{flip}}(r)-D_{\mathrm{same}}(r)$",
        "#009e73",
        lw=2.8,
    )
    ax_d.axhline(0.0, color="0.25", lw=1.2, ls="--", alpha=0.5)
    ax_d.set_ylabel(r"$D_{\mathrm{flip}}(r)-D_{\mathrm{same}}(r)$ [nm]")
    ax_d.legend(loc="best", frameon=False)

    for ax in (ax_q, ax_d):
        ax.set_xlabel("rank $r$")
        ax.set_xlim(0, 1000)
        ax.set_xticks([2, *range(100, 1001, 100)])
        ax.grid(True, color="0.9", lw=0.8)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    non_ref = df.loc[df["rank"] > 1].copy()
    if non_ref.empty:
        non_ref = df.copy()
    means = {
        "along_groove": float(non_ref["frac_along_groove"].median()),
        "alpha_to_beta": float(non_ref["frac_alpha_to_beta"].median()),
        "groove_normal": float(non_ref["frac_groove_normal"].median()),
    }
    print(
        "[direction] median squared-displacement fractions: "
        f"along_groove={means['along_groove']:.3f} "
        f"alpha_to_beta={means['alpha_to_beta']:.3f} "
        f"groove_normal={means['groove_normal']:.3f}"
    )

    vectors = non_ref[
        [
            "clip_shift_along_groove_nm",
            "clip_shift_alpha_to_beta_nm",
            "clip_shift_groove_normal_nm",
        ]
    ].to_numpy(dtype=float)
    vectors = vectors[np.all(np.isfinite(vectors), axis=1)]
    if len(vectors) >= 3:
        cov = np.cov(vectors, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        frac = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
        pc1 = eigvecs[:, 0]
        print(
            "[pca] PC1 in local coordinates "
            f"(along_groove, alpha_to_beta, groove_normal)=({pc1[0]:.3f}, {pc1[1]:.3f}, {pc1[2]:.3f}); "
            f"variance_fraction={frac[0]:.3f}"
        )

    if "clip_core_orientation_dot" in non_ref.columns:
        orient = pd.to_numeric(non_ref["clip_core_orientation_dot"], errors="coerce").dropna()
        if not orient.empty:
            print(
                "[orientation] median q_orient="
                f"{orient.median():.3f}; fraction q_orient<0={float((orient < 0).mean()):.4f}"
            )


def main() -> int:
    args = parse_args()
    if args.rank_stride < 1:
        raise SystemExit("--rank-stride must be >= 1")
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.orientation_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.rank001_contacts_tsv.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_tsv:
        if not args.out_tsv.exists():
            raise SystemExit(f"Cannot reuse missing TSV: {args.out_tsv}")
        df = pd.read_csv(args.out_tsv, sep="\t")
    else:
        df, contacts_df = compute_table(args)
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[write] {args.out_tsv}")
        contacts_df.to_csv(args.rank001_contacts_tsv, sep="\t", index=False)
        print(f"[write] {args.rank001_contacts_tsv}")

    plot_figure(df, args.out_pdf)
    plot_orientation_figure(df, args.orientation_pdf)
    print_summary(df)
    print(f"[write] {args.out_pdf}")
    if set(
        [
            "clip_core_orientation_dot",
            "clip_core_orientation_angle_deg",
            "clip_core_endpoint_flip_minus_same_nm",
        ]
    ).issubset(df.columns):
        print(f"[write] {args.orientation_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
