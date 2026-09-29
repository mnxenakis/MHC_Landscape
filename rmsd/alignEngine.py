#!/usr/bin/env python
"""PyMOL alignment and RMSD pipeline for ranked DQ models.

This script uses the PyMOL Python API (`pymol` or `pymol2`) to:
  1) load the ranked repaired models in the current `*_results` directory,
  2) locate the full TMD span in chains A and B from configurable TMD
     sequences,
  3) align the ensemble using non-CT CA atoms from chains A/B,
  4) compute post-fit whole-chain CA RMSD for alpha/beta and the peptide
     chain C, plus whole-complex CA RMSD, all relative to `rank_001`,
  5) compute per-residue whole-chain RMSD relative to `rank_001` and write it
     into occupancy (`q`) for visualization,
  6) export a processed `rank_001` PDB, RMSD/pLDDT PNGs, and a per-state RMSD
     table in `alignment_data/`.

Run it inside one `*_results` directory with:
  python alignEngine.py

Requirements:
  - PyMOL with Python bindings installed.
  - Ranked `*_Repair_chained.pdb` files present in the current directory
    (preferred), or `*_Repair.pdb` as a fallback.
"""

import glob
import re

import os
import shutil

# --- PyMOL bootstrap (supports either `pymol` or `pymol2`) -------------------
_PYMOL2_SESSION = None

try:
    import pymol  # type: ignore
    from pymol import cmd  # type: ignore

    # Launch PyMOL in quiet, no-GUI mode
    pymol.finish_launching(["pymol", "-cq"])
except ModuleNotFoundError:
    try:
        import pymol2  # type: ignore

        _PYMOL2_SESSION = pymol2.PyMOL()
        _PYMOL2_SESSION.start()
        cmd = _PYMOL2_SESSION.cmd
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyMOL Python bindings not found (cannot import 'pymol' or 'pymol2').\n"
            "Fix options:\n"
            "  1) Install PyMOL in your conda env (recommended):\n"
            "     conda install -c conda-forge pymol-open-source\n"
            "  2) Or run via a system PyMOL executable (if available):\n"
            "     pymol -cq -d \"run rmsd/alignEngine.py; dq_full_pipeline(); quit\"\n"
        ) from e
except Exception:
    # If PyMOL is present but failed to initialize, try pymol2 as a fallback.
    try:
        import pymol2  # type: ignore

        _PYMOL2_SESSION = pymol2.PyMOL()
        _PYMOL2_SESSION.start()
        cmd = _PYMOL2_SESSION.cmd
    except Exception as e:
        raise SystemExit(f"PyMOL is installed but failed to initialize: {e}")

# ================== USER SETTINGS =============================================

# File pattern for ranked input PDBs in the current directory
PDB_PATTERN   = "*_rank_*.pdb"
PDB_SUFFIX_PRIORITY = ("_Repair_chained.pdb", "_Repair.pdb")

# Chain IDs for the DQ alpha, beta, and peptide chains
CHAIN_A       = "A"
CHAIN_B       = "B"
CHAIN_C       = "C"

# Full UniProt TMD sequences for alpha and beta chains. The script identifies
# the TMD span with a small mismatch tolerance and excludes everything after
# the matched TMD end as CT.
TMD_A   = "TVVCALGLSVGLVGIVVGTVFII"  # chain A
TMD_B   = "MLSGIGGFVLGLIFLGLGLII"    # chain B

# Allow up to this many mismatches when finding motifs (polymorphisms)
MAX_MISMATCH  = 2

# Object names
REF_OBJECT    = "dq_ref"
ENSEMBLE_OBJ  = "dq_ensemble"
RANK001_OBJ   = "dq_rank001"

# Ray / PNG settings
RAY_SIZE      = (2000, 2000)
PNG_DPI       = 300
# Use a custom RGB ramp so the gradient is forced to blue -> white -> red.
PLDDT_PALETTE = "dq_blue dq_white dq_red"
RMSD_PALETTE  = "dq_blue dq_white dq_red"

# Directory for processed PDBs, images, and RMSD tables
OUTPUT_DIR    = "alignment_data"

# Performance / verbosity
# - Writing per-residue RMSD into occupancy q and printing residue-by-residue
#   values is relatively slow in PyMOL.
# - Keep detailed per-residue output off unless you are debugging a single case.
PRINT_PER_RES_RMSD = False
# Attempt to reduce PyMOL rebuild overhead during bulk loading and alignment.
FAST_PYMOL_MODE = True

# ============================================================================

_three_to_one = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

def _obj_exists(name: str) -> bool:
    """Safe object-exists check that works on open-source PyMOL."""
    try:
        # present in many builds
        return bool(cmd.object_exists(name))
    except AttributeError:
        # fallback: check list of objects
        return name in cmd.get_object_list()

def _get_seq_and_resi(obj, chain):
    """
    Return (seq, resi_list) for given chain in object,
    using CA atoms to define residue order.
    resi_list is a list of residue IDs (strings).
    """
    sel = f"{obj} and chain {chain} and name CA"
    model = cmd.get_model(sel)
    seq = []
    resi_list = []
    seen = set()
    for at in model.atom:
        key = (at.chain, at.resi)
        if key in seen:
            continue
        seen.add(key)
        aa = _three_to_one.get(at.resn.upper(), "X")
        seq.append(aa)
        resi_list.append(at.resi)  # residue ID (string)
    return "".join(seq), resi_list

def _get_ca_coords_by_resi(obj, chain, state=1):
    """
    Fetch CA coordinates for a given chain/state once to avoid per-residue PyMOL calls.
    Returns (coords_by_resi, resi_order).
    """
    sel = f"{obj} and chain {chain} and name CA"
    model = cmd.get_model(sel, state=state)
    coords = {}
    order = []
    for at in model.atom:
        resi = at.resi
        if resi in coords:
            continue
        coords[resi] = at.coord
        order.append(resi)
    return coords, order

def _get_ca_coords_by_chain_resi(obj, chains, state=1, allowed_resi_by_chain=None):
    """
    Fetch CA coordinates for multiple chains in one pass.

    Returns `(coords_by_key, key_order)` where keys are `(chain, resi)` tuples in
    the reference ordering encountered in the model.
    """
    chain_list = [str(chain).strip() for chain in chains if str(chain).strip()]
    if not chain_list:
        return {}, []

    chain_sel = " or ".join(f"chain {chain}" for chain in chain_list)
    sel = f"{obj} and ({chain_sel}) and name CA"
    model = cmd.get_model(sel, state=state)

    allowed_sets = None
    if allowed_resi_by_chain is not None:
        allowed_sets = {
            str(chain).strip(): set(resi_list)
            for chain, resi_list in allowed_resi_by_chain.items()
        }

    coords = {}
    order = []
    for at in model.atom:
        chain = at.chain
        resi = at.resi
        if allowed_sets is not None:
            allowed_resi = allowed_sets.get(chain)
            if allowed_resi is not None and resi not in allowed_resi:
                continue
        key = (chain, resi)
        if key in coords:
            continue
        coords[key] = at.coord
        order.append(key)
    return coords, order

def _filter_coords_by_resi(coords_by_resi, resi_order, allowed_resi=None):
    """Restrict residue-coordinate mappings to an allowed residue subset."""
    if allowed_resi is None:
        return coords_by_resi, resi_order
    allowed = set(allowed_resi)
    filt_order = [resi for resi in resi_order if resi in allowed]
    filt_coords = {resi: coords_by_resi[resi] for resi in filt_order if resi in coords_by_resi}
    return filt_coords, filt_order

def _selection_for_resi_list(obj_name, chain, resi_list, atom_name=None):
    """Build a PyMOL selection string for one chain and an explicit residue list."""
    if not resi_list:
        return None
    name_clause = f" and name {atom_name}" if atom_name else ""
    resi_spec = "+".join(str(resi) for resi in resi_list)
    return f"({obj_name} and chain {chain}{name_clause} and resi {resi_spec})"

def _define_render_colors():
    """Define exact RGB colors used in the blue -> white -> red ramps."""
    cmd.set_color("dq_blue", [0.0, 0.0, 1.0])
    cmd.set_color("dq_white", [1.0, 1.0, 1.0])
    cmd.set_color("dq_red", [1.0, 0.0, 0.0])

def _find_motif(seq, motif, max_mismatch=MAX_MISMATCH):
    """
    Fuzzy search for motif in seq.
    Returns (start_index, mismatches) or (None, None).
    start_index is 0-based index in seq.
    """
    Ls = len(seq)
    Lm = len(motif)
    best_pos = None
    best_mm  = Lm + 1
    for i in range(Ls - Lm + 1):
        window = seq[i:i+Lm]
        mm = sum(1 for a, b in zip(window, motif) if a != b)
        # Prefer the first occurrence in case of ties (standard motif search behavior)
        if mm < best_mm:
            best_mm = mm
            best_pos = i
    if best_pos is not None and best_mm <= max_mismatch:
        return best_pos, best_mm
    return None, None

def _find_tmd_span(seq, tmd_sequence, max_mismatch=MAX_MISMATCH):
    """
    Find the full TMD sequence in `seq` with a small mismatch tolerance.

    Returns `(start_index, end_index, mismatches)` using 0-based inclusive
    indexing, or `(None, None, None)` if no acceptable hit is found.
    """
    start_idx, mismatches = _find_motif(seq, tmd_sequence, max_mismatch=max_mismatch)
    if start_idx is None:
        return None, None, None
    end_idx = start_idx + len(tmd_sequence) - 1
    return start_idx, end_idx, mismatches

def _find_ranked_files(pattern):
    """
    Find ranked repaired PDBs in the working directory.

    Returns:
    - a list of `(rank, filename)` tuples sorted by rank
    - the reference filename for `rank_001`, or the lowest-rank file if rank 1
      is absent

    If both repaired file variants are present, prefer `*_Repair_chained.pdb`
    because the chain-level RMSD analysis should operate on the chained models.
    """
    pdb_files = glob.glob(pattern)
    ranked = []
    chosen_suffix = None
    for suffix in PDB_SUFFIX_PRIORITY:
        ranked = []
        for fname in pdb_files:
            if not fname.endswith(suffix):
                continue
            m = re.search(r"_rank_([0-9]+)_", fname)
            if m:
                rank = int(m.group(1))
                ranked.append((rank, fname))
        if ranked:
            chosen_suffix = suffix
            break
    if not ranked:
        return [], None

    ranked.sort(key=lambda x: x[0])
    ref_file = None
    for r, f in ranked:
        if r == 1:
            ref_file = f
            break
    if ref_file is None:
        ref_file = ranked[0][1]
    print(f"[dq] Using ranked input suffix: {chosen_suffix}")
    return ranked, ref_file

def _per_res_rmsd_to_q(obj_name, chain, ref_state=1, allowed_resi=None):
    """
    Compute per-residue CA RMSD relative to the reference state and write it
    into occupancy q for coloring in the exported `rank_001` structure.

    RMSD values are computed after the ensemble has already been aligned with
    `cmd.intra_fit`, so they reflect deviations within that common frame.
    """
    import numpy as np

    obj_name = obj_name.strip()
    chain = chain.strip()

    n_states = cmd.count_states(obj_name)
    if n_states < 2:
        print(f"[dq] Object '{obj_name}' has only {n_states} state(s). Need an ensemble.")
        return None

    ref_coords, resi_list = _get_ca_coords_by_resi(obj_name, chain, state=ref_state)
    ref_coords, resi_list = _filter_coords_by_resi(ref_coords, resi_list, allowed_resi)
    if not resi_list:
        print(f"[dq] No CA atoms found for {obj_name} chain {chain}.")
        return None

    n_res = len(resi_list)
    resi_to_idx = {resi: idx for idx, resi in enumerate(resi_list)}
    allowed = set(allowed_resi) if allowed_resi is not None else None

    ref_array = np.full((n_res, 3), np.nan, dtype=float)
    for resi, coord in ref_coords.items():
        idx = resi_to_idx.get(resi)
        if idx is not None:
            ref_array[idx] = coord

    all_coords = np.full((n_states, n_res, 3), np.nan, dtype=float)
    for st in range(1, n_states + 1):
        coords, _ = _get_ca_coords_by_resi(obj_name, chain, state=st)
        if allowed is not None:
            coords = {resi: coord for resi, coord in coords.items() if resi in allowed}
        if not coords:
            continue
        vec = all_coords[st - 1]
        for resi, coord in coords.items():
            idx = resi_to_idx.get(resi)
            if idx is not None:
                vec[idx] = coord

    diff = all_coords - ref_array
    sq = diff * diff
    msd = np.nanmean(sq.sum(axis=2), axis=0)
    rmsd = np.sqrt(np.nan_to_num(msd, nan=0.0))

    rmsd_by_resi = {resi: float(value) for resi, value in zip(resi_list, rmsd)}
    try:
        import pymol  # type: ignore

        pymol.stored.dq_rmsd_by_resi = rmsd_by_resi
        # Single bulk alter is much faster than one call per residue.
        cmd.alter(
            f"{obj_name} and chain {chain}",
            "q = stored.dq_rmsd_by_resi.get(resi, 0.0)",
        )
    except Exception:
        # Fallback: slower but robust.
        for resi, value in rmsd_by_resi.items():
            cmd.alter(f"{obj_name} and chain {chain} and resi {resi}", f"q={float(value)}")

    stats = {
        "chain": chain,
        "n_residues": n_res,
        "mean_rmsd": float(np.nanmean(rmsd)),
        "median_rmsd": float(np.nanmedian(rmsd)),
        "min_rmsd": float(np.nanmin(rmsd)),
        "max_rmsd": float(np.nanmax(rmsd)),
    }
    print(
        f"[dq] RMSD→q written for {obj_name} chain {chain}: "
        f"{n_res} residues (mean={stats['mean_rmsd']:.3f} Å, "
        f"min={stats['min_rmsd']:.3f} Å, max={stats['max_rmsd']:.3f} Å)"
    )
    if PRINT_PER_RES_RMSD:
        for resi, value in zip(resi_list, rmsd):
            print(f"  resi {resi:>4}: {float(value):6.3f} Å")
    return stats

def _print_chain_flexibility_comparison(alpha_stats, beta_stats, peptide_stats=None):
    """Print whole-chain mean per-residue RMSD summaries."""
    if not alpha_stats or not beta_stats:
        return

    alpha_mean = alpha_stats["mean_rmsd"]
    beta_mean = beta_stats["mean_rmsd"]
    diff = beta_mean - alpha_mean

    message = (
        f"[dq] Mean whole-chain per-residue RMSD vs state 1: "
        f"alpha={alpha_mean:.3f} Å ({alpha_stats['n_residues']} residues), "
        f"beta={beta_mean:.3f} Å ({beta_stats['n_residues']} residues)"
    )
    if peptide_stats:
        message += (
            f", peptide(chain {peptide_stats['chain']})="
            f"{peptide_stats['mean_rmsd']:.3f} Å ({peptide_stats['n_residues']} residues)"
        )
    print(message)
    if abs(diff) < 1.0e-9:
        print("[dq] Flexibility call: alpha and beta are tied by mean per-residue RMSD.")
    elif diff > 0.0:
        print(f"[dq] Flexibility call: beta is more flexible than alpha by {diff:.3f} Å in mean per-residue RMSD.")
    else:
        print(f"[dq] Flexibility call: alpha is more flexible than beta by {-diff:.3f} Å in mean per-residue RMSD.")

def _chain_rmsd_by_state(obj_name, chain, ref_state=1, allowed_resi=None):
    """
    Return CA RMSD for one chain in each state relative to the reference state.

    These RMSDs are measured after the full ensemble has been aligned on the
    chosen fit selection, so they capture chain motion/deformation within the
    aligned dimer frame rather than an independent per-chain refit.
    """
    import numpy as np

    obj_name = obj_name.strip()
    chain = chain.strip()

    n_states = cmd.count_states(obj_name)
    if n_states < 1:
        return []

    ref_coords, ref_resi_order = _get_ca_coords_by_resi(obj_name, chain, state=ref_state)
    ref_coords, ref_resi_order = _filter_coords_by_resi(ref_coords, ref_resi_order, allowed_resi)
    if not ref_resi_order:
        print(f"[dq] No CA atoms found for {obj_name} chain {chain}.")
        return []

    allowed = set(allowed_resi) if allowed_resi is not None else None
    out = []
    for st in range(1, n_states + 1):
        mob_coords, _ = _get_ca_coords_by_resi(obj_name, chain, state=st)
        if allowed is not None:
            mob_coords = {resi: coord for resi, coord in mob_coords.items() if resi in allowed}
        common_resi = [resi for resi in ref_resi_order if resi in mob_coords]
        if not common_resi:
            out.append((st, float("nan"), 0))
            continue

        ref_array = np.array([ref_coords[resi] for resi in common_resi], dtype=float)
        mob_array = np.array([mob_coords[resi] for resi in common_resi], dtype=float)
        diff = mob_array - ref_array
        rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        out.append((st, rmsd, len(common_resi)))

    return out

def _multi_chain_rmsd_by_state(obj_name, chains, ref_state=1, allowed_resi_by_chain=None, label="selection"):
    """
    Return CA RMSD for multiple chains in each state relative to the reference state.

    These RMSDs are measured after the full ensemble has already been aligned on
    the chosen fit selection.
    """
    import numpy as np

    obj_name = obj_name.strip()
    chain_list = [str(chain).strip() for chain in chains if str(chain).strip()]
    if not chain_list:
        return []

    n_states = cmd.count_states(obj_name)
    if n_states < 1:
        return []

    ref_coords, ref_key_order = _get_ca_coords_by_chain_resi(
        obj_name,
        chain_list,
        state=ref_state,
        allowed_resi_by_chain=allowed_resi_by_chain,
    )
    if not ref_key_order:
        joined = ",".join(chain_list)
        print(f"[dq] No CA atoms found for {obj_name} chains {joined} ({label}).")
        return []

    out = []
    for st in range(1, n_states + 1):
        mob_coords, _ = _get_ca_coords_by_chain_resi(
            obj_name,
            chain_list,
            state=st,
            allowed_resi_by_chain=allowed_resi_by_chain,
        )
        common_keys = [key for key in ref_key_order if key in mob_coords]
        if not common_keys:
            out.append((st, float("nan"), 0))
            continue

        ref_array = np.array([ref_coords[key] for key in common_keys], dtype=float)
        mob_array = np.array([mob_coords[key] for key in common_keys], dtype=float)
        diff = mob_array - ref_array
        rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        out.append((st, rmsd, len(common_keys)))

    return out

def _write_chain_rmsd_table(
    out_path,
    ranked_files,
    intra_fit_rmsds,
    complex_rmsds,
    alpha_rmsds,
    beta_rmsds,
    peptide_rmsds,
):
    """
    Write one row per ranked model for the aligned ensemble.

    The table includes:
    - `fit_selection_intra_fit_rmsd`: PyMOL RMSD returned by the non-CT fit selection
    - `whole_complex_ca_rmsd`: post-fit whole-complex CA RMSD relative to state 1
    - `alpha_whole_ca_rmsd` / `beta_whole_ca_rmsd` / `peptide_chain_c_ca_rmsd`:
      post-fit whole-chain CA RMSD relative to state 1
    """
    complex_by_state = {state: (rmsd, n_ca) for state, rmsd, n_ca in complex_rmsds}
    alpha_by_state = {state: (rmsd, n_ca) for state, rmsd, n_ca in alpha_rmsds}
    beta_by_state = {state: (rmsd, n_ca) for state, rmsd, n_ca in beta_rmsds}
    peptide_by_state = {state: (rmsd, n_ca) for state, rmsd, n_ca in peptide_rmsds}

    with open(out_path, "w") as handle:
        handle.write(
            "state\trank\tfile\tfit_selection_intra_fit_rmsd\t"
            "whole_complex_ca_rmsd\twhole_complex_n_ca\t"
            "alpha_whole_ca_rmsd\talpha_whole_n_ca\t"
            "beta_whole_ca_rmsd\tbeta_whole_n_ca\t"
            "peptide_chain_c_ca_rmsd\tpeptide_chain_c_n_ca\n"
        )
        for state, (rank, fname) in enumerate(ranked_files, start=1):
            complex_rmsd, complex_n_ca = complex_by_state.get(state, (float("nan"), 0))
            alpha_rmsd, alpha_n_ca = alpha_by_state.get(state, (float("nan"), 0))
            beta_rmsd, beta_n_ca = beta_by_state.get(state, (float("nan"), 0))
            peptide_rmsd, peptide_n_ca = peptide_by_state.get(state, (float("nan"), 0))
            core_rmsd = float("nan")
            if state - 1 < len(intra_fit_rmsds):
                core_rmsd = intra_fit_rmsds[state - 1]
            handle.write(
                f"{state}\t{rank}\t{os.path.basename(fname)}\t{core_rmsd:.6f}\t"
                f"{complex_rmsd:.6f}\t{complex_n_ca}\t"
                f"{alpha_rmsd:.6f}\t{alpha_n_ca}\t"
                f"{beta_rmsd:.6f}\t{beta_n_ca}\t"
                f"{peptide_rmsd:.6f}\t{peptide_n_ca}\n"
            )

    print(f"[dq] Saved chain RMSD table to: {out_path}")

    for label, values in (
        ("whole complex", complex_rmsds),
        ("alpha", alpha_rmsds),
        ("beta", beta_rmsds),
        ("peptide chain C", peptide_rmsds),
    ):
        finite = [rmsd for _state, rmsd, _n_ca in values if rmsd == rmsd]
        if finite:
            print(
                f"[dq] {label} CA RMSD range (vs state 1, after non-CT fit): "
                f"{min(finite):.3f} to {max(finite):.3f} Å"
            )

def dq_full_pipeline(core_sel=None):
    """
    Full pipeline:
      1) Find *_rank_###_*.pdb and reference *_rank_001_*.
      2) Detect the full TMD span for chain A/B via fuzzy sequence matching.
      3) Load the ensemble and align states using non-CT CA atoms from chains A/B.
      4) Write post-fit whole-complex and whole-chain RMSD values versus rank_001.
      5) Encode whole-chain per-residue RMSD in occupancy q for chains A/B/C.
      6) Extract state 1 (rank_001) and save the processed PDB plus pLDDT/RMSD
         PNGs.
    """

    # 1) Find ranked files and reference
    ranked_files, ref_file = _find_ranked_files(PDB_PATTERN)
    if not ranked_files:
        print(f"[dq] No PDBs matching '{PDB_PATTERN}' found in {os.getcwd()}.")
        return

    print("[dq] Ranked files:")
    for r, f in ranked_files:
        print(f"   rank {r:4d} -> {f}")
    print(f"[dq] Using reference file: {ref_file}")

    rank001_base = os.path.splitext(os.path.basename(ref_file))[0]

    # Rebuild the output directory so legacy files do not accumulate across runs.
    if os.path.isdir(OUTPUT_DIR):
        print(f"[dq] Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    elif os.path.exists(OUTPUT_DIR):
        print(f"[dq] Removing existing non-directory output path: {OUTPUT_DIR}")
        os.remove(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2) Load reference and detect TMD spans. CT is defined implicitly as
    # everything after the matched TMD end.
    if _obj_exists(REF_OBJECT):
        cmd.delete(REF_OBJECT)
    cmd.load(ref_file, REF_OBJECT)

    # Chain A
    seqA, resiA = _get_seq_and_resi(REF_OBJECT, CHAIN_A)
    print(f"[dq] Chain {CHAIN_A} length: {len(seqA)} aa")
    startA, endA, mmA = _find_tmd_span(seqA, TMD_A, MAX_MISMATCH)
    if startA is None or endA is None:
        print(f"[dq] ERROR: TMD sequence for chain {CHAIN_A} not found (≤ {MAX_MISMATCH} mismatches).")
        return
    tmd_start_resi_A = resiA[startA]
    tmd_end_resi_A = resiA[endA]
    keep_resi_A = resiA[:endA + 1]
    print(f"[dq] Chain {CHAIN_A} TMD span approx at seq index {startA+1}-{endA+1} (1-based), "
          f"start resi {tmd_start_resi_A}, end resi {tmd_end_resi_A}, mismatches = {mmA}")
    print(f"[dq]    matched TMD = {seqA[startA:endA+1]}")

    # Chain B
    seqB, resiB = _get_seq_and_resi(REF_OBJECT, CHAIN_B)
    print(f"[dq] Chain {CHAIN_B} length: {len(seqB)} aa")
    startB, endB, mmB = _find_tmd_span(seqB, TMD_B, MAX_MISMATCH)
    if startB is None or endB is None:
        print(f"[dq] ERROR: TMD sequence for chain {CHAIN_B} not found (≤ {MAX_MISMATCH} mismatches).")
        return
    tmd_start_resi_B = resiB[startB]
    tmd_end_resi_B = resiB[endB]
    keep_resi_B = resiB[:endB + 1]
    print(f"[dq] Chain {CHAIN_B} TMD span approx at seq index {startB+1}-{endB+1} (1-based), "
          f"start resi {tmd_start_resi_B}, end resi {tmd_end_resi_B}, mismatches = {mmB}")
    print(f"[dq]    matched TMD = {seqB[startB:endB+1]}")

    print(f"[dq] -> TMD starts at: chain {CHAIN_A} resi {tmd_start_resi_A}, "
          f"chain {CHAIN_B} resi {tmd_start_resi_B}")
    print(f"[dq] -> TMD ends at: chain {CHAIN_A} resi {tmd_end_resi_A}, "
          f"chain {CHAIN_B} resi {tmd_end_resi_B}")
    print(f"[dq] -> Non-CT residues kept: chain {CHAIN_A}={len(keep_resi_A)}, "
          f"chain {CHAIN_B}={len(keep_resi_B)}")

    seqC, resiC = _get_seq_and_resi(REF_OBJECT, CHAIN_C)
    has_chain_c = bool(resiC)
    if has_chain_c:
        print(f"[dq] Chain {CHAIN_C} length: {len(seqC)} aa (peptide chain; whole chain included in RMSD reporting)")
    else:
        print(f"[dq] Chain {CHAIN_C} not found in reference; peptide RMSD reporting will be skipped.")

    # 3) Load ensemble and align all states in a common non-CT frame
    if FAST_PYMOL_MODE:
        try:
            cmd.set("defer_builds_mode", 3)
        except Exception:
            pass

    if _obj_exists(ENSEMBLE_OBJ):
        cmd.delete(ENSEMBLE_OBJ)

    for i, (rank, fname) in enumerate(ranked_files, start=1):
        print(f"[dq] Loading state {i} (rank {rank}) from {fname}")
        cmd.load(fname, ENSEMBLE_OBJ)

    # Fit using all non-CT CA atoms from chains A and B. CT residues and chain C
    # stay loaded in the object but are excluded from fitting.
    if core_sel is None:
        selA = _selection_for_resi_list(ENSEMBLE_OBJ, CHAIN_A, keep_resi_A, atom_name="CA")
        selB = _selection_for_resi_list(ENSEMBLE_OBJ, CHAIN_B, keep_resi_B, atom_name="CA")
        core_sel = (
            f"({selA} or {selB})"
        )
    
    print(f"[dq] Running intra_fit on non-CT CA: {core_sel}")
    rms_list = cmd.intra_fit(core_sel)

    print("[dq] intra_fit RMSDs (vs state 1):")
    for i, rms in enumerate(rms_list, start=1):
        print(f"   state {i:4d}: RMSD = {rms:.3f} Å")

    # 4) Export post-fit whole-complex and whole-chain RMSD, then encode
    # whole-chain per-residue RMSD in q for visualization.
    complex_chains = [CHAIN_A, CHAIN_B]
    if has_chain_c:
        complex_chains.append(CHAIN_C)
    complex_rmsds = _multi_chain_rmsd_by_state(
        ENSEMBLE_OBJ,
        complex_chains,
        ref_state=1,
        label="whole complex",
    )
    alpha_rmsds = _chain_rmsd_by_state(ENSEMBLE_OBJ, CHAIN_A, ref_state=1)
    beta_rmsds = _chain_rmsd_by_state(ENSEMBLE_OBJ, CHAIN_B, ref_state=1)
    peptide_rmsds = _chain_rmsd_by_state(ENSEMBLE_OBJ, CHAIN_C, ref_state=1) if has_chain_c else []
    out_tsv = os.path.join(OUTPUT_DIR, f"{rank001_base}_chain_rmsd.tsv")
    _write_chain_rmsd_table(out_tsv, ranked_files, rms_list, complex_rmsds, alpha_rmsds, beta_rmsds, peptide_rmsds)

    alpha_per_res_stats = _per_res_rmsd_to_q(ENSEMBLE_OBJ, CHAIN_A, ref_state=1)
    beta_per_res_stats = _per_res_rmsd_to_q(ENSEMBLE_OBJ, CHAIN_B, ref_state=1)
    peptide_per_res_stats = _per_res_rmsd_to_q(ENSEMBLE_OBJ, CHAIN_C, ref_state=1) if has_chain_c else None
    _print_chain_flexibility_comparison(alpha_per_res_stats, beta_per_res_stats, peptide_per_res_stats)
    cmd.rebuild()

    # 5) Export the reference state with pLDDT and RMSD visualizations
    if _obj_exists(RANK001_OBJ):
        cmd.delete(RANK001_OBJ)
    cmd.create(RANK001_OBJ, f"{ENSEMBLE_OBJ} and state 1")

    out_pdb = os.path.join(OUTPUT_DIR, f"{rank001_base}_withRMSD.pdb")
    cmd.save(out_pdb, RANK001_OBJ)
    print(f"[dq] Saved processed rank_001 model with RMSD to: {out_pdb}")

    # Common view for ray tracing
    cmd.hide("everything", RANK001_OBJ)
    cmd.show("cartoon", RANK001_OBJ)
    cmd.orient(RANK001_OBJ)
    cmd.bg_color("white")
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", "off")
    _define_render_colors()
    if FAST_PYMOL_MODE:
        try:
            cmd.set("defer_builds_mode", 1)
        except Exception:
            pass

    # pLDDT (B-factor) image
    cmd.bg_color("white")
    cmd.spectrum("b", PLDDT_PALETTE, RANK001_OBJ)
    cmd.ray(RAY_SIZE[0], RAY_SIZE[1])
    out_png_plddt = os.path.join(OUTPUT_DIR, f"{rank001_base}_pLDDT.png")
    cmd.png(out_png_plddt, dpi=PNG_DPI)
    print(f"[dq] Saved pLDDT-colored image to: {out_png_plddt}")

    # RMSD (occupancy q) image
    cmd.bg_color("white")
    rmsd_color_sel = " or ".join(f"({RANK001_OBJ} and chain {chain})" for chain in complex_chains)
    cmd.color("gray80", RANK001_OBJ)
    cmd.spectrum("q", RMSD_PALETTE, rmsd_color_sel)
    cmd.ray(RAY_SIZE[0], RAY_SIZE[1])
    out_png_rmsd = os.path.join(OUTPUT_DIR, f"{rank001_base}_RMSD.png")
    cmd.png(out_png_rmsd, dpi=PNG_DPI)
    print(f"[dq] Saved RMSD-colored image to: {out_png_rmsd}")

    print("[dq] Pipeline complete.")

if __name__ == "__main__":
    dq_full_pipeline()
    # Clean shutdown for both backends
    if _PYMOL2_SESSION is not None:
        try:
            _PYMOL2_SESSION.stop()
        except Exception:
            pass
    else:
        cmd.quit()
