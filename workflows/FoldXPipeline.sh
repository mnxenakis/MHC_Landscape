#!/bin/bash
#SBATCH -p defq
#SBATCH -J foldxPipe
#SBATCH -o slurm_foldxPipe_%j.out
#SBATCH -e slurm_foldxPipe_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH -t 8:00:00

set -uo pipefail
shopt -s nullglob
# We do not use `set -e` intentionally.  The script runs many FoldX commands
# inside helper functions and GNU parallel; explicit `|| return 1` checks make
# the failure point clearer in the log.

###############################################################################
# FoldXPipeline.sh
#
# Clean, single-pass FoldX workflow for CLIP-loaded HLA-DQ models.
#
# For each raw *.pdb model in the current directory:
#   1. Repair the full trimer.
#   2. Chain-normalize the repaired trimer:
#        A = alpha, B = beta, C = CLIP.
#   3. Compute whole-trimer stability.
#   4. Compute pairwise interactions on the repaired chained trimer:
#        A/B whole, A/B CT-off, A/C whole, B/C whole.
#   5. Extract A, B, C as isolated component PDBs.
#   6. Compute component Stability without a second component-level repair.
#   7. Repair the extracted components.
#   8. Compute component Stability after this second repair.
#
# Optional:
#   --extra-ab-truncations also computes A/B CT+TMD-off and A/B TMD-only
#   interaction energies using the same chain-renaming logic as the legacy
#   foldx_subparts.sh workflow.
###############################################################################

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDX="${FOLDX:-foldx}"
SCRIPT_DIR="${SCRIPT_DIR:-$PIPELINE_DIR}"
PYTHON="${PYTHON:-python}"

# After `detect_chain.py`, all downstream code assumes these chain IDs.
# Do not change them locally unless detect_chain.py is changed accordingly.
CHAIN_A="A"
CHAIN_B="B"
CHAIN_C="C"

JOBS="${SLURM_CPUS_PER_TASK:-1}"
# Keep reruns cheap: if an expected output already exists, the corresponding
# step is skipped.  Set SKIP_EXISTING=0 when you need to force recomputation.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_TMP="${CLEAN_TMP:-1}"
EXTRA_AB_TRUNCATIONS=0
DRALPHA=0

usage() {
    cat <<EOF
Usage: $0 [--extra-ab-truncations] [--DRalpha]

Run in a directory containing raw AlphaFold/ColabFold *.pdb files.

Options:
  --extra-ab-truncations   Also compute AB CT+TMD-off and AB TMD-only.
  --DRalpha                Pass --DRalpha to align_seq.py when segments.json
                           must be generated.

Environment knobs:
  FOLDX                    FoldX binary path.
  SCRIPT_DIR               Directory containing detect_chain.py, align_seq.py,
                           and rename_chain.py.
  PYTHON                   Python interpreter for helper scripts.
  SEGMENTS_FILE            Segment boundary JSON (default: segments.json).
  SEGMENTS_PDB             Optional PDB passed to align_seq.py.
  SKIP_EXISTING=0|1        Skip existing outputs when possible (default: 1).
  CLEAN_TMP=0|1            Remove per-model temporary directory (default: 1).

Output summary:
  After all FoldX jobs finish, the script writes:
        ../<pair_dir_name>_FoldXEnergies.tsv
  with one row per AF2 rank.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra-ab-truncations)
            EXTRA_AB_TRUNCATIONS=1
            shift
            ;;
        --DRalpha)
            DRALPHA=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if command -v module >/dev/null 2>&1; then
    module load parallel >/dev/null 2>&1 || true
fi

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "[error] missing required file: $path" >&2
        exit 1
    fi
}

require_executable() {
    local tool="$1"

    # Accept either an executable path or a command available on PATH.
    if [[ -x "$tool" ]]; then
        return 0
    fi
    if command -v "$tool" >/dev/null 2>&1; then
        return 0
    fi

    echo "[error] executable not found: $tool" >&2
    exit 1
}

require_executable "$FOLDX"
require_file "$SCRIPT_DIR/detect_chain.py"
require_file "$SCRIPT_DIR/align_seq.py"
require_file "$SCRIPT_DIR/rename_chain.py"

###############################################################################
# detect_chain.py expects FASTA at:
#   <results_parent>/<results_parent_basename>.fasta
# In our datasets, the real FASTA usually matches the current directory stem:
#   <basename_of_current_dir_without__results>.fasta
# Example:
#   cwd: .../1.1_5.1_pept1E_results
#   fasta: .../1.1_5.1/1.1_5.1_pept1E.fasta
# We create/refresh the detect_chain expected filename as a symlink.
###############################################################################

RESULTS_PARENT_DIR="$(dirname "$PWD")"
RESULTS_PARENT_BASE="$(basename "$RESULTS_PARENT_DIR")"
RESULTS_DIR_BASE="$(basename "$PWD")"
RESULTS_DIR_STEM="${RESULTS_DIR_BASE%_results}"
DETECT_CHAIN_FASTA_EXPECTED="$RESULTS_PARENT_DIR/${RESULTS_PARENT_BASE}.fasta"

if [[ ! -f "$DETECT_CHAIN_FASTA_EXPECTED" ]]; then
    CHOSEN_FASTA=""

    # Preferred source: name derived from current *_results directory.
    PREFERRED_FASTA="$RESULTS_PARENT_DIR/${RESULTS_DIR_STEM}.fasta"
    if [[ -f "$PREFERRED_FASTA" ]]; then
        CHOSEN_FASTA="$PREFERRED_FASTA"
    else
        FASTA_CANDIDATES=("$RESULTS_PARENT_DIR"/"${RESULTS_DIR_STEM}"*.fasta)
        if (( ${#FASTA_CANDIDATES[@]} == 1 )); then
            CHOSEN_FASTA="${FASTA_CANDIDATES[0]}"
        elif (( ${#FASTA_CANDIDATES[@]} == 0 )); then
            FASTA_CANDIDATES=("$RESULTS_PARENT_DIR"/"${RESULTS_PARENT_BASE}"*.fasta)
            if (( ${#FASTA_CANDIDATES[@]} == 1 )); then
                CHOSEN_FASTA="${FASTA_CANDIDATES[0]}"
            elif (( ${#FASTA_CANDIDATES[@]} == 0 )); then
                FASTA_CANDIDATES=("$RESULTS_PARENT_DIR"/*.fasta)
            fi
        fi
    fi

    if [[ -n "$CHOSEN_FASTA" ]]; then
        echo "[fasta] detect_chain expected FASTA not found: $DETECT_CHAIN_FASTA_EXPECTED"
        echo "[fasta] using: $CHOSEN_FASTA"
        ln -sfn "$CHOSEN_FASTA" "$DETECT_CHAIN_FASTA_EXPECTED"
    elif (( ${#FASTA_CANDIDATES[@]} > 1 )); then
        echo "[warning] multiple FASTA candidates found in $RESULTS_PARENT_DIR" >&2
        echo "[warning] set up $DETECT_CHAIN_FASTA_EXPECTED manually to avoid ambiguity" >&2
    else
        echo "[warning] no FASTA candidates found in $RESULTS_PARENT_DIR" >&2
        echo "[warning] detect_chain.py may fail if chain relabeling is needed" >&2
    fi
fi

###############################################################################
# Segment boundaries are needed for the CT-off AB interaction and, optionally,
# for CT+TMD-off / TMD-only AB interactions.
###############################################################################

SEGMENTS_FILE="${SEGMENTS_FILE:-segments.json}"
ALIGN_FLAGS=()
if [[ "$DRALPHA" -eq 1 ]]; then
    ALIGN_FLAGS+=(--DRalpha)
fi

if [[ ! -f "$SEGMENTS_FILE" ]]; then
    echo "[segments] $SEGMENTS_FILE not found; generating it with align_seq.py"
    ALIGN_ARGS=()
    if [[ -n "${SEGMENTS_PDB:-}" ]]; then
        ALIGN_ARGS=(--pdb "$SEGMENTS_PDB")
    else
        AUTO_SEGMENTS_PDB=""

        # Prefer rank-like raw models when auto-selecting a PDB for segment
        # detection. This avoids older align_seq.py variants that may fail when
        # no PDB is provided and they try to resolve a FASTA path instead.
        for candidate in *_rank_001*.pdb *_rank_*.pdb *.pdb; do
            case "$candidate" in
                *_Repair*.pdb|*_chained.pdb|*_alpha_whole.pdb|*_beta_whole.pdb|*_alpha_chainA.pdb|*_beta_chainB.pdb|*_CLIP_chainC.pdb|*_AB_CT_off.pdb|*_AB_CT_TMD_off.pdb)
                    continue
                    ;;
            esac
            AUTO_SEGMENTS_PDB="$candidate"
            break
        done

        if [[ -n "$AUTO_SEGMENTS_PDB" ]]; then
            echo "[segments] auto-using PDB: $AUTO_SEGMENTS_PDB"
            ALIGN_ARGS=(--pdb "$AUTO_SEGMENTS_PDB")
        fi
    fi
    "$PYTHON" "$SCRIPT_DIR/align_seq.py" "${ALIGN_ARGS[@]}" "${ALIGN_FLAGS[@]}" --output "$SEGMENTS_FILE" || {
        echo "[error] align_seq.py failed" >&2
        echo "[hint] set SEGMENTS_PDB=<a raw model pdb> or provide SEGMENTS_FILE manually" >&2
        exit 1
    }
fi

read CT_A_START CT_A_END TMD_A_START TMD_A_END CT_B_START CT_B_END TMD_B_START TMD_B_END < <(
    "$PYTHON" - "$SEGMENTS_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    data = json.load(handle)

def fetch(chain, key):
    try:
        start, end = data[chain][key]
        return int(start), int(end)
    except Exception as exc:
        raise SystemExit(f"[error] missing {chain}.{key} in {path}: {exc}")

ct_a = fetch("A", "CT")
tmd_a = fetch("A", "TMD")
ct_b = fetch("B", "CT")
tmd_b = fetch("B", "TMD")
print(ct_a[0], ct_a[1], tmd_a[0], tmd_a[1], ct_b[0], ct_b[1], tmd_b[0], tmd_b[1])
PY
) || {
    echo "[error] failed to parse $SEGMENTS_FILE" >&2
    exit 1
}

export FOLDX SCRIPT_DIR PYTHON
export CT_A_START CT_A_END TMD_A_START TMD_A_END
export CT_B_START CT_B_END TMD_B_START TMD_B_END
export SKIP_EXISTING CLEAN_TMP EXTRA_AB_TRUNCATIONS

###############################################################################
# Small helpers
###############################################################################

run_repair() {
    local input_pdb="$1"
    local repaired_pdb="${input_pdb%.pdb}_Repair.pdb"

    # FoldX RepairPDB always writes <input>_Repair.pdb.  This helper only wraps
    # that naming convention and verifies that the output exists.
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$repaired_pdb" ]]; then
        echo "[skip] RepairPDB exists: $repaired_pdb"
        return 0
    fi

    echo "[foldx] RepairPDB $input_pdb"
    "$FOLDX" --command=RepairPDB --pdb="$input_pdb" || return 1

    [[ -f "$repaired_pdb" ]] || {
        echo "[error] expected repaired PDB not found: $repaired_pdb" >&2
        return 1
    }
}

run_stability() {
    local input_pdb="$1"
    local expected="${input_pdb%.pdb}_0_ST.fxout"

    # FoldX Stability writes <input>_0_ST.fxout.  We leave this native name in
    # place for ordinary repaired structures.
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$expected" ]]; then
        echo "[skip] Stability exists: $expected"
        return 0
    fi

    echo "[foldx] Stability $input_pdb"
    "$FOLDX" --command=Stability --pdb="$input_pdb" || return 1

    [[ -f "$expected" ]] || {
        echo "[error] expected Stability output not found: $expected" >&2
        return 1
    }
}

run_stability_as() {
    local input_pdb="$1"
    local output_fxout="$2"
    local native_fxout="${input_pdb%.pdb}_0_ST.fxout"

    # Used only for "NoRepair" component energies.  FoldX will first write its
    # native <component>_0_ST.fxout; we immediately rename it to an explicit
    # *_NoRepair_0_ST.fxout file so it cannot be confused with repaired-chain
    # Stability outputs.
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_fxout" ]]; then
        echo "[skip] Stability exists: $output_fxout"
        return 0
    fi

    run_stability "$input_pdb" || return 1
    mv -f "$native_fxout" "$output_fxout" || {
        echo "[error] failed to rename $native_fxout -> $output_fxout" >&2
        return 1
    }
    [[ -f "$output_fxout" ]] || {
        echo "[error] expected Stability output not found after rename: $output_fxout" >&2
        return 1
    }
    echo "[write] $output_fxout"
}

run_analyse_complex() {
    local input_pdb="$1"
    local chains="$2"
    local prefix="$3"
    local base="${input_pdb%.pdb}"
    local matches=()

    # AnalyseComplex produces several files with FoldX-generated names.  The
    # important one for interaction energy is usually Summary_*.fxout, but we
    # preserve Interaction/Interface/Indiv files too for later auditing.
    if [[ "$SKIP_EXISTING" -eq 1 && -f "${prefix}_Summary.fxout" ]]; then
        echo "[skip] AnalyseComplex exists: ${prefix}_Summary.fxout"
        return 0
    fi

    echo "[foldx] AnalyseComplex $chains on $input_pdb"
    "$FOLDX" --command=AnalyseComplex \
        --pdb="$input_pdb" \
        --analyseComplexChains="$chains" || return 1

    # FoldX writes files named Summary_<pdb-base>_*.fxout etc.
    # Move only when exactly one file matches to avoid silently using stale data.
    matches=(Interaction_"$base"_*.fxout)
    if (( ${#matches[@]} == 1 )); then
        mv -f "${matches[0]}" "${prefix}_Interaction.fxout" || return 1
    elif (( ${#matches[@]} > 1 )); then
        echo "[error] multiple Interaction files matched for $base" >&2
        return 1
    fi

    matches=(Summary_"$base"_*.fxout)
    if (( ${#matches[@]} == 1 )); then
        mv -f "${matches[0]}" "${prefix}_Summary.fxout" || return 1
    elif (( ${#matches[@]} > 1 )); then
        echo "[error] multiple Summary files matched for $base" >&2
        return 1
    fi

    matches=(Interface_Residues_"$base"_*.fxout)
    if (( ${#matches[@]} == 1 )); then
        mv -f "${matches[0]}" "${prefix}_Interface.fxout" || return 1
    elif (( ${#matches[@]} > 1 )); then
        echo "[error] multiple Interface_Residues files matched for $base" >&2
        return 1
    fi

    matches=(Indiv_energies_"$base"_*.fxout)
    if (( ${#matches[@]} == 1 )); then
        mv -f "${matches[0]}" "${prefix}_Indiv.fxout" || return 1
    elif (( ${#matches[@]} > 1 )); then
        echo "[error] multiple Indiv_energies files matched for $base" >&2
        return 1
    fi

    [[ -f "${prefix}_Summary.fxout" || -f "${prefix}_Interaction.fxout" ]] || {
        echo "[warning] AnalyseComplex output was not found for prefix: $prefix" >&2
    }
}

extract_chain() {
    local input_pdb="$1"
    local chain_id="$2"
    local output_pdb="$3"

    # Write one isolated component PDB from the repaired, chain-normalized
    # trimer.  These isolated structures are the input for both:
    #   - NoRepair component Stability
    #   - component-level RepairPDB followed by Stability
    "$PYTHON" - "$input_pdb" "$chain_id" "$output_pdb" <<'PY'
import sys

input_pdb, chain_id, output_pdb = sys.argv[1:4]
kept = 0
with open(input_pdb) as src, open(output_pdb, "w") as dst:
    for line in src:
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            if line[21].strip() == chain_id:
                dst.write(line)
                if line.startswith(("ATOM  ", "HETATM")):
                    kept += 1
        elif line.startswith("END"):
            dst.write(line)
if kept == 0:
    raise SystemExit(f"[error] no atoms found for chain {chain_id} in {input_pdb}")
PY
}

make_ct_off_pdb() {
    local input_pdb="$1"
    local output_pdb="$2"

    # CT-off AB interaction is computed by moving CT residues out of chains
    # A/B into E/F.  AnalyseComplex(A,B) then sees alpha/beta without CT while
    # the coordinates and all non-CT atoms remain unchanged.
    cp -f "$input_pdb" "$output_pdb"
    "$PYTHON" "$SCRIPT_DIR/rename_chain.py" \
        --pdb "$output_pdb" \
        --chain-A-start "$CT_A_START" --chain-A-end "$CT_A_END" \
        --chain-B-start "$CT_B_START" --chain-B-end "$CT_B_END" \
        --replace_A E \
        --replace_B F
}

make_ct_tmd_off_pdb() {
    local ct_off_pdb="$1"
    local output_pdb="$2"

    # Optional extra decomposition: after CT has already been moved to E/F,
    # move TMD residues out of A/B into G/H.  Then:
    #   AnalyseComplex(A,B) = ectodomain-only AB interaction
    #   AnalyseComplex(G,H) = TMD-only AB interaction
    cp -f "$ct_off_pdb" "$output_pdb"
    "$PYTHON" "$SCRIPT_DIR/rename_chain.py" \
        --pdb "$output_pdb" \
        --chain-A-start "$TMD_A_START" --chain-A-end "$TMD_A_END" \
        --chain-B-start "$TMD_B_START" --chain-B-end "$TMD_B_END" \
        --replace_A G \
        --replace_B H
}

process_model() {
    local raw_pdb="$1"
    local raw_base="${raw_pdb%.pdb}"
    # tag_base documents provenance.  All extracted components are named from
    # *_Repair_chained.pdb, so *_Repair_alpha_chainA_NoRepair_0_ST.fxout means:
    #   full trimer repaired once, chain extracted, no second component repair.
    local tag_base="${raw_base}_Repair"
    local repaired_pdb="${raw_base}_Repair.pdb"
    local chained_pdb="${raw_base}_Repair_chained.pdb"
    local tmp_dir="${raw_base}_foldx_pipeline_tmp"

    echo "======================================================================"
    echo "[model] $raw_pdb"
    echo "======================================================================"

    ###########################################################################
    # 1) Repair the full trimer once.
    ###########################################################################
    if [[ -f "$repaired_pdb" ]]; then
        echo "[skip] full-model RepairPDB exists: $repaired_pdb"
    else
        run_repair "$raw_pdb" || return 1
    fi

    ###########################################################################
    # 2) Normalize chain IDs on the repaired full trimer.
    #    This creates the single trusted parent structure for all downstream
    #    calculations: *_Repair_chained.pdb.
    ###########################################################################
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$chained_pdb" ]]; then
        echo "[skip] chained PDB exists: $chained_pdb"
    else
        echo "[chain] detect_chain.py $repaired_pdb"
        "$PYTHON" "$SCRIPT_DIR/detect_chain.py" "$repaired_pdb" || return 1
        [[ -f "$chained_pdb" ]] || {
            echo "[error] expected chained PDB not found: $chained_pdb" >&2
            return 1
        }
    fi

    ###########################################################################
    # 3) Whole-molecule stability.
    ###########################################################################
    run_stability "$chained_pdb" || return 1

    ###########################################################################
    # 4) Pairwise interactions on the repaired full trimer.
    #    These are not repeated after isolated-chain repair.
    ###########################################################################
    run_analyse_complex "$chained_pdb" "A,B" "${tag_base}_AB_whole" || return 1
    run_analyse_complex "$chained_pdb" "A,C" "${tag_base}_AC_whole" || return 1
    run_analyse_complex "$chained_pdb" "B,C" "${tag_base}_BC_whole" || return 1

    ###########################################################################
    # 5) AB interaction with CT removed from chains A/B.
    ###########################################################################
    local ct_off_pdb="${tag_base}_AB_CT_off.pdb"
    make_ct_off_pdb "$chained_pdb" "$ct_off_pdb" || return 1
    run_analyse_complex "$ct_off_pdb" "A,B" "${tag_base}_AB_CT_off" || return 1

    ###########################################################################
    # 6) Optional extra AB decompositions.
    ###########################################################################
    if [[ "$EXTRA_AB_TRUNCATIONS" -eq 1 ]]; then
        local ct_tmd_off_pdb="${tag_base}_AB_CT_TMD_off.pdb"
        make_ct_tmd_off_pdb "$ct_off_pdb" "$ct_tmd_off_pdb" || return 1
        run_analyse_complex "$ct_tmd_off_pdb" "A,B" "${tag_base}_AB_CT_TMD_off" || return 1
        run_analyse_complex "$ct_tmd_off_pdb" "G,H" "${tag_base}_AB_TMD_only" || return 1
    fi

    ###########################################################################
    # 7) Extract individual components from the repaired chained trimer.
    ###########################################################################
    local alpha_pdb="${tag_base}_alpha_chainA.pdb"
    local beta_pdb="${tag_base}_beta_chainB.pdb"
    local clip_pdb="${tag_base}_CLIP_chainC.pdb"
    extract_chain "$chained_pdb" "A" "$alpha_pdb" || return 1
    extract_chain "$chained_pdb" "B" "$beta_pdb" || return 1
    extract_chain "$chained_pdb" "C" "$clip_pdb" || return 1

    ###########################################################################
    # 8) Component Stability without a second component-level repair.
    ###########################################################################
    run_stability_as "$alpha_pdb" "${tag_base}_alpha_chainA_NoRepair_0_ST.fxout" || return 1
    run_stability_as "$beta_pdb" "${tag_base}_beta_chainB_NoRepair_0_ST.fxout" || return 1
    run_stability_as "$clip_pdb" "${tag_base}_CLIP_chainC_NoRepair_0_ST.fxout" || return 1

    ###########################################################################
    # 9) Repair isolated components and compute their repaired Stability.
    ###########################################################################
    run_repair "$alpha_pdb" || return 1
    run_repair "$beta_pdb" || return 1
    run_repair "$clip_pdb" || return 1

    run_stability "${tag_base}_alpha_chainA_Repair.pdb" || return 1
    run_stability "${tag_base}_beta_chainB_Repair.pdb" || return 1
    run_stability "${tag_base}_CLIP_chainC_Repair.pdb" || return 1

    if [[ "$CLEAN_TMP" -eq 1 ]]; then
        rm -rf "$tmp_dir"
    fi

    echo "[done] $raw_pdb"
}

export -f run_repair run_stability run_stability_as run_analyse_complex
export -f extract_chain make_ct_off_pdb make_ct_tmd_off_pdb process_model

###############################################################################
# Select only raw input PDBs.  Derived files are intentionally ignored so reruns
# do not recursively analyze repaired/extracted/truncated structures.
###############################################################################

INPUT_PDBS=()
for pdb in *.pdb; do
    case "$pdb" in
        *_Repair*.pdb|*_chained.pdb|*_alpha_whole.pdb|*_beta_whole.pdb|*_alpha_chainA.pdb|*_beta_chainB.pdb|*_CLIP_chainC.pdb|*_AB_CT_off.pdb|*_AB_CT_TMD_off.pdb)
            continue
            ;;
    esac
    INPUT_PDBS+=("$pdb")
done

if (( ${#INPUT_PDBS[@]} == 0 )); then
    echo "[info] no raw *.pdb files found in $PWD"
    exit 0
fi

###############################################################################
# Every raw input PDB must correspond to exactly one AF2 rank.  This prevents
# accidentally mixing derived files or malformed names into the per-rank TSV.
###############################################################################

declare -A SEEN_RANKS=()
for pdb in "${INPUT_PDBS[@]}"; do
    if [[ "$pdb" =~ rank_(1000|[0-9]{3})([^0-9]|$) ]]; then
        rank="${BASH_REMATCH[1]}"
    else
        echo "[error] raw input PDB has no rank_001...rank_1000 token: $pdb" >&2
        exit 1
    fi

    if [[ -n "${SEEN_RANKS[$rank]:-}" ]]; then
        echo "[error] duplicate AF2 rank detected: rank_$rank" >&2
        echo "        ${SEEN_RANKS[$rank]}" >&2
        echo "        $pdb" >&2
        exit 1
    fi
    SEEN_RANKS[$rank]="$pdb"
done

echo "[run] models=${#INPUT_PDBS[@]} jobs=$JOBS extra_ab_truncations=$EXTRA_AB_TRUNCATIONS"

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$JOBS" process_model ::: "${INPUT_PDBS[@]}" || exit 1
else
    echo "[warning] GNU parallel not found; running serially"
    for pdb in "${INPUT_PDBS[@]}"; do
        process_model "$pdb" || exit 1
    done
fi

"$PYTHON" "$PIPELINE_DIR/extract_foldx_rank_energies.py" --results-dir "$PWD" || exit 1
