# MHC_Landscape

This repository contains the key scripts used to analyze structure-derived
energetic rules of CLIP-loaded HLA-DQ heterodimers. We focus on a
panel of full-length, mature HLA-DQ molecules formed by combining
12 HLA-DQ alpha chains with 18 HLA-DQ beta chains, giving
`12 x 18 = 216` alpha/beta heterodimers. For each heterodimer, rank-ordered
ColabFold/AlphaFold2 models were generated and analyzed with FoldX, RMSD-based
structural comparisons, and distribution-space clustering.

This release contains the core scripts for the main computational workflows.

## Biological Motivation

Stable CLIP-loaded alpha/beta heterodimer formation is an
early energetic checkpoint: alpha/beta combinations must form a favorable
scaffold while also accommodating CLIP at a manageable energetic cost.

The scripts in this release quantify this balance. FoldX calculations decompose
each modeled heterodimer into whole-molecule stability, intrinsic chain terms,
and pairwise interaction terms. Rank-ordered ColabFold/AlphaFold2 models are
then treated as an ensemble of alternative CLIP-loaded conformational states for
each molecule. The clustering algorithms ask whether these rank-wise energetic
fingerprints define reproducible alpha/beta pairing regimes, and whether those
regimes correspond to biologically meaningful cis/trans organization.

The same workflow can be used for any peptide-loaded HLA-DQ complex by replacing the CLIP sequence in the input FASTA.

## FASTA Input Convention

Each ColabFold input FASTA represents one peptide-loaded HLA-DQ heterodimer as
a three-chain complex:

```text
alpha_sequence:beta_sequence:peptide_sequence
```

The colon-separated order is important. Downstream FoldX and RMSD workflows
assume that the modeled complex can be normalized to:

```text
chain A = HLA-DQ alpha
chain B = HLA-DQ beta
chain C = peptide
```

The alpha and beta entries correspond to the full-length mature HLA-DQ chains
used in the 216-molecule panel. In this study, the peptide entry corresponds to
CLIP by default, but the same format can be used to model alternative loaded
peptides.

## Directory Layout

The analysis assumes one parent directory per alpha-chain allele and one child
directory per alpha/beta combination:

```text
ROOT/
  X.Y/
    X.Y_Z.W/
      X.Y_Z.W_results/
        *_rank_001_*.pdb
        *_rank_002_*.pdb
        ...
        *_rank_1000_*.pdb
```

Here:

- `X.Y` identifies the HLA-DQ alpha-chain code.
- `Z.W` identifies the HLA-DQ beta-chain code.
- `X.Y_Z.W` identifies one alpha/beta heterodimer.
- `X.Y_Z.W_results/` contains the rank-ordered ColabFold/AlphaFold2 models for
  that heterodimer.

For example:

```text
1.1/
  1.1_5.1/
    1.1_5.1_results/
      DQA1_1_1_DQB1_5_1_unrelaxed_rank_001_*.pdb
      ...
```

Most scripts are designed to be run either inside a single `X.Y_Z.W_results`
directory or from the top-level `ROOT/` directory containing all alpha-chain
parents.

## Data Scope

This repository contains code, not the full structural archive. ...

## Included Code

```text
DQ_clustering/
  Core Python package for energy-distribution clustering, bootstrap analysis,
  silhouette/cohesion scoring, and FoldX energy summaries.

fasta_seq/
  DQA1_prot.fasta
  DQB1_prot.fasta
  SequenceAlign.py
  SequenceFinder.py

workflows/
  colabfold_gpu.sh
  FoldXPipeline.sh
  extract_foldx_rank_energies.py

rmsd/
  alignEngine.py

contact_analysis/
  Curated scripts for CLIP-core boundness, CLIP orientation reversal,
  groove-relative CLIP displacement, CLIP-core/groove contact-register maps,
  and alpha/beta interface contact topology.

features/
  Minimal residue-axis helper used by contact-analysis plotting scripts.
```

### `fasta_seq/`

This directory contains the input-sequence utilities used to construct and
inspect the HLA-DQ alpha and beta sequences before ColabFold modeling.

- `DQA1_prot.fasta`
  Protein FASTA records for available HLA-DQA1 alleles from the IPD-IMGT/HLA
  database.
- `DQB1_prot.fasta`
  Protein FASTA records for available HLA-DQB1 alleles from the IPD-IMGT/HLA
  database.
- `SequenceFinder.py`
  Resolves one or two allele codes from the FASTA database and reports sequence
  length, global BLOSUM62 alignment, percent identity, and entropy-based
  similarity. If only one allele is supplied, it is compared with the chain
  reference allele.
- `SequenceAlign.py`
  Selects multiple DQA1 or DQB1 alleles, aligns them with MAFFT, and writes a
  mature-sequence-numbered residue table. The script subtracts the signal
  peptide offset used in this project (`23` residues for DQA1 and `32` residues
  for DQB1), so the output table is indexed in mature-chain coordinates.

Examples:

```bash
python fasta_seq/SequenceFinder.py fasta_seq/DQA1_prot.fasta 01:01 05:01

python fasta_seq/SequenceAlign.py fasta_seq/DQB1_prot.fasta \
  05:01 02:01 02:02 03:01 03:02 \
  -o DQB1_selected_alignment.tsv
```

### `workflows/colabfold_gpu.sh`

SLURM workflow for generating rank-ordered ColabFold models from an input FASTA
file. It writes one `<basename>_results/` directory containing ranked PDB and
JSON output files.

### `workflows/FoldXPipeline.sh`

Single-pass FoldX workflow for one `X.Y_Z.W_results/` directory. For each raw
ranked PDB model it:

1. repairs the full trimer;
2. normalizes chains so that alpha is chain `A`, beta is chain `B`, and CLIP is
   chain `C`;
3. computes whole-molecule stability;
4. computes pairwise interaction energies for `A/B`, `A/C`, and `B/C`;
5. extracts alpha, beta, and CLIP components;
6. computes component stabilities before and after component-level repair;
7. writes a parent-level rank-wise energy table through
   `extract_foldx_rank_energies.py`.

Run from one results directory:

```bash
sbatch /path/to/workflows/FoldXPipeline.sh
```

The FoldX binary can be supplied with:

```bash
FOLDX=/path/to/foldx sbatch /path/to/workflows/FoldXPipeline.sh
```

The workflow uses the bundled helper scripts `detect_chain.py`,
`align_seq.py`, and `rename_chain.py` by default. You can override their
location with `SCRIPT_DIR=/path/to/helpers` if needed. The Python interpreter
defaults to `python` and can be overridden with `PYTHON=/path/to/python`.

### `workflows/extract_foldx_rank_energies.py`

Extracts one row per AF2 rank from the FoldX outputs generated by
`FoldXPipeline.sh`. By default, it writes:

```text
../X.Y_Z.W_FoldXEnergies.tsv
```

from inside `X.Y_Z.W_results/`.

### `rmsd/alignEngine.py`

PyMOL-based workflow for rank-wise RMSD analysis. It aligns each rank model to
the corresponding rank-1 model using non-cytoplasmic-tail C-alpha atoms from
the alpha and beta chains, then computes post-fit C-alpha RMSD values for the
alpha chain, beta chain, CLIP peptide, and whole alpha/beta/CLIP complex.

Run from one results directory containing repaired chained PDB files:

```bash
python /path/to/rmsd/alignEngine.py
```

### `contact_analysis/`

This directory contains the structural contact-analysis code used to reproduce
the CLIP-boundness, CLIP-reorientation, CLIP-core/groove contact-register, and
alpha/beta interface contact-topology results. It is intentionally curated:
exploratory scripts combining contact maps with FoldX residue-energy terms are
not included in this release subset.

See `contact_analysis/README.md` for the contact definition, expected inputs,
script order, and figure-oriented workflows.

## Python CLI

The publication package is named `DQ_clustering`. From this release directory:

```bash
python -m DQ_clustering.cli_final --help
```

The CLI exposes four workflows:

```text
cluster
bootstrap
bootstrap-plot
energy-summary
```

### Clustering

The clustering workflow represents each heterodimer by a rank-wise energy
distribution, typically the alpha/beta interaction energy distribution, and
clusters these distributions without using biological class labels.

Biologically, this tests whether HLA-DQ molecules partition into distinct
alpha/beta energetic regimes using only their computed energy-distribution
fingerprints. Class labels such as cis/trans status are used only after
clustering to interpret whether a label-free energetic partition corresponds to
known pairing behavior.

Example:

```bash
python -m DQ_clustering.cli_final cluster ROOT \
  --metric interaction_AB \
  --mol-part CT_off \
  --cluster-mode bregman \
  --clusters 2 \
  --metadata-dir /path/to/metadata
```

### Bootstrap and Silhouette/Cohesion Scoring

Bootstrap workflows quantify stability of the inferred clustering structure
across random resampling. The clustering code also computes silhouette-based
criteria in Jensen-Shannon distribution space and class/cohesion summaries used
to evaluate whether the most favorable energetic cluster is enriched in cis or
trans heterodimers.

The silhouette score provides a class-agnostic measure of how naturally the
energy-distribution space can be partitioned for a given number of clusters.
The cohesion summaries then ask whether the energetically most favorable
cluster is preferentially occupied by biologically relevant molecule classes.
This separates cluster selection from biological interpretation.

Example:

```bash
python -m DQ_clustering.cli_final bootstrap ROOT \
  --metric interaction_AB \
  --mol-part CT_off \
  --cluster-mode bregman \
  --clusters 2 \
  --bootstrap 1000 \
  --metadata-dir /path/to/metadata
```

### Energy Summary

The energy-summary workflow builds per-heterodimer scalar summaries from
rank-wise FoldX energy distributions, including median, mean, IQR, mode, and
normalized ensemble free-energy summaries.

These summaries convert the 1000-rank energy distribution of each molecule into
compact descriptors for comparing alpha/beta coupling, CLIP accommodation, and
whole-molecule stability across the 216-heterodimer panel. The ensemble
free-energy measure is computed from the normalized rank ensemble,
`F = -RT log(mean_r exp[-E(r)/RT])`, using molar energy units. This convention
removes the artificial `-RT log(N_r)` sample-count contribution.

Example:

```bash
python -m DQ_clustering.cli_final energy-summary ROOT \
  --metric interaction_AB \
  --mol-parts whole CT_off \
  --summary ensemble \
  --ensemble-RT 0.596 \
  --metadata-dir /path/to/metadata
```

## Metadata

The clustering workflows use an external metadata workbook to map heterodimer
labels to biological classes. The metadata directory can be supplied explicitly:

```bash
--metadata-dir /path/to/metadata --metadata-pattern "*.xlsx"
```

Alternatively, set:

```bash
export DQ_METADATA_DIR=/path/to/metadata
```

## Installation

Create an environment with the Python dependencies listed in `pyproject.toml`.
For local use from the release directory:

```bash
pip install -e .
```

The workflow scripts also require external software:

- ColabFold for `workflows/colabfold_gpu.sh`;
- FoldX for `workflows/FoldXPipeline.sh`;
- PyMOL Python bindings for `rmsd/alignEngine.py`;
- MAFFT for `fasta_seq/SequenceAlign.py`;
- repaired, chained PDB files for the `contact_analysis/` scripts.

### Tested Dependencies

The following were checked during release validation on Linux:

- Python `3.10.18`
- FoldX `5.1`
- `pip install -e .` from the release root
- `python -m DQ_clustering.cli_final --help`
- `python workflows/extract_foldx_rank_energies.py --help`
- `python fasta_seq/SequenceFinder.py fasta_seq/DQA1_prot.fasta 01:01`
- `python fasta_seq/SequenceAlign.py --help`
- one-model real-data subset run of `workflows/FoldXPipeline.sh` with FoldX `5.1`

The following tools are required for specific workflows but were not exercised
in the same fresh-environment validation pass:

- MAFFT for `fasta_seq/SequenceAlign.py` alignment runs
- PyMOL Python bindings for `rmsd/alignEngine.py`

### Reproducibility Smoke Test

The commands below provide a minimal verification path from a fresh
environment.

1. Install the package from the release root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Verify the publication CLI starts:

```bash
python -m DQ_clustering.cli_final --help
```

Expected result: the command exits successfully and lists the subcommands
`cluster`, `bootstrap`, `bootstrap-plot`, and `energy-summary`.

3. Verify the FoldX energy extractor starts:

```bash
python workflows/extract_foldx_rank_energies.py --help
```

Expected result: the command exits successfully and shows the default output
name `../<parent-dir-name>_FoldXEnergies.tsv`.

4. Verify the sequence helper scripts start:

```bash
python fasta_seq/SequenceFinder.py fasta_seq/DQA1_prot.fasta 01:01
python fasta_seq/SequenceAlign.py --help
```

Expected result: `SequenceFinder.py` prints a resolved allele comparison
against the chain reference, and `SequenceAlign.py --help` exits successfully.

5. Verify the FoldX workflow on one real model copied into a temporary
`X.Y/X.Y_Z.W/X.Y_Z.W_results/` directory together with its parent FASTA, then
run:

```bash
FOLDX=/path/to/foldx bash /path/to/workflows/FoldXPipeline.sh
```

Expected result:

- the workflow writes `segments.json` inside the `*_results` directory;
- the workflow writes `../X.Y_Z.W_FoldXEnergies.tsv` in the parent pair
  directory;
- the TSV contains one row per processed rank and populated energy columns.

## Notes

This release is not a complete archive of every exploratory plotting script used
during analysis. It is a curated set of the key scripts needed to reproduce the
rank-wise ColabFold/FoldX/RMSD/contact-analysis data generation steps and the
main energy-distribution clustering analyses.
