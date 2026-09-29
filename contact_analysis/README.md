# Contact Analysis

This directory contains the curated contact-analysis scripts used for the
structural parts of the HLA-DQ/CLIP study. They support the main contact-based
results: CLIP groove retention, CLIP orientation reversal, conserved CLIP-core
anchoring, and alpha/beta interface contact topology under rank-dependent
perturbation.

The code expects repaired, chained PDB files produced by the FoldX workflow,
with the following chain convention:

```text
chain A = HLA-DQ alpha
chain B = HLA-DQ beta
chain C = CLIP
```

Either pass `--root /path/to/root` where available or set:

```bash
export HLA_DQ_ROOT=/path/to/root
```

The release vendors the mature-chain reference tables used to map
allele-specific residue numbers onto the shared DQA1/DQB1 residue axes:

```text
reference_tables/DQA1_prot.tsv
reference_tables/DQB1_prot.tsv
```

## Reproducibility Scope

This directory contains code, not the full structural archive. The scripts
reproduce the contact analyses when supplied with the AF2/FoldX model archive
and the MD-derived contact summaries used in the study. PDB-scanning scripts
expect repaired, chained PDB files from the FoldX workflow; comparison plotters
expect the intermediate TSV files listed below.

## Contact Definition

Unless stated otherwise, a residue pair is counted as being in contact when at
least one heavy atom of one residue lies within 5 Angstrom of at least one
heavy atom of the other residue. Contact probability is the fraction of sampled
structures or frames in which that residue pair satisfies this criterion.

For CLIP-core analyses, the default core is:

```text
M5/R6/M7/A8/T9/P10/L11/L12/M13
```

The frozen analysis parameters used in the publication workflow are:

```text
contact cutoff:                 5 Angstrom heavy-atom distance
CLIP core:                      M5-R6-M7-A8-T9-P10-L11-L12-M13
rank-1 reference window:        r = 1
high-rank perturbation window:  r = 901-1000
contact-shell threshold:        p_contact >= 0.2
recurrent alpha/beta threshold: p_contact >= 0.5
persistent-contact threshold:   p_contact > 0.9
orientation reversal:           CLIP-core orientation dot product < 0
```

## Included Scripts

```text
check_clip_core_dissociation.py
  Tests whether each CLIP-core residue remains within 5 Angstrom of the
  alpha/beta groove across the AF2 rank ensemble.

clip_groove_fluctuation_direction.py
  Builds the groove-relative CLIP coordinate system, computes CLIP
  displacement components, and calls canonical versus reverse-bound CLIP
  orientation.

plot_clip_groove_fluctuation_direction.py
plot_clip_groove_fluctuation_nonrev.py
  Regenerate displacement figures from the geometry TSV, including the
  canonical-orientation-only displacement analysis.

plot_clip_orientation_reversal_heatmap.py
plot_clip_orientation_reversal_heatmap_combined.py
  Convert orientation calls into molecule-specific and high-rank
  orientation-reversal probability heatmaps.

extract_clip_core_contacts_by_rank.py
plot_clip_core_contact_probabilities.py
  Extract groove residues contacting the CLIP core at selected rank windows and
  summarize/plot contact probabilities on the shared residue axes.

clip_contact_register_maps.py
plot_clip_contact_register_maps.py
plot_clip_contact_register_maps_by_reversion.py
plot_clip_contact_register_maps_reversed_only.py
  Resolve which individual CLIP-core residues are contacted by recurrent
  groove residues, including canonical and reverse-bound contact-register maps.

clip_core_residue_contacts_by_orientation.py
  Summarizes CLIP-core residue contacts separately for canonical and
  reverse-bound high-rank conformations.

extract_alpha_beta_contacts_by_rank.py
plot_alpha_beta_interface_contact_comparison.py
  Extract and compare alpha/beta interface contacts for MD, rank-1 AF2, and
  high-rank AF2 contact-topology analyses.
```

Exploratory scripts that combine contact maps with FoldX residue-energy terms
are intentionally not included here. Those analyses depend on a separate
mechanistic-energy layer and are not required for reproducing the core contact
figures.

## Workflow Summary

Run the workflow in the order below. PDB-scanning steps are the expensive
steps; once their TSV outputs exist, the plotting steps are fast.

| Step | Script | Main inputs | Main outputs | Cost |
| --- | --- | --- | --- | --- |
| 1 | `clip_groove_fluctuation_direction.py` | repaired chained rank PDBs | `clip_groove_fluctuation_direction.tsv`, orientation/displacement PDFs, rank-1 direct-contact TSV | expensive |
| 2 | `check_clip_core_dissociation.py` | repaired chained rank PDBs | `clip_core_residue_boundness_by_rank.tsv`, `clip_core_residue_boundness_summary.tsv` | expensive |
| 3 | `extract_clip_core_contacts_by_rank.py` | repaired chained rank PDBs | `clip_core_rank001_direct_contacts.tsv`, `clip_core_highrank901_1000_direct_contacts.tsv` | expensive |
| 4 | `plot_clip_core_contact_probabilities.py` | CLIP-core direct-contact TSVs, reference tables | CLIP-core contact-probability summaries and PDFs | fast |
| 5 | `clip_contact_register_maps.py` | contact-probability summaries, repaired chained rank PDBs | `clip_core_contact_register_rank_zones.tsv`, partial register-map TSVs, register-map PDF | expensive |
| 6 | `plot_clip_contact_register_maps_by_reversion.py` | register-map TSVs, geometry/orientation TSV | canonical and reverse-bound contact-register PDFs | fast |
| 7 | `extract_alpha_beta_contacts_by_rank.py` | repaired chained rank PDBs | alpha/beta direct-contact TSVs | expensive |
| 8 | `plot_alpha_beta_interface_contact_comparison.py` | MD contact summary, AF2 alpha/beta contact TSVs | alpha/beta interface contact-comparison PDFs | fast |

## Figure Mapping

The table below maps scripts to the contact-analysis results they support.
Figure numbering can change without affecting computational provenance.

| Result or figure component | Primary script(s) | Required upstream files |
| --- | --- | --- |
| CLIP-core remains groove-bound | `check_clip_core_dissociation.py` | repaired chained rank PDBs |
| CLIP orientation reversal probability | `clip_groove_fluctuation_direction.py`, `plot_clip_orientation_reversal_heatmap.py`, `plot_clip_orientation_reversal_heatmap_combined.py` | `clip_groove_fluctuation_direction.tsv` |
| Groove-relative CLIP displacement | `clip_groove_fluctuation_direction.py`, `plot_clip_groove_fluctuation_direction.py`, `plot_clip_groove_fluctuation_nonrev.py` | repaired chained rank PDBs or cached geometry TSV |
| Rank-1 and high-rank CLIP-core contact topology | `extract_clip_core_contacts_by_rank.py`, `plot_clip_core_contact_probabilities.py` | CLIP-core direct-contact TSVs |
| Canonical versus reverse-bound CLIP-core contact-register maps | `clip_contact_register_maps.py`, `plot_clip_contact_register_maps_by_reversion.py`, `plot_clip_contact_register_maps_reversed_only.py` | contact summaries, geometry/orientation TSV, register-map partials |
| Canonical versus reverse-bound CLIP-core residue contacts | `clip_core_residue_contacts_by_orientation.py` | geometry/orientation TSV, repaired chained rank PDBs |
| Alpha/beta interface contact topology | `extract_alpha_beta_contacts_by_rank.py`, `plot_alpha_beta_interface_contact_comparison.py` | MD summary TSV, AF2 alpha/beta direct-contact TSVs |

## Example Commands

Run geometry and orientation analysis first:

```bash
python contact_analysis/clip_groove_fluctuation_direction.py --root "$HLA_DQ_ROOT"
```

Then compute CLIP-core boundness:

```bash
python contact_analysis/check_clip_core_dissociation.py --root "$HLA_DQ_ROOT"
```

Extract rank-1 and high-rank CLIP-core contact rows:

```bash
python contact_analysis/extract_clip_core_contacts_by_rank.py \
  --root "$HLA_DQ_ROOT" \
  --rank 1 \
  --out-tsv "$HLA_DQ_ROOT/reports/RMSD_figures/clip_core_rank001_direct_contacts.tsv"

python contact_analysis/extract_clip_core_contacts_by_rank.py \
  --root "$HLA_DQ_ROOT" \
  --rank-start 901 \
  --rank-end 1000 \
  --out-tsv "$HLA_DQ_ROOT/reports/RMSD_figures/clip_core_highrank901_1000_direct_contacts.tsv"
```

Summarize and plot CLIP-core contacts:

```bash
python contact_analysis/plot_clip_core_contact_probabilities.py
```

Build contact-register maps:

```bash
python contact_analysis/clip_contact_register_maps.py --root "$HLA_DQ_ROOT"
python contact_analysis/plot_clip_contact_register_maps_by_reversion.py
```

Extract alpha/beta interface contacts:

```bash
python contact_analysis/extract_alpha_beta_contacts_by_rank.py \
  --root "$HLA_DQ_ROOT" \
  --rank 1 \
  --out-tsv "$HLA_DQ_ROOT/reports/RMSD_figures/alpha_beta_rank001_direct_contacts.tsv"

python contact_analysis/extract_alpha_beta_contacts_by_rank.py \
  --root "$HLA_DQ_ROOT" \
  --rank-start 901 \
  --rank-end 1000 \
  --out-tsv "$HLA_DQ_ROOT/reports/RMSD_figures/alpha_beta_highrank901_1000_direct_contacts.tsv"
```

Then compare MD, rank-1 AF2, and high-rank AF2 alpha/beta contact summaries:

```bash
python contact_analysis/plot_alpha_beta_interface_contact_comparison.py
```

## Notes

Several plotting scripts can regenerate figures directly from intermediate TSV
files. After the expensive PDB-scanning steps have been run once, this is the
preferred route.
