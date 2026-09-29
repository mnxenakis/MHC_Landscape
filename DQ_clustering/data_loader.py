"""Dataset loading utilities for FoldX/DQ analysis.

Responsibilities:
1. Discover directories that encode allele labels (``DQA`` parent and ``DQA_DQB`` combinations).
2. Locate FoldX energy output files supporting both *new* and *legacy* layouts.
3. Parse energy tables tolerating optional leading model-name column and NA markers.
4. Enrich datasets with per-label class metadata sourced from an Excel grid.
5. Apply optional rank filtering while keeping extra energy component arrays aligned.

Why this exists instead of ad‑hoc scripts:
- Centralizes layout assumptions (naming regexes, file naming patterns) so changing
    disk organization or adding another molecule partition only requires editing here.
- Separates metadata (Excel) concerns from raw energy parsing for clearer testing paths.
- Provides small, composable helpers (`iter_foldx_energy_dirs`, `locate_foldx_energy_file`,
    `read_energy_columns`) that can be reused in alternate loaders (e.g., streaming, lazy).

Design notes:
- Folder selection uses two regexes for DQA and DQA_DQB directory names and is
    otherwise layout-agnostic (does not assume deeper nesting).
- When a class map is provided, datasets without a known class are skipped
    (and optionally collected in ``missing_labels``). This keeps downstream
    plots consistent when class-specific highlighting is expected.
- Magic numbers and patterns are consolidated as named constants and documented
    below so they can be tuned centrally if the layout evolves.

Performance considerations (current scale):
- Files are read fully line-by-line into Python lists; for typical FoldX result sizes
    this is inexpensive. If thousands of large files appear, consider refactoring
    `read_energy_columns` into a generator yielding chunks, or memory-map for speed.
- Rank filtering is vectorized via boolean masks; no Python loops are used there.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from .algorithms import FoldxEnergyDataset, format_class_value

# ------------------------------
# Directory layout patterns
# ------------------------------
# Parent and combined directory name patterns.
# - Parent: either a numeric allele like "1.1" or a DR alpha parent "DRA".
# - Combined: either a numeric pair like "1.1_2.3" or "DRA_2.3".
DQA = re.compile(r"^(?:DQA.*|DRA|\d+\.\d+)$")  # e.g., "DQA*", "DRA", or allele like "1.1"
DQA_DQB = re.compile(r"^(?:DQB.*|DRA_\d+\.\d+|\d+\.\d+_\d+\.\d+)$")  # e.g., "DQB*", "DRA_2.3", or "1.1_2.3"

# ------------------------------
# Metadata discovery defaults
# ------------------------------
# DEFAULT_DIR: Default location of the Excel workbook with label metadata.
# Override with DQ_METADATA_DIR or pass --metadata-dir on the CLI.
# PATTERN: Glob used to pick the first workbook if multiple exist
# LAUNCH: If True, open the workbook interactively instead of parsing it
DEFAULT_DIR = Path(os.environ.get("DQ_METADATA_DIR", ".")).expanduser().resolve()
PATTERN = "*.xlsx"  # Workbook discovery pattern (first match used)
LAUNCH = False  # If True: open interactively instead of parsing into DataFrame


def open_default_xlsx(
    default_dir: Path = DEFAULT_DIR,
    pattern: str = PATTERN,
    launch: bool = LAUNCH,
) -> pd.DataFrame | None:
    """Open the first XLSX file in the default directory or load it into pandas.

    Contract:
    - If launch=True, the first matching workbook is opened via OS handler and
      the function returns None (preview mode).
    - Otherwise the workbook is loaded into a pandas DataFrame and returned.

        Notes:
        - The first matching file by name sort is chosen if multiple workbooks exist.
        - To control sheet selection or parsing options, use pandas directly on the
            returned path in a custom workflow.
    """

    default_dir = default_dir.expanduser().resolve()
    if not default_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {default_dir}")

    matches = sorted(default_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {default_dir}")

    target = matches[0]
    if launch:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target)], check=False)
        else:
            subprocess.run(["xdg-open", str(target)], check=False)
        return None

    df = pd.read_excel(target)
    return df


def collect_label_class_pairs(
    df: pd.DataFrame,
    *,
    row_idx: int = 0,
    pattern: str | re.Pattern[str] = r"^(?:\d+\.\d+|DRA)$",
    suffix_pattern: str | re.Pattern[str] = r"^\d+\.\d+$",
) -> tuple[list[str], list[str], list[tuple[str, object]]]:
    """Extract parent labels, suffix labels, and label/class pairs from Excel.

    The Excel grid is treated as a matrix where:
    - The first column contains parent allele codes (e.g., 1.1, DRA)
    - The selected row (row_idx) contains suffix allele codes (e.g., 2.3)
    - The cell at (parent_row, suffix_col) contains the class value for the
      combined label "parent_suffix" (e.g., 1.1_2.3)

    Parameters:
    - df: Workbook contents loaded as a DataFrame (first sheet by default)
    - row_idx: Row index containing suffix allele codes (0-based)
    - pattern: Regex describing allowed parent labels; can be a precompiled pattern
    - suffix_pattern: Regex describing allowed suffix labels; can be a precompiled pattern

    Returns:
    - unique parent labels (as strings)
    - unique suffix labels (as strings)
    - list of ("parent_suffix", class_value) tuples
    """

    if isinstance(pattern, str):  # Allow caller to pass precompiled pattern for reuse
        pattern = re.compile(pattern)
    if isinstance(suffix_pattern, str):
        suffix_pattern = re.compile(suffix_pattern)

    first_column = (
        df.iloc[:, 0]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    first_col_mask = first_column.str.match(pattern)
    first_col_indices = np.flatnonzero(first_col_mask.to_numpy())
    if first_col_indices.size == 0:
        raise ValueError("No allele-formatted entries found in the first column.")
    first_col_labels = [first_column.iat[idx] for idx in first_col_indices]

    if df.shape[0] <= row_idx:
        raise ValueError("Excel sheet does not contain the requested row.")

    row_series = (
        df.iloc[row_idx]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    row_mask = row_series.str.match(suffix_pattern)
    row_indices = np.flatnonzero(row_mask.to_numpy())
    if row_indices.size == 0:
        raise ValueError("No allele-formatted entries found in the selected row.")
    row_labels = [row_series.iat[idx] for idx in row_indices]

    pairs: list[tuple[str, object]] = []
    for row_pos in first_col_indices:
        parent = first_column.iat[row_pos]
        for col_pos in row_indices:
            suffix = row_series.iat[col_pos]
            value = df.iat[row_pos, col_pos]
            pairs.append((f"{parent}_{suffix}", value))
    unique_parents = list(dict.fromkeys(first_col_labels))
    unique_suffixes = list(dict.fromkeys(row_labels))
    return unique_parents, unique_suffixes, pairs


def iter_foldx_energy_dirs(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield ``(DQA_dir, DQA_DQB_dir)`` pairs that appear to encode energy datasets.

    Traversal contract:
    - Scans only the immediate children of ``root`` for parent allele directories.
    - For each parent directory, scans only one level deeper for combined labels.

    Rationale:
    - Eliminates deep recursive walks which could introduce unrelated folders.
    - Keeps ordering deterministic (sorted) aiding reproducible downstream plots.
    """

    for dqa_dir in sorted(p for p in root.iterdir() if p.is_dir() and DQA.match(p.name)):
        for dqa_dqb_dir in sorted(p for p in dqa_dir.iterdir() if p.is_dir() and DQA_DQB.match(p.name)):
            yield dqa_dir, dqa_dqb_dir


def locate_foldx_energy_file(dqa_dqb_dir: Path, mol_part: str = "whole") -> tuple[Path | None, str]:
    """Resolve the canonical energy file path for a combined allele directory.

    Layout precedence (first found wins):
    1. New explicit part file: ``<label>_foldxEnergies_<mol_part>.txt``
    2. New implicit file (no part suffix): ``<label>_foldxEnergies.txt``
    3. Legacy ``*_results/freeEnergies.txt`` inside a results subdirectory.

    Notes:
    - Returning ``None`` signals absence; caller decides to skip silently.
    - ``mol_part`` allows future extension (e.g., "chainA", "interface") without
      altering discovery logic elsewhere.
    """

    preferred_name = f"{dqa_dqb_dir.name}_foldxEnergies_{mol_part}.txt"
    candidate = dqa_dqb_dir / preferred_name
    if candidate.is_file():
        return candidate, "explicit"

    fallback_name = f"{dqa_dqb_dir.name}_foldxEnergies.txt"
    candidate = dqa_dqb_dir / fallback_name
    if candidate.is_file():
        return candidate, "implicit"

    legacy_results = next((d for d in dqa_dqb_dir.iterdir() if d.is_dir() and d.name.endswith("_results")), None)
    if legacy_results is None:
        combo = dqa_dqb_dir / f"{dqa_dqb_dir.parent.name}_{dqa_dqb_dir.name}_results"
        legacy_results = combo if combo.is_dir() else None
    if legacy_results is not None:
        legacy_file = legacy_results / "freeEnergies.txt"
        if legacy_file.is_file():
            return legacy_file, "legacy"
    return None, "missing"


# List the extra FoldX columns we expect, in order. Keeping this as a named
# constant avoids scattering column names across the parser.
NA_MARKERS = {"na", "nan", "none", ""}  # Tokens treated explicitly as missing/NA

FOLDX_EXTRA_KEYS: tuple[str, ...] = (
    # Order matters; parsed sequentially to align columns by position.
    # If upstream adds/removes columns, update here only.
    "intraclashes_A",
    "intraclashes_B",
    "interaction_AB",
    "stability_A",
    "stability_B",
)


def _parse_float_token(token: str) -> tuple[float | None, bool]:
    """Convert a whitespace-delimited token into a float.

    Returns ``(value, was_na)`` where:
    - ``value`` is ``None`` if parsing failed (non-numeric string), else float (possibly ``nan``).
    - ``was_na`` is True only for explicit NA markers (informational for higher-level warnings).

    Distinguishing ``was_na`` from generic parse failure lets callers decide
    whether to silently skip invalid numeric junk while still warning on NA prevalence.
    """

    stripped = token.strip()
    lowered = stripped.lower()
    if lowered in NA_MARKERS:
        return float("nan"), True
    try:
        return float(stripped), False
    except ValueError:
        return None, False



def read_energy_columns(foldx_energy_file: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Extract rank, primary energy, and additional FoldX component arrays.

    Format (whitespace-separated):
        Optional model name | rank | primary energy | extra columns...

    Autodetection strategy:
    - Attempt to parse first token as integer rank; if that fails, treat first token as model name.
    - Energy token is then the next column. Remaining columns mapped onto ``FOLDX_EXTRA_KEYS`` by order.

    Robustness features:
    - Skips comment/blank lines.
    - NA markers converted to ``nan`` and flagged (single consolidated warning per file).
    - Invalid numeric tokens in extra columns become ``nan`` (preserves alignment lengths).
    - Rows with invalid/missing primary energy are dropped entirely (cannot contribute statistics).

    Returns
    -------
    ranks : ndarray[int32]
        Integer ranks (AlphaFold or model ordering).
    energies : ndarray[float64]
        Primary FoldX energy values (no non-finite values).
    extras : dict[str, ndarray[float64]]
        Additional component arrays; each length matches ``energies``.
    """

    ranks: list[int] = []
    energies: list[float] = []
    extras: dict[str, list[float]] = {k: [] for k in FOLDX_EXTRA_KEYS}
    saw_na_token = False

    with foldx_energy_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            
            # Minimum required: rank & energy (2 cols) OR model name + rank + energy (3 cols)
            if len(parts) < 2:
                continue
            
            # Format autodetect: try interpreting first token as rank; else treat it as model name
            try:
                rank_token = parts[0]
                if rank_token.strip().lower() in NA_MARKERS:
                    saw_na_token = True
                    continue
                rank = int(rank_token)
                energy_token = parts[1]
                extras_start_idx = 2
            except (ValueError, IndexError):
                # First column is likely a model name; shift indices accordingly
                if len(parts) < 3:
                    continue
                try:
                    rank_token = parts[1]
                    if rank_token.strip().lower() in NA_MARKERS:
                        saw_na_token = True
                        continue
                    rank = int(rank_token)
                    energy_token = parts[2]
                    extras_start_idx = 3
                except (ValueError, IndexError):
                    continue

            energy, energy_was_na = _parse_float_token(energy_token)
            if energy_was_na:
                saw_na_token = True
            if energy is None or not np.isfinite(energy):
                # Skip rows with missing/invalid primary energy: statistics need a numeric base
                continue
            
            ranks.append(rank)
            energies.append(energy)
            
            # Parse extra energy components preserving row alignment
            for offset, key in enumerate(FOLDX_EXTRA_KEYS):
                col_idx = extras_start_idx + offset
                value = float("nan")
                if col_idx < len(parts):
                    parsed, was_na = _parse_float_token(parts[col_idx])
                    if was_na:
                        saw_na_token = True
                    if parsed is not None:
                        value = parsed
                extras[key].append(value)

    ranks_arr = np.asarray(ranks, dtype=np.int32)
    energies_arr = np.asarray(energies, dtype=np.float64)
    extras_arr = {key: np.asarray(values, dtype=np.float64) for key, values in extras.items()}
    if saw_na_token:
        print(f"Warning: detected NA tokens in {foldx_energy_file}; treated as NaN or skipped.")
    return ranks_arr, energies_arr, extras_arr


def iter_foldx_energy_datasets(
    root: Path,
    *,
    class_map: dict[str, object] | None = None,
    missing_labels: set[str] | None = None,
    log_file: Path | None = None,
    mol_part: str = "whole",
) -> Iterable[FoldxEnergyDataset]:
    """Iterate over foldx energy datasets and yield their raw energy samples.

    - class_map: optional mapping from label (e.g., "1.1_2.3") to class value
      (0/1 or any object consumable by format_class_value)
        - missing_labels: optionally collect labels missing from class_map
    - log_file: optional path to write a one-line summary per dataset discovered

        Behavior:
        - If ``class_map`` is provided, only datasets whose label exists in the map
            are yielded; others are skipped (and optionally recorded in
            ``missing_labels``).
    """

    log_handle = log_file.open("w", encoding="utf-8") if log_file is not None else None  # One-line summaries
    for dqa_dir, dqa_dqb_dir in iter_foldx_energy_dirs(root):
        foldx_energy_file, source = locate_foldx_energy_file(dqa_dqb_dir, mol_part=mol_part)
        if foldx_energy_file is None:
            continue
        if source != "explicit":
            print(
                f"Warning: using {source} foldx energies for {dqa_dqb_dir.name} "
                f"(requested mol-part={mol_part})."
            )
        ranks, energies, extras = read_energy_columns(foldx_energy_file)
        if energies.size == 0:
            continue
        
        # Validate that ranks and energies have matching sizes
        if ranks.size != energies.size:
            import warnings
            warnings.warn(
                f"Size mismatch in {foldx_energy_file}: ranks={ranks.size}, energies={energies.size}. "
                f"Truncating to minimum length.",
                RuntimeWarning,
            )
            min_len = min(energies.size, ranks.size)
            energies = energies[:min_len]
            ranks = ranks[:min_len]
            extras = {k: v[:min_len] for k, v in extras.items()}
        
        class_value = None
        if class_map is not None:
            class_value = class_map.get(dqa_dqb_dir.name)
            if class_value is None:
                if missing_labels is not None:
                    missing_labels.add(dqa_dqb_dir.name)
                continue
        # Basic summary stats for optional logging (cheap operations on small arrays)
        mean = float(np.mean(energies))
        std = float(np.std(energies))
        median = float(np.median(energies))
        q1, q3 = np.quantile(energies, [0.25, 0.75])
        iqr = float(q3 - q1)
        class_repr = format_class_value(class_value)
        line = (
            f"{dqa_dqb_dir.name}\tclass={class_repr}\t"
            f"mean={mean:.4f}\tstd={std:.4f}\tmedian={median:.4f}\tiqr={iqr:.4f}"
        )
        if log_handle is not None:
            log_handle.write(line + "\n")
        yield FoldxEnergyDataset(dqa_dir.name, dqa_dqb_dir.name, energies, ranks, extras, class_value)
    if log_handle is not None:
        log_handle.close()


@dataclass
class Metadata:
    """Container for label metadata extracted from the Excel sheet."""

    dataframe: pd.DataFrame
    parents: list[str]
    suffixes: list[str]
    label_classes: list[tuple[str, object]]


class DatasetLoader:
    """High-level orchestrator for metadata ingestion + dataset preparation.

    Workflow:
    1) load_metadata(): read Excel and extract (parent, suffix) label grid and
       label->class pairs
    2) load_datasets(): walk directories, parse energies, apply rank filters, and
       attach class values from the map

     Notes:
     - ``load_datasets`` applies inclusive rank bounds (``rank_min``/``rank_max``)
        to both the primary energies and all extra energy arrays.
     - Datasets with no remaining samples after filtering are skipped.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def load_metadata(
        self,
        *,
        excel_dir: Optional[Path] = None,
        pattern: str = "*.xlsx",
        launch: bool = False,
    ) -> Metadata:
        """Load the Excel metadata and return structured label information.

        - excel_dir: directory to look for the workbook (defaults to DEFAULT_DIR)
        - pattern: glob pattern, the first matching workbook is used
        - launch: if True, open the workbook interactively instead of parsing
        """

        df = open_default_xlsx(
            default_dir=excel_dir or DEFAULT_DIR,
            pattern=pattern,
            launch=launch,
        )
        if df is None:
            raise RuntimeError("Excel metadata preview requested but not loaded.")
        parents, suffixes, label_classes = collect_label_class_pairs(df, row_idx=0)
        return Metadata(dataframe=df, parents=parents, suffixes=suffixes, label_classes=label_classes)

    def load_datasets(
        self,
        *,
        class_map: Optional[dict[str, object]] = None,
        missing_labels: Optional[set[str]] = None,
        log_file: Optional[Path] = None,
        rank_min: Optional[int] = None,
        rank_max: Optional[int] = None,
        mol_part: str = "whole",
    ) -> list[FoldxEnergyDataset]:
        """Iterate through energy folders and return prepared datasets.

        - rank_min, rank_max: inclusive bounds applied to AlphaFold ranks;
          if omitted, all ranks are included.
        """

        if not self.root.is_dir():
            raise FileNotFoundError(f"Root path does not exist or is not a directory: {self.root}")

        raw_datasets = list(
            iter_foldx_energy_datasets(
                self.root,
                class_map=class_map,
                missing_labels=missing_labels,
                log_file=log_file,
                mol_part=mol_part,
            )
        )
        datasets: list[FoldxEnergyDataset] = []
        lower = int(rank_min) if rank_min is not None else None
        upper = int(rank_max) if rank_max is not None else None
        for ds in raw_datasets:
            # Apply inclusive rank bounds if provided
            mask = np.ones_like(ds.energies, dtype=bool)  # Start with all samples selected
            if lower is not None:
                mask &= ds.ranks >= lower
            if upper is not None:
                mask &= ds.ranks <= upper

            filtered_energies = ds.energies[mask]
            filtered_ranks = ds.ranks[mask]
            filtered_extras = {k: v[mask] for k, v in ds.extra_energies.items()}  # Maintain positional alignment

            # Skip datasets with no samples in the selected rank range
            if filtered_energies.size == 0:
                continue

            # Only materialise a new dataclass if any filtering occurred
            if filtered_energies.size != ds.energies.size:  # Only clone dataclass if a change occurred
                ds = replace(
                    ds,
                    energies=filtered_energies,
                    ranks=filtered_ranks,
                    extra_energies=filtered_extras,
                )
            datasets.append(ds)
        if not datasets:
            raise RuntimeError("No foldx energy datasets found.")
        return datasets


__all__ = [
    "DatasetLoader",
    "Metadata",
    "FoldxEnergyDataset",
    "open_default_xlsx",
    "collect_label_class_pairs",
    "iter_foldx_energy_datasets",
    "locate_foldx_energy_file",
    "read_energy_columns",
]
