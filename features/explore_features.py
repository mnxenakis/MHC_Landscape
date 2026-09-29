#!/usr/bin/env python3
"""Residue-axis helpers used by the release contact-analysis scripts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_REF_DIR = Path(__file__).resolve().parents[1] / "contact_analysis" / "reference_tables"
DEFAULT_ALPHA_REF = DEFAULT_REF_DIR / "DQA1_prot.tsv"
DEFAULT_BETA_REF = DEFAULT_REF_DIR / "DQB1_prot.tsv"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}
NONSTD_TO_1 = {
    "H1S": "H",
    "H2S": "H",
    "HSD": "H",
    "HSE": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
}
HET_RE = re.compile(r"(?P<a>[0-9.]+)_(?P<b>[0-9.]+)")
ALLELE_RE = re.compile(r"\*(\d+):(\d+)")


@dataclass(frozen=True)
class ChainReference:
    chain: str
    table_path: Path
    axis_df: pd.DataFrame
    allele_columns: dict[str, str]
    allele_sequences: dict[str, pd.DataFrame]
    polymorphic_labels: tuple[str, ...]


@dataclass(frozen=True)
class HeterodimerCodes:
    alpha: str
    beta: str


def to_aa1(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {".", "-", "NAN", "DEL", "GAP"}:
        return "."
    text = text.split("/")[0]
    if text in NONSTD_TO_1:
        return NONSTD_TO_1[text]
    if len(text) == 1:
        return text
    return AA3_TO_1.get(text, text)


def format_axis_label(residue_number: str, residues: Iterable[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for aa in residues:
        if aa not in seen:
            ordered.append(aa)
            seen.add(aa)
    if len(ordered) == 1:
        return f"{ordered[0]}{residue_number}"
    return f"{'/'.join(ordered)}{residue_number}"


def load_reference_table(path: Path, chain: str) -> ChainReference:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "residue_number" not in df.columns:
        raise SystemExit(f"{path}: missing residue_number column")

    df["residue_number"] = df["residue_number"].astype(str).str.strip()
    df["sort_value"] = pd.to_numeric(df["residue_number"], errors="coerce")
    df = df[df["sort_value"].notna()].copy()
    df = df.sort_values("sort_value").reset_index(drop=True)

    allele_columns: dict[str, str] = {}
    for col in df.columns:
        if col in {"residue_number", "sort_value"}:
            continue
        match = ALLELE_RE.search(str(col))
        if match:
            allele_columns[f"{int(match.group(1))}.{int(match.group(2))}"] = col
    if not allele_columns:
        raise SystemExit(f"{path}: could not infer allele columns")

    mature = df[df["sort_value"] >= 1.0].copy()
    axis_rows: list[dict[str, object]] = []
    polymorphic_labels: list[str] = []
    for _, row in mature.iterrows():
        residue_number = str(row["residue_number"])
        residues = [to_aa1(row[col]) for col in allele_columns.values()]
        nongap = sorted({aa for aa in residues if aa != "."})
        includes_gap = any(aa == "." for aa in residues)
        states = (["."] if includes_gap else []) + nongap
        label = format_axis_label(residue_number, states if states else ["."])
        axis_rows.append(
            {
                "msa_resnum": residue_number,
                "display_resnum": residue_number,
                "label": label,
                "sort_value": float(row["sort_value"]),
            }
        )
        if len(states) > 1:
            polymorphic_labels.append(label)

    axis_df = pd.DataFrame(axis_rows)
    label_map = dict(zip(axis_df["msa_resnum"], axis_df["label"], strict=True))

    allele_sequences: dict[str, pd.DataFrame] = {}
    for code, col in allele_columns.items():
        seq = mature[["residue_number", "sort_value", col]].copy()
        seq["aa"] = seq[col].map(to_aa1)
        seq = seq[seq["aa"] != "."].copy().reset_index(drop=True)
        seq["source_resnum"] = np.arange(1, len(seq) + 1, dtype=int)
        seq["axis_label"] = seq["residue_number"].astype(str).map(label_map)
        seq = seq.rename(columns={"residue_number": "msa_resnum"})
        allele_sequences[code] = seq[
            ["source_resnum", "msa_resnum", "sort_value", "aa", "axis_label"]
        ].copy()

    return ChainReference(
        chain=chain,
        table_path=path,
        axis_df=axis_df,
        allele_columns=allele_columns,
        allele_sequences=allele_sequences,
        polymorphic_labels=tuple(polymorphic_labels),
    )


def parse_heterodimer_codes(name: str, alpha_ref: ChainReference, beta_ref: ChainReference) -> HeterodimerCodes:
    match = HET_RE.search(name)
    if not match:
        raise SystemExit(f"Could not parse heterodimer code from filename: {name}")
    alpha = match.group("a")
    beta = match.group("b")
    if alpha not in alpha_ref.allele_columns:
        raise SystemExit(f"{name}: unknown alpha code {alpha}")
    if beta not in beta_ref.allele_columns:
        raise SystemExit(f"{name}: unknown beta code {beta}")
    return HeterodimerCodes(alpha=alpha, beta=beta)
