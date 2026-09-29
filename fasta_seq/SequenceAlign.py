#!/usr/bin/env python3
"""
SequenceAligner.py

=====================
HOW TO RUN
=====================

Basic usage:

    - For the HLA-DQA1* (01:01 is the reference):
    python SequenceAlign.py DQA1_prot.fasta 01:01 01:02 01:03 01:04 02:01 03:01 03:02 04:01 05:01 05:03 05:05 06:0

    - For the HLA-DQB1* (05:01 is the reference):
    python SequenceAlign.py DQB1_prot.fasta 05:01 02:01 02:02 03:01 03:02 03:03 03:04 03:05 04:01 04:02 05:02 05:03 05:04 06:01 06:02 06:03 06:04 06:09 


With output file:
    python SequenceAlign.py DQB1_prot.fasta 05:01 05:03 06:02 -o output.tsv

Optional: also write aligned FASTA:
    python SequenceAlign.py DQA1_prot.fasta 01:01 01:02 02:01 --aligned-fasta aligned.fasta

Requirements:
    - Python 3
    - Biopython
    - MAFFT installed and available in PATH

=====================
KEY CONCEPTS
=====================

- Sequences are aligned using MAFFT → gaps "-" are introduced
- The FIRST sequence is used as an "anchor" (reference for numbering)
- IMPORTANT:
    "-" in anchor sequence = insertion in other sequences
    "-" in other sequences = deletion relative to anchor

- No alignment columns are skipped
- Residue numbers are emitted as ascending integers across alignment columns
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except Exception as exc:
    print(f"ERROR: Biopython is required: {exc}", file=sys.stderr)
    raise SystemExit(1)


# Match valid allele formats like 01:01 or 01:02:01
ALLELE_RE = re.compile(r"\d+:\d+(?::\d+){0,2}[A-Za-z]?")

# Remove non-letter characters from sequences
NON_SEQ_RE = re.compile(r"[^A-Za-z]")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def validate_allele_code(allele_code: str) -> bool:
    return bool(ALLELE_RE.fullmatch(allele_code))


def choose_chain_type_from_file(fasta_file: str) -> str:
    """Infer chain type from filename."""
    name = Path(fasta_file).name.upper()
    if "DQA1" in name:
        return "DQA1"
    if "DQB1" in name:
        return "DQB1"
    raise ValueError("Filename must contain DQA1 or DQB1.")


def sanitize_sequence(seq: str) -> str:
    """Keep only letters (valid protein residues)."""
    return NON_SEQ_RE.sub("", seq).upper()


def extract_allele_after_first_star(header: str) -> Optional[str]:
    """
    Extract allele code after '*' in FASTA header.
    Example: HLA-DQA1*01:01 → 01:01
    """
    s = header.strip().lstrip(">")
    star_index = s.find("*")
    if star_index == -1:
        return None

    match = re.match(r"([0-9A-Za-z:]+)", s[star_index + 1 :])
    return match.group(1) if match else None


def parse_fasta_database(fasta_file: Path) -> List[Tuple[str, str]]:
    """Read FASTA and clean sequences."""
    records = list(SeqIO.parse(str(fasta_file), "fasta"))
    if not records:
        fail(f"No FASTA records found in {fasta_file}")

    parsed = []
    for record in records:
        seq = sanitize_sequence(str(record.seq))
        if seq:
            parsed.append((record.description.strip(), seq))

    if not parsed:
        fail("No usable sequences found.")

    return parsed


def resolve_allele(records, requested):
    """
    Resolve requested allele:
    - exact match first
    - otherwise first prefix match
    """
    for header, seq in records:
        allele = extract_allele_after_first_star(header)
        if allele == requested:
            return requested, allele, seq

    for header, seq in records:
        allele = extract_allele_after_first_star(header)
        if allele and allele.startswith(requested):
            return requested, allele, seq

    return None


def make_unique_names(names):
    """Ensure sequence IDs are unique."""
    counts = {}
    out = []

    for name in names:
        if name not in counts:
            counts[name] = 1
            out.append(name)
        else:
            counts[name] += 1
            out.append(f"{name}_{counts[name]}")

    return out


def build_selected_records(fasta_records, chain_type, requested_alleles):
    """Select and label sequences."""
    names, seqs = [], []

    print("Resolved alleles:")
    for requested in requested_alleles:
        resolved = resolve_allele(fasta_records, requested)
        if resolved is None:
            fail(f"Could not resolve allele: {requested}")

        _, allele, seq = resolved
        label = f"HLA-{chain_type}*{allele}"

        print(f"  {requested} -> {allele}")

        names.append(label)
        seqs.append(seq)

    names = make_unique_names(names)

    return [SeqRecord(Seq(s), id=n, description=n) for n, s in zip(names, seqs)]


def run_mafft(records, mafft_exe):
    """Run MAFFT alignment."""
    if shutil.which(mafft_exe) is None:
        fail("MAFFT not found.")

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        SeqIO.write(records, tmp, "fasta")
        tmp_path = tmp.name

    proc = subprocess.run(
        [mafft_exe, "--auto", tmp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        fail(proc.stderr)

    return proc.stdout


def parse_aligned_fasta(text):
    """Parse MAFFT output."""
    records = list(SeqIO.parse(StringIO(text), "fasta"))

    if not records:
        fail("Failed to parse alignment.")

    return records

def build_table(aligned_records, offset):
    """
    Build alignment table with mature-sequence numbering.

    offset:
        number of residues to subtract (signal peptide length)
    """

    names = [r.id for r in aligned_records]
    seqs = [str(r.seq).upper() for r in aligned_records]

    ref = seqs[0]

    alignment_pos = 0

    rows = []
    header = ["residue_number"] + names

    for col in range(len(ref)):
        alignment_pos += 1
        res_id = str(alignment_pos - offset)

        row = [res_id]

        for seq in seqs:
            aa = seq[col]
            row.append("del" if aa == "-" else aa)

        rows.append(row)

    return header, rows

# mature sequence starts with 1,2,3, ...
def get_offset(chain_type):
    if chain_type == "DQA1":
        return 23
    elif chain_type == "DQB1":
        return 32
    else:
        fail("Unknown chain type for offset.")

def write_tsv(header, rows, out_path):
    with open(out_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fasta")
    parser.add_argument("alleles", nargs="+")
    parser.add_argument("-o", "--output-tsv")
    parser.add_argument("--mafft", default="mafft")

    args = parser.parse_args()

    fasta = Path(args.input_fasta)
    out = args.output_tsv or fasta.with_suffix(".tsv")

    records = parse_fasta_database(fasta)
    chain = choose_chain_type_from_file(str(fasta))
    selected = build_selected_records(records, chain, args.alleles)

    aln_text = run_mafft(selected, args.mafft)
    aligned = parse_aligned_fasta(aln_text)

    offset = get_offset(chain)
    header, rows = build_table(aligned, offset)
    write_tsv(header, rows, out)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()