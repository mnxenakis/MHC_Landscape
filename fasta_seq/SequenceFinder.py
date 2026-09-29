#!/usr/bin/env python3
"""
Compare HLA-DQ allele protein sequences from a FASTA file and optionally
write the exact terminal output to an HTML report.

Behavior
--------
1) Accept DQA1_prot.fasta or DQB1_prot.fasta as input.
2) Resolve allele inputs deterministically:
   - exact match if available
   - otherwise first prefix match in FASTA order
3) If one allele is given, compare it against the chain reference allele:
   - DQA1 reference = 01:01:01:01
   - DQB1 reference = 05:01:01:01
4) If two alleles are given, compare those two resolved alleles.
5) Report:
   - sequence lengths
   - whether lengths differ
   - global alignment (BLOSUM62)
   - percent identity and entropy metrics across all global alignment columns,
     including indels
6) Optional HTML reproduces the exact CMD output.

Notes
-----
- Output is alignment-focused only.
- Residue numbering has been removed.
- Indels/deletions are treated as biologically meaningful events and are
  counted in alignment-wide metrics.
"""

from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"

ALLELE_RE = re.compile(r"\d+:\d+(?::\d+){0,2}[A-Za-z]?")
NON_SEQ_RE = re.compile(r"[^A-Za-z*]")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

REFERENCE_ALLELES = {
    "DQA1": {"allele": "01:01:01:01"},
    "DQB1": {"allele": "05:01:01:01"},
}

Record = Tuple[str, str]
ResolvedMatch = Tuple[str, str, str]  # requested, resolved allele token, sequence
CMD_LOG: List[str] = []


def emit(*args, sep: str = " ", end: str = "\n") -> None:
    text = sep.join(str(a) for a in args) + end
    sys.stdout.write(text)
    CMD_LOG.append(text)


def sanitize_sequence(seq: str) -> str:
    return NON_SEQ_RE.sub("", seq).upper()


def validate_allele_code(allele_code: str) -> bool:
    return bool(ALLELE_RE.fullmatch(allele_code))


def extract_allele_after_first_star(header: str) -> Optional[str]:
    s = header.strip().lstrip(">")
    star_index = s.find("*")
    if star_index == -1:
        return None
    match = re.match(r"([0-9A-Za-z:]+)", s[star_index + 1 :])
    return match.group(1) if match else None


def parse_fasta(fasta_file: str) -> List[Record]:
    path = Path(fasta_file)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {fasta_file}")

    records: List[Record] = []
    current_header: Optional[str] = None
    current_chunks: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, sanitize_sequence("".join(current_chunks))))
                current_header = line[1:].strip()
                current_chunks = []
            else:
                if current_header is None:
                    raise ValueError(
                        f"Invalid FASTA format: sequence before first header at line {line_number}"
                    )
                current_chunks.append(line)

    if current_header is not None:
        records.append((current_header, sanitize_sequence("".join(current_chunks))))
    return records


def choose_chain_type_from_file(fasta_file: str) -> str:
    name = Path(fasta_file).name.upper()
    if "DQA1" in name:
        return "DQA1"
    if "DQB1" in name:
        return "DQB1"
    raise ValueError("Could not infer chain type from FASTA filename; expected DQA1 or DQB1.")


def resolve_allele(records: Sequence[Record], requested: str) -> Optional[ResolvedMatch]:
    exact_hits: List[Tuple[str, str]] = []
    prefix_hits: List[Tuple[str, str]] = []

    for header, seq in records:
        allele = extract_allele_after_first_star(header)
        if allele is None:
            continue
        if allele == requested:
            exact_hits.append((allele, seq))
        elif allele.startswith(requested):
            prefix_hits.append((allele, seq))

    if exact_hits:
        allele, seq = exact_hits[0]
        return requested, allele, seq
    if prefix_hits:
        allele, seq = prefix_hits[0]
        return requested, allele, seq
    return None


def get_reference_match(records: Sequence[Record], chain_type: str) -> Tuple[str, str]:
    ref_req = REFERENCE_ALLELES[chain_type]["allele"]
    resolved = resolve_allele(records, ref_req)
    if resolved is None:
        raise ValueError(f"Could not resolve reference allele {ref_req} for {chain_type}.")
    _, resolved_token, seq = resolved
    return resolved_token, seq


def wrap_plain(seq: str, width: int = 80) -> List[str]:
    return [seq[i:i + width] for i in range(0, len(seq), width)]


def wrap_ansi_text(text: str, width: int = 80) -> List[str]:
    lines: List[str] = []
    current: List[str] = []
    visible = 0
    i = 0
    while i < len(text):
        match = ANSI_RE.match(text, i)
        if match:
            current.append(match.group(0))
            i = match.end()
            continue
        current.append(text[i])
        visible += 1
        i += 1
        if visible >= width:
            lines.append("".join(current))
            current = []
            visible = 0
    if current:
        lines.append("".join(current))
    return lines


def print_wrapped_sequence(label: str, seq: str, width: int = 80) -> None:
    emit(f"\n{label}:")
    chunks = wrap_ansi_text(seq, width=width) if "\x1b[" in seq else wrap_plain(seq, width=width)
    for chunk in chunks:
        emit(chunk)


def color_alignment_terminal(aln1: str, aln2: str) -> Tuple[str, str]:
    out1: List[str] = []
    out2: List[str] = []
    for a, b in zip(aln1, aln2):
        if a == b:
            out1.append(a)
            out2.append(b)
        else:
            out1.append(f"{ANSI_RED}{a}{ANSI_RESET}")
            out2.append(f"{ANSI_RED}{b}{ANSI_RESET}")
    return "".join(out1), "".join(out2)


def compute_shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def compute_entropy_similarity(aln1: str, aln2: str) -> Tuple[float, float, int]:
    """
    Compute per-column Shannon entropy across all global alignment columns,
    treating indels as biologically meaningful symbols.

    For a two-sequence alignment, the maximum entropy of a single column is
    1 bit, so similarity is normalized as 1 - average_entropy.
    """
    entropy_sum = 0.0
    columns = 0

    for a, b in zip(aln1, aln2):
        if a == "-" and b == "-":
            continue
        counts: Dict[str, int] = {}
        for residue in (a, b):
            counts[residue] = counts.get(residue, 0) + 1
        entropy_sum += compute_shannon_entropy(counts)
        columns += 1

    if columns == 0:
        return 0.0, 0.0, 0

    avg_entropy = entropy_sum / columns
    similarity = 1.0 - avg_entropy
    return avg_entropy, similarity, columns


def load_aligner():
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except Exception as exc:
        raise RuntimeError(f"Biopython not installed or incomplete: {exc}")
    matrix = substitution_matrices.load("BLOSUM62")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    return aligner


def pad_pair(chunk1: str, chunk2: str) -> Tuple[str, str]:
    if len(chunk1) == len(chunk2):
        return chunk1, chunk2
    if len(chunk1) < len(chunk2):
        return chunk1 + "-" * (len(chunk2) - len(chunk1)), chunk2
    return chunk1, chunk2 + "-" * (len(chunk1) - len(chunk2))


def reconstruct_gapped_alignment(seq1: str, seq2: str, alignment) -> Tuple[str, str]:
    coords = alignment.coordinates
    out1: List[str] = []
    out2: List[str] = []
    for i in range(coords.shape[1] - 1):
        s1_start, s1_end = int(coords[0, i]), int(coords[0, i + 1])
        s2_start, s2_end = int(coords[1, i]), int(coords[1, i + 1])
        step1 = s1_end - s1_start
        step2 = s2_end - s2_start
        if step1 > 0 and step2 > 0:
            chunk1, chunk2 = pad_pair(seq1[s1_start:s1_end], seq2[s2_start:s2_end])
            out1.append(chunk1)
            out2.append(chunk2)
        elif step1 > 0:
            out1.append(seq1[s1_start:s1_end])
            out2.append("-" * step1)
        elif step2 > 0:
            out1.append("-" * step2)
            out2.append(seq2[s2_start:s2_end])
    return "".join(out1), "".join(out2)


def global_align(seq1: str, seq2: str) -> Tuple[str, str, float]:
    aligner = load_aligner()
    alignments = aligner.align(seq1, seq2)
    if len(alignments) == 0:
        raise RuntimeError("No global alignment found.")
    best = alignments[0]
    gapped1, gapped2 = reconstruct_gapped_alignment(seq1, seq2, best)
    return gapped1, gapped2, best.score


def print_alignment_block(
    label1: str,
    label2: str,
    colored1: str,
    colored2: str,
    width: int = 100,
) -> None:
    blocks1 = wrap_ansi_text(colored1, width=width)
    blocks2 = wrap_ansi_text(colored2, width=width)
    label_width = max(len(label1), len(label2))

    if len(blocks1) != len(blocks2):
        raise RuntimeError("Internal formatting error: alignment block counts differ.")

    emit("")
    for b1, b2 in zip(blocks1, blocks2):
        emit(f"{label1.ljust(label_width)}  {b1}")
        emit(f"{label2.ljust(label_width)}  {b2}")
        emit("")


def report_global_alignment(
    seq1: str,
    seq2: str,
    label1: str,
    label2: str,
    width: int = 80,
) -> None:
    try:
        gapped1, gapped2, score = global_align(seq1, seq2)
    except Exception as exc:
        emit("\n❌ Global alignment failed.")
        emit(f"Details: {exc}")
        return

    colored1, colored2 = color_alignment_terminal(gapped1, gapped2)

    emit("\n### Sequence Length Summary ###")
    emit(f"Length {label1}: {len(seq1)} aa")
    emit(f"Length {label2}: {len(seq2)} aa")
    emit(f"Raw length difference: {abs(len(seq1) - len(seq2))} aa")
    if len(seq1) == len(seq2):
        emit("Lengths are identical.")
    else:
        emit("Lengths differ.")

    emit("\n### Global Alignment (BLOSUM62) ###")
    emit(f"Alignment score: {score}")

    print_alignment_block(label1, label2, colored1, colored2, width=width)

    alignment_columns = sum(1 for a, b in zip(gapped1, gapped2) if not (a == "-" and b == "-"))
    matches_all = sum(1 for a, b in zip(gapped1, gapped2) if a == b and not (a == "-" and b == "-"))
    pid_all = (matches_all / alignment_columns * 100.0) if alignment_columns else 0.0

    aligned_residue_positions = sum(1 for a, b in zip(gapped1, gapped2) if a != "-" and b != "-")
    residue_matches = sum(1 for a, b in zip(gapped1, gapped2) if a == b and a != "-" and b != "-")
    pid_residue_only = (
        residue_matches / aligned_residue_positions * 100.0 if aligned_residue_positions else 0.0
    )

    avg_entropy, similarity, columns = compute_entropy_similarity(gapped1, gapped2)

    emit(f"\nTotal alignment columns including indels: {alignment_columns}")
    emit(f"Percent identity over all alignment columns including indels: {pid_all:.2f}%")
    emit(f"Aligned residue-residue positions only: {aligned_residue_positions}")
    emit(f"Percent identity over residue-residue positions only: {pid_residue_only:.2f}%")
    emit(f"Average Shannon entropy (all global alignment columns including indels): {avg_entropy:.4f}")
    emit(f"Entropy-based similarity (all global alignment columns including indels, 0–1): {similarity:.4f}")
    emit(f"Alignment columns used for global entropy: {columns}")


def ansi_to_html(text: str) -> str:
    parts: List[str] = []
    i = 0
    red_on = False

    while i < len(text):
        match = ANSI_RE.match(text, i)
        if match:
            code = match.group(0)
            if code == ANSI_RED:
                if red_on:
                    parts.append("</span>")
                parts.append('<span class="diff">')
                red_on = True
            elif code == ANSI_RESET:
                if red_on:
                    parts.append("</span>")
                    red_on = False
            i = match.end()
            continue

        parts.append(html.escape(text[i]))
        i += 1

    if red_on:
        parts.append("</span>")
    return "".join(parts)


def build_html_report(output_file: str, cmd_output: str) -> None:
    cmd_output_html = ansi_to_html(cmd_output)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sequence comparison report</title>
<style>
body {{
    background: white;
    color: black;
    font-family: Arial, Helvetica, sans-serif;
    margin: 28px;
}}
h1 {{
    margin-bottom: 0.5em;
}}
.cmdblock {{
    background: white;
    color: black;
    border: 1px solid #ccc;
    padding: 14px;
    font-family: "Courier New", Courier, monospace;
    font-size: 14px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
}}
.diff {{
    color: #cc0000;
    font-weight: 700;
}}
</style>
</head>
<body>
<h1>Command-line output report</h1>
<div class="cmdblock">{cmd_output_html}</div>
</body>
</html>
"""
    Path(output_file).write_text(doc, encoding="utf-8")


def print_usage() -> None:
    emit("Usage:")
    emit("  python SequenceFinder.py FASTA_FILE ALLELE")
    emit("  python SequenceFinder.py FASTA_FILE ALLELE [OUTPUT.html]")
    emit("  python SequenceFinder.py FASTA_FILE ALLELE1 ALLELE2 [OUTPUT.html]")
    emit("Example:")
    emit("  python SequenceFinder.py DQB1_prot.fasta 05:03")
    emit("  python SequenceFinder.py DQB1_prot.fasta 05:03 report_beta_0503.html")
    emit("  python SequenceFinder.py DQA1_prot.fasta 01:04 02:01")


def parse_cli(argv: Sequence[str]) -> Tuple[str, str, Optional[str], Optional[str]]:
    if len(argv) not in (3, 4, 5):
        raise ValueError("usage")

    fasta_file = argv[1]
    allele1 = argv[2]
    allele2: Optional[str] = None
    output_html: Optional[str] = None

    if len(argv) == 4:
        if argv[3].lower().endswith(".html"):
            output_html = argv[3]
        else:
            allele2 = argv[3]
    elif len(argv) == 5:
        allele2 = argv[3]
        output_html = argv[4]

    return fasta_file, allele1, allele2, output_html


def main() -> int:
    try:
        fasta_file, allele1_req, allele2_req, output_html = parse_cli(sys.argv)
    except ValueError:
        print_usage()
        return 1

    for allele in (allele1_req, allele2_req):
        if allele is not None and not validate_allele_code(allele):
            emit(f"❌ Invalid allele code: {allele}")
            emit("Expected 2-field, 3-field, or 4-field form, e.g. 01:01, 03:05:01, or 01:02:01:01")
            return 1

    try:
        records = parse_fasta(fasta_file)
        chain_type = choose_chain_type_from_file(fasta_file)
    except Exception as exc:
        emit(f"❌ Failed to initialize from {fasta_file}: {exc}")
        return 1

    resolved1 = resolve_allele(records, allele1_req)
    if resolved1 is None:
        emit(f"❌ Could not resolve allele: {allele1_req}")
        return 1
    _, resolved_token1, seq1 = resolved1
    emit(f"Found sequence 1: {allele1_req} -> {resolved_token1}")
    label1 = f"HLA-{chain_type}*{resolved_token1}"

    if allele2_req is None:
        ref_token, ref_seq = get_reference_match(records, chain_type)
        seq2 = ref_seq
        label2 = f"HLA-{chain_type}*{ref_token}"
        emit(f"Found reference sequence: {ref_token}")
    else:
        resolved2 = resolve_allele(records, allele2_req)
        if resolved2 is None:
            emit(f"❌ Could not resolve allele: {allele2_req}")
            return 1
        _, resolved_token2, seq2 = resolved2
        label2 = f"HLA-{chain_type}*{resolved_token2}"
        emit(f"Found sequence 2: {allele2_req} -> {resolved_token2}")

    report_global_alignment(seq1, seq2, label1, label2)

    if output_html:
        try:
            build_html_report(output_html, "".join(CMD_LOG))
            emit(f"\nHTML report written: {output_html}")
        except Exception as exc:
            emit(f"\n❌ Failed to write HTML report: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())