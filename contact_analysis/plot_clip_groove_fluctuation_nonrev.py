#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import pandas as pd

try:
    from .clip_groove_fluctuation_direction import OUT_DIR, ROOT, plot_figure
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import OUT_DIR, ROOT, plot_figure


DEFAULT_INPUT_TSV = OUT_DIR / "clip_groove_fluctuation_direction.tsv"
DEFAULT_CALLS_TSV = OUT_DIR / "clip_core_orientation_calls_216.tsv"
DEFAULT_NONREV_TSV = OUT_DIR / "clip_groove_fluctuation_direction_canonical_orientation_216.tsv"
DEFAULT_NONREV_PDF = OUT_DIR / "clip_groove_fluctuation_direction_canonical_orientation_216.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create explicit CLIP orientation calls and regenerate the "
            "groove-relative fluctuation figure for canonical-orientation "
            "CLIP-core conformations."
        )
    )
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--calls-tsv", type=Path, default=DEFAULT_CALLS_TSV)
    parser.add_argument("--nonrev-tsv", type=Path, default=DEFAULT_NONREV_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_NONREV_PDF)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--orientation-threshold", type=float, default=0.0)
    return parser.parse_args()


def alpha_code(heterodimer: str) -> str:
    return str(heterodimer).split("_", 1)[0]


def main() -> int:
    args = parse_args()
    if not args.input_tsv.exists():
        raise SystemExit(f"Missing input TSV: {args.input_tsv}")
    df = pd.read_csv(args.input_tsv, sep="\t")
    required = {"heterodimer", "rank", "clip_core_orientation_dot"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{args.input_tsv}: missing column(s) {sorted(missing)}")

    df["heterodimer"] = df["heterodimer"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["clip_core_orientation_dot"] = pd.to_numeric(df["clip_core_orientation_dot"], errors="coerce")
    df = df.dropna(subset=["rank", "clip_core_orientation_dot"]).copy()
    df["rank"] = df["rank"].astype(int)

    exclude = set(args.exclude_alpha or [])
    df = df.loc[~df["heterodimer"].map(alpha_code).isin(exclude)].copy()
    df["clip_orientation_state"] = "canonical"
    df.loc[df["clip_core_orientation_dot"] < args.orientation_threshold, "clip_orientation_state"] = "rev"
    df["is_clip_orientation_reversed"] = df["clip_orientation_state"].eq("rev")

    calls = df[
        [
            "heterodimer",
            "rank",
            "pdb_file",
            "clip_core_orientation_dot",
            "clip_core_orientation_angle_deg",
            "clip_core_endpoint_same_nm",
            "clip_core_endpoint_flip_nm",
            "clip_core_endpoint_flip_minus_same_nm",
            "clip_orientation_state",
            "is_clip_orientation_reversed",
        ]
    ].copy()
    args.calls_tsv.parent.mkdir(parents=True, exist_ok=True)
    calls.to_csv(args.calls_tsv, sep="\t", index=False)
    print(f"[write] {args.calls_tsv}")

    nonrev = df.loc[~df["is_clip_orientation_reversed"]].copy()
    nonrev.to_csv(args.nonrev_tsv, sep="\t", index=False)
    print(f"[write] {args.nonrev_tsv}")

    plot_figure(nonrev, args.out_pdf)
    print(f"[write] {args.out_pdf}")

    total_rows = len(df)
    rev_rows = int(df["is_clip_orientation_reversed"].sum())
    print(
        "[summary] "
        f"heterodimers={df['heterodimer'].nunique()} ranks={df['rank'].nunique()} "
        f"observations={total_rows} reversed={rev_rows} "
        f"p_rev={rev_rows / total_rows:.6f}"
    )
    high = df.loc[df["rank"].between(901, 1000)]
    if not high.empty:
        high_rev = int(high["is_clip_orientation_reversed"].sum())
        print(
            "[summary high-rank 901-1000] "
            f"observations={len(high)} reversed={high_rev} "
            f"p_rev={high_rev / len(high):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
