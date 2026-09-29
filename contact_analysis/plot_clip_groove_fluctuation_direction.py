#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .clip_groove_fluctuation_direction import (
        DEFAULT_ORIENTATION_PDF,
        DEFAULT_PDF,
        DEFAULT_TSV,
        plot_figure,
        plot_orientation_figure,
        print_summary,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from clip_groove_fluctuation_direction import (
        DEFAULT_ORIENTATION_PDF,
        DEFAULT_PDF,
        DEFAULT_TSV,
        plot_figure,
        plot_orientation_figure,
        print_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the CLIP groove fluctuation direction PDF from the cached TSV."
    )
    parser.add_argument("--in-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--orientation-pdf", type=Path, default=DEFAULT_ORIENTATION_PDF)
    parser.add_argument("--exclude-alpha", nargs="*", default=["1.5", "3.3", "5.2"])
    parser.add_argument("--orientation-threshold", type=float, default=0.0)
    return parser.parse_args()


def alpha_code(heterodimer: str) -> str:
    return str(heterodimer).split("_", 1)[0]


def main() -> int:
    args = parse_args()
    if not args.in_tsv.exists():
        raise SystemExit(f"Missing cached TSV: {args.in_tsv}")

    df = pd.read_csv(args.in_tsv, sep="\t")

    plot_df = df.copy()
    if args.exclude_alpha:
        exclude = set(args.exclude_alpha)
        plot_df = plot_df.loc[~plot_df["heterodimer"].map(alpha_code).isin(exclude)].copy()
    if "clip_core_orientation_dot" not in plot_df.columns:
        raise SystemExit(
            f"{args.in_tsv}: missing clip_core_orientation_dot; rerun the geometry extraction first."
        )
    plot_df["clip_core_orientation_dot"] = pd.to_numeric(
        plot_df["clip_core_orientation_dot"], errors="coerce"
    )
    plot_df = plot_df.loc[plot_df["clip_core_orientation_dot"] >= args.orientation_threshold].copy()

    plot_figure(plot_df, args.out_pdf)
    plot_orientation_figure(df, args.orientation_pdf)
    print_summary(plot_df)
    print(f"[write] {args.out_pdf}")
    if "clip_core_orientation_dot" in df.columns:
        print(f"[write] {args.orientation_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
