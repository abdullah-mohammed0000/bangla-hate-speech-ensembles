#!/usr/bin/env python
"""Build the split files (A_*, B_*, test_*) from the raw corpora.

Usage:  python scripts/prepare_data.py --raw data/raw --out data/splits
The script prints OK when every output file is byte-identical to the file used in the thesis.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs.data import prepare_splits  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="folder with the raw corpora (zips or extracted files)")
    ap.add_argument("--out", default="data/splits", help="output folder for the split CSVs")
    a = ap.parse_args()
    prepare_splits(a.raw, a.out)
