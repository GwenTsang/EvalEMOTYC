#!/usr/bin/env python3
"""Prépare 4 sous-ensembles XLSX EMOTYC, contigus ou non contigus."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = [
    ("homophobie", Path("golds/homophobie_annotations_gold_flat_updated.xlsx")),
    ("obésité", Path("golds/obésité_annotations_gold_flat_updated.xlsx")),
    ("religion", Path("golds/religion_annotations_gold_flat_updated.xlsx")),
    ("racisme", Path("golds/racisme_annotations_gold_flat_updated.xlsx")),
]


def repo(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-files", nargs="+", type=Path,
                        help="Remplace les 4 XLSX par défaut. Chemins relatifs au repo.")
    parser.add_argument("--mode", choices=("contiguous", "noncontiguous"), default="contiguous")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path,
                        help="Défaut: results/prepared_xlsx_samples_<timestamp>")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size doit être > 0")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
    sources = [(p.stem, p) for p in args.source_files] if args.source_files else DEFAULT_SOURCES
    out_dir = repo(args.out_dir) if args.out_dir else ROOT / "results" / f"prepared_xlsx_samples_{datetime.now():%Y%m%d_%H%M%S}"
    subsets_dir = out_dir / "subsets"

    print(f"Seed: {seed}")
    print(f"Mode: {args.mode}")
    print(f"Sample size: {args.sample_size}")

    manifest = {
        "seed": seed,
        "mode": args.mode,
        "sample_size": args.sample_size,
        "subsets_dir": str(subsets_dir),
        "samples": [],
    }

    if not args.dry_run:
        subsets_dir.mkdir(parents=True, exist_ok=True)

    for order, (label, raw_path) in enumerate(sources, start=1):
        source = repo(raw_path)
        if not source.exists():
            raise FileNotFoundError(source)

        df = pd.read_excel(source)
        if len(df) < args.sample_size:
            raise ValueError(f"{source.name}: {len(df)} lignes < sample-size {args.sample_size}")

        if args.mode == "contiguous":
            start = rng.randint(0, len(df) - args.sample_size)
            rows = list(range(start, start + args.sample_size))
            range_tag = f"rows_{rows[0]}_{rows[-1]}"
            row_desc = f"start={rows[0]} end_exclusive={rows[-1] + 1} excel_rows={rows[0] + 2}-{rows[-1] + 2}"
        else:
            start = None
            rows = sorted(rng.sample(range(len(df)), args.sample_size))
            range_tag = f"{args.sample_size}_rows"
            preview = rows[:8]
            row_desc = f"rows_0based={preview}{'...' if len(rows) > len(preview) else ''}"

        out_file = subsets_dir / f"{order:02d}_{label}_{args.mode}_{range_tag}.xlsx"
        print(f"  {order:02d}. {label}: {source.name} len={len(df)} {row_desc}")

        info = {
            "order": order,
            "label": label,
            "source_file": str(source),
            "source_n_rows": len(df),
            "output_file": str(out_file),
            "rows_0based": rows,
            "excel_rows": [r + 2 for r in rows],
            "start_0based": start,
            "end_exclusive_0based": None if start is None else rows[-1] + 1,
        }
        manifest["samples"].append(info)

        if args.dry_run:
            continue

        sample = df.iloc[rows].copy().reset_index(drop=True)
        for col, values in reversed([
            ("sample_seed", seed),
            ("sample_mode", args.mode),
            ("sample_label", label),
            ("sample_source_file", source.name),
            ("sample_source_row_0based", rows),
            ("sample_excel_row", [r + 2 for r in rows]),
            ("sample_subset_pos", list(range(len(rows)))),
        ]):
            sample.insert(0, col, values)
        sample.to_excel(out_file, index=False)

    if args.dry_run:
        print("\nDry-run: aucun fichier écrit.")
        return

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest["samples"]).to_excel(out_dir / "manifest.xlsx", index=False)
    print(f"\nSous-ensembles: {subsets_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
