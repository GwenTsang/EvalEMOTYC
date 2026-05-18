#!/usr/bin/env python3
"""Lance emotyc_predict.py séparément sur chaque XLSX, puis fusionne les sorties."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_XLSX_FILES = [
    Path("golds/homophobie_annotations_gold_flat_updated.xlsx"),
    Path("golds/obésité_annotations_gold_flat_updated.xlsx"),
    Path("golds/religion_annotations_gold_flat_updated.xlsx"),
    Path("golds/racisme_annotations_gold_flat_updated.xlsx"),
]


def repo(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def selected_files(args: argparse.Namespace) -> list[Path]:
    if args.xlsx_files:
        files = [repo(p) for p in args.xlsx_files]
    elif args.input_dir:
        input_dir = repo(args.input_dir)
        files = sorted(p for p in input_dir.glob("*.xlsx") if p.is_file() and not p.name.startswith("~$"))
    else:
        files = [repo(p) for p in DEFAULT_XLSX_FILES]

    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Fichiers absents:\n" + "\n".join(str(p) for p in missing))
    if not files:
        raise SystemExit("Aucun XLSX sélectionné.")
    return files


def merge_outputs(out_dir: Path, runs: list[dict]) -> None:
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    frames, summaries = [], []

    with (merged_dir / "emotyc_predictions_merged.jsonl").open("w", encoding="utf-8") as merged_jsonl:
        for run in runs:
            if run["status"] != "ok":
                continue

            input_file = Path(run["input_file"])
            run_dir = Path(run["output_dir"])
            source = pd.read_excel(input_file)
            meta_cols = [c for c in source.columns if c.startswith("sample_")]

            pred_xlsx = run_dir / "emotyc_predictions_output.xlsx"
            if pred_xlsx.exists():
                pred = pd.read_excel(pred_xlsx)
                if len(pred) != len(source):
                    raise RuntimeError(f"Taille incohérente: {pred_xlsx} ({len(pred)} vs {len(source)})")
                pred.insert(0, "orchestrator_source_file", input_file.name)
                pred.insert(1, "orchestrator_run_order", run["order"])
                for col in reversed(meta_cols):
                    pred.insert(2, col, source[col].to_numpy())
                frames.append(pred)

            pred_jsonl = run_dir / "emotyc_predictions.jsonl"
            if pred_jsonl.exists():
                meta = source[meta_cols].to_dict("records") if meta_cols else [{} for _ in range(len(source))]
                for i, line in enumerate(pred_jsonl.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record.update({
                        "orchestrator_source_file": input_file.name,
                        "orchestrator_input_file": str(input_file),
                        "orchestrator_run_order": run["order"],
                    })
                    if i < len(meta):
                        record.update(meta[i])
                    merged_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary = run_dir / "emotyc_predictions_summary.json"
            if summary.exists():
                data = json.loads(summary.read_text(encoding="utf-8"))
                data.update({
                    "orchestrator_source_file": input_file.name,
                    "orchestrator_input_file": str(input_file),
                    "orchestrator_run_order": run["order"],
                })
                summaries.append(data)

    if frames:
        pd.concat(frames, ignore_index=True).to_excel(merged_dir / "emotyc_predictions_output_merged.xlsx", index=False)
    (merged_dir / "emotyc_predictions_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--xlsx-files", nargs="+", type=Path,
                        help="Liste explicite de XLSX. Chemins relatifs au repo.")
    source.add_argument("--input-dir", type=Path,
                        help="Dossier de XLSX à traiter, utile pour results/.../subsets.")
    parser.add_argument("--out-dir", type=Path, help="Défaut: results/orchestrated_emotyc_<timestamp>")
    parser.add_argument("--predict-script", type=Path, default=Path("emotyc_predict_details.py"),
                        help="Script d'inférence à utiliser. Défaut: emotyc_predict_details.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--template", choices=("bca", "bca_spaced"), default="bca_spaced")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode-threshold", type=float, default=0.06)
    parser.add_argument("--no-context", dest="use_context", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--continue-on-error", dest="stop_on_error", action="store_false")
    parser.set_defaults(use_context=True, merge=True, stop_on_error=True)
    args = parser.parse_args()

    files = selected_files(args)
    out_dir = repo(args.out_dir) if args.out_dir else ROOT / "results" / f"orchestrated_emotyc_{datetime.now():%Y%m%d_%H%M%S}"
    predict_script = repo(args.predict_script)
    python = str(repo(Path(args.python))) if "/" in args.python or args.python.startswith(".") else args.python

    print(f"Output dir: {out_dir}")
    print(f"Fichiers sélectionnés: {len(files)}")
    runs = []

    for order, xlsx in enumerate(files, start=1):
        run_dir = out_dir / "runs" / f"{order:02d}_{xlsx.stem}"
        cmd = [python, str(predict_script), "--xlsx", str(xlsx), "--out_dir", str(run_dir),
               "--batch-size", str(args.batch_size)]
        # --template et --mode-threshold ne sont supportés que par emotyc_predict_details.py
        if predict_script.name == "emotyc_predict_details.py":
            cmd.extend(["--template", args.template,
                        "--mode-threshold", str(args.mode_threshold)])
        if args.use_context:
            cmd.append("--use-context")

        print(f"\n[{order}/{len(files)}] {xlsx.name}")
        print(shlex.join(cmd))

        run = {"order": order, "input_file": str(xlsx), "output_dir": str(run_dir),
               "command": cmd, "returncode": None, "status": "dry_run"}
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "command.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
            completed = subprocess.run(cmd, check=False)
            run["returncode"] = completed.returncode
            run["status"] = "ok" if completed.returncode == 0 else "failed"
            if completed.returncode and args.stop_on_error:
                runs.append(run)
                break
        runs.append(run)

    if args.dry_run:
        print("\nDry-run: aucune inférence lancée, aucun fichier écrit.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "orchestrator_manifest.json").write_text(
        json.dumps({"files": [str(p) for p in files], "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.merge:
        merge_outputs(out_dir, runs)
        print(f"\nSorties fusionnées: {out_dir / 'merged'}")
    print(f"Manifest: {out_dir / 'orchestrator_manifest.json'}")


if __name__ == "__main__":
    main()
