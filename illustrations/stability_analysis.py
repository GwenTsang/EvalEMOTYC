#!/usr/bin/env python3
"""
stability_analysis.py — Analyse de la stabilité des performances d'EMOTYC
sur les corpus évalués (CyberAggAdo, TextToKids, etc.).

Parse tous les fichiers `emotyc_predictions_summary.json` dans les
sous-dossiers de results/<corpus>, extrait macro_f1 et micro_f1,
et affiche des statistiques descriptives (moyenne, écart-type, min, max)
groupées par condition : « Context » vs « NoContext ».
"""

from __future__ import annotations

import json
import statistics
import argparse
from pathlib import Path


def collect_metrics(results_dir: Path) -> dict[str, list[dict]]:
    """
    Parcourt tous les sous-dossiers de `results_dir` et collecte les
    métriques globales depuis chaque emotyc_predictions_summary.json.

    Retourne un dictionnaire { "Context": [...], "NoContext": [...] }
    où chaque entrée contient le nom du dossier, macro_f1 et micro_f1.
    """
    groups: dict[str, list[dict]] = {"Context": [], "NoContext": []}

    for summary_path in sorted(results_dir.glob("*/emotyc_predictions_summary.json")):
        folder_name = summary_path.parent.name

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        global_metrics = data.get("global_metrics", {})
        macro_f1 = global_metrics.get("macro_f1")
        micro_f1 = global_metrics.get("micro_f1")

        if macro_f1 is None or micro_f1 is None:
            print(f"  ⚠ Métriques manquantes dans {summary_path}, ignoré.")
            continue

        entry = {
            "folder": folder_name,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }

        if folder_name.startswith("NoContext"):
            groups["NoContext"].append(entry)
        else:
            groups["Context"].append(entry)

    return groups


def compute_stats(values: list[float]) -> dict[str, float]:
    """Calcule moyenne, écart-type, min, max, étendue (range)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "range": 0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def format_table_row(label: str, stats: dict[str, float]) -> str:
    """Formate une ligne du tableau récapitulatif."""
    return (
        f"  {label:<12s}  "
        f"{stats['n']:>3d}   "
        f"{stats['mean']:.4f}   "
        f"{stats['std']:.4f}   "
        f"{stats['min']:.4f}   "
        f"{stats['max']:.4f}   "
        f"{stats['range']:.4f}"
    )


def print_header():
    print(
        f"  {'Métrique':<12s}  "
        f"{'N':>3s}   "
        f"{'Moy.':>6s}   "
        f"{'É.-T.':>6s}   "
        f"{'Min':>6s}   "
        f"{'Max':>6s}   "
        f"{'Étendue':>7s}"
    )
    print("  " + "─" * 68)


def print_group_report(group_name: str, entries: list[dict]):
    """Affiche le rapport détaillé pour un groupe (Context ou NoContext)."""

    macro_vals = [e["macro_f1"] for e in entries]
    micro_vals = [e["micro_f1"] for e in entries]

    macro_stats = compute_stats(macro_vals)
    micro_stats = compute_stats(micro_vals)

    print(f"\n{'═' * 72}")
    print(f"  {group_name.upper()} — {len(entries)} configuration(s)")
    print(f"{'═' * 72}")

    # Détail par dossier
    print(f"\n  {'Dossier':<65s}  {'macro_f1':>8s}  {'micro_f1':>8s}")
    print("  " + "─" * 85)
    for e in entries:
        print(f"  {e['folder']:<65s}  {e['macro_f1']:>8.4f}  {e['micro_f1']:>8.4f}")

    # Statistiques descriptives
    print(f"\n  Statistiques descriptives :")
    print_header()
    print(format_table_row("macro_f1", macro_stats))
    print(format_table_row("micro_f1", micro_stats))

    # Interprétation rapide
    print(f"\n  → Variabilité du macro_f1 : écart-type = {macro_stats['std']:.4f}, "
          f"étendue = {macro_stats['range']:.4f}")
    print(f"  → Variabilité du micro_f1 : écart-type = {micro_stats['std']:.4f}, "
          f"étendue = {micro_stats['range']:.4f}")

    return macro_stats, micro_stats


def analyze_corpus(corpus_name: str, corpus_dir: Path):
    """Analyse complète d'un corpus : collecte, rapport par groupe, comparaison."""

    print(f"\n{'▓' * 72}")
    print(f"  📊 CORPUS : {corpus_name}")
    print(f"     {corpus_dir}")
    print(f"{'▓' * 72}")

    groups = collect_metrics(corpus_dir)

    all_stats = {}
    for group_name in ("Context", "NoContext"):
        entries = groups[group_name]
        if not entries:
            print(f"\n⚠ Aucune configuration trouvée pour le groupe « {group_name} ».")
            continue
        macro_s, micro_s = print_group_report(group_name, entries)
        all_stats[group_name] = {"macro_f1": macro_s, "micro_f1": micro_s}

    # Comparaison Context vs NoContext
    if "Context" in all_stats and "NoContext" in all_stats:
        print(f"\n{'═' * 72}")
        print(f"  COMPARAISON CONTEXT vs NOCONTEXT")
        print(f"{'═' * 72}")

        for metric in ("macro_f1", "micro_f1"):
            ctx = all_stats["Context"][metric]
            no_ctx = all_stats["NoContext"][metric]
            delta = ctx["mean"] - no_ctx["mean"]
            sign = "+" if delta >= 0 else ""
            print(
                f"\n  {metric}:"
                f"\n    Context   → moy. = {ctx['mean']:.4f}  (σ = {ctx['std']:.4f})"
                f"\n    NoContext → moy. = {no_ctx['mean']:.4f}  (σ = {no_ctx['std']:.4f})"
                f"\n    Δ (Context − NoContext) = {sign}{delta:.4f}"
            )

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyse de la stabilité des performances EMOTYC sur tous les corpus."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Chemin vers le dossier results/ contenant les sous-dossiers par corpus (défaut: auto-détecté).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Dossier introuvable : {results_dir}")
        return

    # Découvrir tous les corpus (sous-dossiers contenant au moins un JSON de résultats)
    corpus_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and list(d.glob("*/emotyc_predictions_summary.json"))
    ])

    if not corpus_dirs:
        print(f"Aucun corpus trouvé dans {results_dir}")
        return

    print(f"\n Analyse de stabilité — EMOTYC")
    print(f"   Dossier racine : {results_dir}")
    print(f"   Corpus détectés : {', '.join(d.name for d in corpus_dirs)}")

    for corpus_dir in corpus_dirs:
        analyze_corpus(corpus_dir.name, corpus_dir)

    print("Analyse terminée.")

if __name__ == "__main__":
    main()
