#!/usr/bin/env python3
"""
delta_heatmap.py — Heatmap de transferabilité EMOTYC : TextToKids → CyberAggAdo.

Pour chaque label × métrique, calcule le delta absolu (TTK − Cyber)
et colore séquentiellement selon l'amplitude de la différence.
La performance sur TextToKids sert d'étalon.

Usage :
    python delta_heatmap.py
    python delta_heatmap.py --open
"""

from __future__ import annotations

import json
import argparse
import webbrowser
from pathlib import Path

# ── Réutilisation de la config du repo ──────────────────────────────────────
from emotyc_config import (
    ALL_LABELS,
    DISPLAY_NAMES,
)

# ── Chemins par défaut ──────────────────────────────────────────────────────
RESULTS = Path(__file__).parent / "results"
DEFAULT_CYBER = (
    RESULTS / "CyberAggAdo"
    / "NoContextTemplateAvecEspace_RerunCurrentOnly_BS260_20260511_111410"
    / "emotyc_predictions_summary.json"
)
DEFAULT_TTK = (
    RESULTS / "TextToKids"
    / "NoContextTemplateAvecEspaceMode006"
    / "emotyc_predictions_summary.json"
)
DEFAULT_OUT = RESULTS / "heatmap_delta.html"

# Métriques affichées (les plus interprétables)
METRICS = ["f1", "precision", "recall"]
METRIC_DISPLAY = {"f1": "Δ_F1", "precision": "Δ_Précision", "recall": "Δ_Rappel"}


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def display(label: str) -> str:
    """Nom d'affichage avec accents."""
    return DISPLAY_NAMES.get(label, label)


def delta_color(delta: float) -> str:
    """
    Retourne une couleur CSS pour un delta absolu [0, 1].

    Palette séquentielle (même palette que proportion_heatmap) :
      0.0 (aucune différence)   → vert doux  hsl(145, 45%, 88%)
      0.5 (grosse différence)   → ambre      hsl(35,  85%, 65%)
      1.0 (différence maximale) → rouge prof hsl(0,   70%, 42%)

    Interpolation linéaire par morceaux en HSL.
    """
    d = max(0.0, min(1.0, delta))
    # Réutilise la même interpolation : retention = 1 - delta
    r = 1.0 - d

    if r >= 0.5:
        t = 2.0 * (1.0 - r)
        h = 145 + t * (35 - 145)
        s = 45  + t * (85 - 45)
        l = 88  + t * (65 - 88)
    else:
        t = 2.0 * (0.5 - r)
        h = 35 + t * (0 - 35)
        s = 85 + t * (70 - 85)
        l = 65 + t * (42 - 65)
    return f"hsl({h:.0f}, {s:.0f}%, {l:.0f}%)"


def text_color(delta: float) -> str:
    """Texte sombre sauf pour les deltas très élevés (fond foncé)."""
    return "#fff" if delta > 0.75 else "#1a1a1a"


def gold_support(entry: dict) -> int:
    """Nombre d'instances positives dans le gold (TP + FN)."""
    return (entry.get("tp", 0) or 0) + (entry.get("fn", 0) or 0)


def build_heatmap(cyber: dict, ttk: dict, config_name: str) -> str:
    """Construit le HTML de la heatmap de deltas."""

    cyber_by_label = {e["label"]: e for e in cyber["per_label"]}
    ttk_by_label = {e["label"]: e for e in ttk["per_label"]}

    cg = cyber["global_metrics"]
    tg = ttk["global_metrics"]

    # ── Trier les labels par support décroissant dans CyberAggAdo ──────
    sorted_labels = sorted(
        ALL_LABELS,
        key=lambda lb: gold_support(cyber_by_label.get(lb, {})),
        reverse=True,
    )

    # ── Construire les lignes du tableau ────────────────────────────────
    table_body = ""

    for label in sorted_labels:
        t = ttk_by_label.get(label, {})
        c = cyber_by_label.get(label, {})

        support_ttk = gold_support(t)
        support_cyber = gold_support(c)

        cells = f'<td class="label-cell">{display(label)}</td>'

        # Colonnes taille d'échantillon
        cells += f'<td class="sample-cell">{support_ttk:,}</td>'
        cells += f'<td class="sample-cell">{support_cyber:,}</td>'

        for m in METRICS:
            tv = t.get(m, 0) or 0  # TTK (étalon)
            cv = c.get(m, 0) or 0  # Cyber

            delta = tv - cv  # Différence absolue (positif = TTK meilleur)
            abs_delta = abs(delta)
            bg = delta_color(abs_delta)
            fg = text_color(abs_delta)

            # Afficher avec signe si négatif (Cyber meilleur), sans signe sinon
            if delta < 0:
                delta_text = f"{delta:.2f}"
            else:
                delta_text = f"{delta:.2f}"

            cells += (
                f'<td class="heat-cell" style="background:{bg}; color:{fg};">'
                f'{delta_text}'
                f'</td>'
            )

        table_body += f"<tr>{cells}</tr>\n"

    # ── Lignes globales ─────────────────────────────────────────────────
    global_metrics_display = {"macro_f1": "Macro F1", "micro_f1": "Micro F1"}
    n_cols = 1 + 2 + len(METRICS)  # label + 2 sample cols + metrics
    table_body += (
        f'<tr class="group-header">'
        f'<td colspan="{n_cols}">Métriques globales</td>'
        f"</tr>\n"
    )
    for gm, gm_display in global_metrics_display.items():
        tv = tg[gm]
        cv = cg[gm]
        delta = tv - cv
        abs_delta = abs(delta)
        bg = delta_color(abs_delta)
        fg = text_color(abs_delta)
        cells = f'<td class="label-cell">{gm_display}</td>'
        # Sample cols: show total corpus sizes for global metrics
        cells += f'<td class="sample-cell">{tg["n_samples"]:,}</td>'
        cells += f'<td class="sample-cell">{cg["n_samples"]:,}</td>'
        cells += (
            f'<td class="heat-cell" style="background:{bg}; color:{fg};">'
            f'{delta:.4f}'
            f'</td>'
        )
        # Fill remaining metric columns
        remaining = len(METRICS) - 1
        if remaining > 0:
            cells += f'<td colspan="{remaining}" class="empty-cell"></td>'
        table_body += f"<tr>{cells}</tr>\n"

    # ── En-têtes ────────────────────────────────────────────────────────
    metric_headers = ""
    for m in METRICS:
        md = METRIC_DISPLAY[m]
        metric_headers += f'<th class="sub-header">{md}</th>'

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heatmap de transferabilité — EMOTYC</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #f7f7f8;
      --surface: #fff;
      --border: #e2e2e5;
      --text: #1a1a1a;
      --muted: #71717a;
      --accent: #3b3b3f;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: "Inter", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2.5rem 3rem;
      line-height: 1.5;
    }}

    /* ── Header ────────────────────────────────────────────────── */
    h1 {{
      font-size: 1.3rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: .84rem;
      margin-top: .2rem;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin: 1.2rem 0 1.6rem;
      font-size: .8rem;
      color: var(--muted);
    }}
    .meta-item {{
      display: flex;
      align-items: center;
      gap: .4rem;
    }}
    .dot {{
      width: 8px; height: 8px; border-radius: 50%;
      display: inline-block;
    }}
    .dot-ttk {{ background: #6366f1; }}
    .dot-cyber {{ background: #f59e0b; }}

    /* ── Table ──────────────────────────────────────────────────── */
    table {{
      border-collapse: separate;
      border-spacing: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      width: auto;
      font-size: .82rem;
    }}
    th, td {{
      padding: .5rem .7rem;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    thead th {{
      background: #fafafa;
      font-weight: 600;
      text-align: center;
      border-bottom: 2px solid var(--border);
    }}
    thead th:first-child {{
      text-align: left;
    }}
    .sub-header {{
      font-size: .72rem;
      font-weight: 500;
      color: var(--muted);
    }}

    /* Group header rows */
    .group-header td {{
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      font-size: .78rem;
      letter-spacing: .03em;
      text-transform: uppercase;
      padding: .4rem .7rem;
    }}

    /* Label column */
    .label-cell {{
      text-align: left;
      font-weight: 500;
      padding-left: 1rem;
      background: #fdfdfd;
      min-width: 120px;
    }}

    /* Heat cells */
    .heat-cell {{
      text-align: center;
      font-weight: 700;
      font-size: .85rem;
      font-variant-numeric: tabular-nums;
      min-width: 65px;
      letter-spacing: .02em;
    }}


    /* Sample size cells */
    .sample-cell {{
      text-align: center;
      font-variant-numeric: tabular-nums;
      font-size: .78rem;
      color: var(--muted);
      background: #fafafa;
      border-right: 1px solid var(--border);
    }}

    /* N/A cells */
    .na-cell {{
      text-align: center;
      background: #f4f4f5;
    }}
    .na {{
      color: #a1a1aa;
      font-size: .72rem;
      font-style: italic;
    }}

    .empty-cell {{
      background: #f9f9f9;
      border-right: none;
    }}

    /* Last row no border */
    tbody tr:last-child td {{
      border-bottom: none;
    }}

    /* ── Color legend ──────────────────────────────────────────── */
    .legend {{
      margin-top: 1.5rem;
      display: flex;
      align-items: center;
      gap: .6rem;
      font-size: .75rem;
      color: var(--muted);
    }}
    .legend-label {{
      font-weight: 500;
      margin-right: .2rem;
    }}
    .gradient-bar {{
      width: 200px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid var(--border);
    }}
    .legend-note {{
      font-size: .72rem;
      color: var(--muted);
      margin-top: .6rem;
    }}
  </style>
</head>
<body>
  <h1>Transferabilité d'EMOTYC : TextToKids → CyberAggAdo</h1>
  <p class="subtitle">
    Différence absolue par label (Δ = TTK − Cyber) — config. <strong>{config_name}</strong>
  </p>

  <div class="meta">
    <div class="meta-item">
      <span class="dot dot-ttk"></span>
      <strong>TextToKids</strong> (étalon) — {tg['n_samples']:,} unités textuelles
    </div>
    <div class="meta-item">
      <span class="dot dot-cyber"></span>
      <strong>CyberAggAdo</strong> — {cg['n_samples']:,} unités textuelles
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th rowspan="2">Label</th>
        <th colspan="2">Taille échantillon</th>
        <th colspan="{len(METRICS)}">Différence Δ = TTK − Cyber</th>
      </tr>
      <tr>
        <th class="sub-header">TTK</th>
        <th class="sub-header">Cyber</th>
        {metric_headers}
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>

  <div class="legend">
    <span class="legend-label">Δ :</span>
    <span>0.00</span>
    <div class="gradient-bar" style="background: linear-gradient(to right,
      hsl(145, 45%, 88%),
      hsl(90, 55%, 78%),
      hsl(35, 85%, 65%),
      hsl(15, 78%, 52%),
      hsl(0, 70%, 42%)
    );"></div>
    <span>1.00</span>
  </div>
  <p class="legend-note">
    0.00 = performance identique &nbsp;·&nbsp;
    Valeur positive = TTK meilleur &nbsp;·&nbsp;
    Valeur négative = CyberAggAdo meilleur
  </p>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Génère une heatmap HTML de deltas EMOTYC (TTK − CyberAggAdo)."
    )
    parser.add_argument("--cyber", type=Path, default=DEFAULT_CYBER,
                        help="JSON summary CyberAggAdo.")
    parser.add_argument("--ttk", type=Path, default=DEFAULT_TTK,
                        help="JSON summary TextToKids (étalon).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Fichier HTML de sortie.")
    parser.add_argument("--config-name", type=str, default="Configuration Personnalisée",
                        help="Nom de la configuration pour l'affichage du titre.")
    parser.add_argument("--open", action="store_true",
                        help="Ouvre le HTML dans le navigateur.")
    args = parser.parse_args()

    cyber = load_summary(args.cyber)
    ttk = load_summary(args.ttk)

    html = build_heatmap(cyber, ttk, args.config_name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Heatmap générée : {args.out.resolve()}")

    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
