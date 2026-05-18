#!/usr/bin/env python3
"""
Convertit emotyc_predictions_summary.json (format normalisé) en rapport HTML.

Conçu exclusivement pour le format normalisé produit par :
  - emotyc_predict.py (19 labels, per_label)
  - emotyc_predict_details.py (19 labels, per_label)
  - convert_emotyc_legacy_to_new.py (conversion depuis l'ancien format)

Usage basique (table unique, 19 labels) :
    python json_summary_to_html.py \\
        --json emotyc_predictions_summary.json \\
        --out emotyc_predictions_summary.html

Usage avec sous-tables par groupe sémantique :
    python json_summary_to_html.py \\
        --json emotyc_predictions_summary.json \\
        --out emotyc_predictions_summary.html \\
        --groups
"""

import argparse
import json
from html import escape

from emotyc_config import LABEL_GROUPS, GROUP_DISPLAY_NAMES


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATAGE
# ═══════════════════════════════════════════════════════════════════════════

def fmt(value):
    """Formate proprement les valeurs pour HTML."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def html_table(headers, rows, caption=None):
    """Construit une table HTML simple avec balises <table>."""
    parts = []
    parts.append("<table>")
    if caption:
        parts.append(f"  <caption>{escape(caption)}</caption>")
    parts.append("  <thead>")
    parts.append("    <tr>")
    for h in headers:
        parts.append(f"      <th>{escape(str(h))}</th>")
    parts.append("    </tr>")
    parts.append("  </thead>")
    parts.append("  <tbody>")

    for row in rows:
        parts.append("    <tr>")
        for cell in row:
            parts.append(f"      <td>{fmt(cell)}</td>")
        parts.append("    </tr>")

    parts.append("  </tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def html_kv_table(pairs, caption=None):
    """Table clé-valeur à 2 colonnes."""
    return html_table(["Champ", "Valeur"], pairs, caption=caption)


# ═══════════════════════════════════════════════════════════════════════════
#  RECONSTRUCTION DES GROUPES
# ═══════════════════════════════════════════════════════════════════════════

def _labels_in_group(group_key):
    """Retourne l'ensemble des labels appartenant à un groupe."""
    return set(LABEL_GROUPS.get(group_key, []))


def _compute_group_metrics(rows):
    """Calcule macro-F1 et micro-F1 à partir d'un sous-ensemble de per_label."""
    if not rows:
        return {}

    f1_values = [r.get("f1") for r in rows if r.get("f1") is not None]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else None

    tp = sum(int(r.get("tp", 0) or 0) for r in rows)
    fp = sum(int(r.get("fp", 0) or 0) for r in rows)
    fn = sum(int(r.get("fn", 0) or 0) for r in rows)
    denom = 2 * tp + fp + fn
    micro_f1 = (2 * tp) / denom if denom > 0 else None

    result = {}
    if macro_f1 is not None:
        result["Macro-F1"] = round(macro_f1, 4)
    if micro_f1 is not None:
        result["Micro-F1"] = round(micro_f1, 4)
    result["Nombre de labels"] = len(rows)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DU HTML
# ═══════════════════════════════════════════════════════════════════════════

PER_LABEL_HEADERS = [
    "Label",
    "TP", "FP", "FN", "TN",
    "Accuracy", "Kappa",
    "F1", "Precision", "Recall",
    "Prévalence gold", "Prévalence pred",
]


def per_label_row(r):
    """Extrait les colonnes d'une entrée per_label en liste ordonnée."""
    return [
        r.get("label"),
        r.get("tp"), r.get("fp"), r.get("fn"), r.get("tn"),
        r.get("accuracy"), r.get("kappa"),
        r.get("f1"), r.get("precision"), r.get("recall"),
        r.get("prevalence_gold"), r.get("prevalence_pred"),
    ]


def global_metrics_rows(gm):
    """Construit les lignes pour un bloc de métriques globales."""
    rows = []
    for key, label in [
        ("macro_f1", "Macro-F1"),
        ("micro_f1", "Micro-F1"),
        ("exact_match", "Exact Match"),
        ("n_samples", "Nombre d'exemples"),
        ("n_labels", "Nombre de labels"),
    ]:
        val = gm.get(key)
        if val is not None:
            rows.append([label, val])
    return rows


def build_html(summary, use_groups=False):
    """Construit le HTML complet depuis un résumé normalisé."""

    source_xlsx = summary.get("source_xlsx", "")
    n_samples = summary.get("n_samples", "")
    template = summary.get("template", "")
    threshold = summary.get("threshold", "")
    per_label = summary.get("per_label", [])
    global_metrics = summary.get("global_metrics", {})
    legacy = summary.get("legacy_metadata", {})

    # ── Informations générales ────────────────────────────────────────
    metadata_rows = [
        ["Fichier source", source_xlsx],
        ["Nombre d'exemples", n_samples],
        ["Template", template],
        ["Seuil", threshold],
    ]
    if legacy.get("mode_threshold") is not None:
        metadata_rows.append(["Seuil modes", legacy["mode_threshold"]])
    if legacy.get("n_divergent_rows") is not None:
        metadata_rows.append(["Lignes divergentes", legacy["n_divergent_rows"]])

    metadata_html = html_kv_table(metadata_rows)

    # ── Métriques globales ────────────────────────────────────────────
    global_html = ""
    if global_metrics:
        rows = global_metrics_rows(global_metrics)
        if rows:
            global_html = html_kv_table(rows)
    if not global_html:
        global_html = "<p>Aucune métrique globale disponible.</p>"

    # ── Métriques par label ───────────────────────────────────────────
    per_label_html_parts = []

    if use_groups:
        # Construire un index label → entrée pour le filtrage par groupe
        label_index = {r.get("label"): r for r in per_label}
        matched_labels = set()

        for group_key, group_name in GROUP_DISPLAY_NAMES.items():
            group_label_names = _labels_in_group(group_key)
            group_rows = [
                label_index[name]
                for name in LABEL_GROUPS.get(group_key, [])
                if name in label_index
            ]
            if not group_rows:
                continue

            matched_labels.update(r.get("label") for r in group_rows)

            # Métriques agrégées du groupe
            group_gm = _compute_group_metrics(group_rows)
            group_gm_html = ""
            if group_gm:
                gm_rows = [[k, v] for k, v in group_gm.items()]
                group_gm_html = html_kv_table(gm_rows) + "\n"

            table_rows = [per_label_row(r) for r in group_rows]
            per_label_html_parts.append(
                f"<h3>{escape(group_name)}</h3>\n"
                + group_gm_html
                + html_table(PER_LABEL_HEADERS, table_rows)
            )

        # Labels non classés dans aucun groupe
        ungrouped = [r for r in per_label if r.get("label") not in matched_labels]
        if ungrouped:
            table_rows = [per_label_row(r) for r in ungrouped]
            per_label_html_parts.append(
                "<h3>Autres</h3>\n"
                + html_table(PER_LABEL_HEADERS, table_rows)
            )
    else:
        # Mode par défaut : une seule table plate
        table_rows = [per_label_row(r) for r in per_label]
        per_label_html_parts.append(
            html_table(PER_LABEL_HEADERS, table_rows)
        )

    per_label_html = "\n\n".join(per_label_html_parts)

    # ── Assemblage ────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Résumé EMOTYC — {escape(str(source_xlsx))}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #222;
      background: #fff;
    }}

    h1, h2, h3 {{
      font-weight: 600;
    }}

    h1 {{
      border-bottom: 2px solid #ccc;
      padding-bottom: 8px;
    }}

    h3 {{
      color: #555;
      margin-top: 24px;
    }}

    table {{
      border-collapse: collapse;
      margin-bottom: 24px;
      width: 100%;
      max-width: 1200px;
    }}

    caption {{
      text-align: left;
      font-weight: 600;
      margin-bottom: 4px;
      color: #444;
    }}

    th, td {{
      border: 1px solid #ccc;
      padding: 6px 10px;
      text-align: left;
      font-size: 14px;
    }}

    th {{
      background: #f2f2f2;
    }}

    td {{
      background: #fff;
    }}
  </style>
</head>
<body>

<h1>Résumé EMOTYC</h1>

<h2>Informations générales</h2>
{metadata_html}

<h2>Métriques globales</h2>
{global_html}

<h2>Métriques par label</h2>
{per_label_html}

</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convertit emotyc_predictions_summary.json en fichier HTML."
    )
    parser.add_argument("--json", required=True,
                        help="Chemin vers le fichier JSON résumé (format normalisé)")
    parser.add_argument("--out", required=True,
                        help="Chemin du fichier HTML de sortie")
    parser.add_argument("--groups", action="store_true",
                        help="Afficher les métriques découpées par groupe sémantique "
                             "(émotions, modes, types, caractère émotionnel)")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    html = build_html(summary, use_groups=args.groups)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    mode = "avec groupes sémantiques" if args.groups else "table unique"
    print(f"HTML exporté ({mode}) : {args.out}")


if __name__ == "__main__":
    main()
