#!/usr/bin/env python3
"""
Convertit emotyc_predictions_summary.json en rapport HTML sobre.

Usage :
    python json_summary_to_html.py \
        --json emotyc_predictions_summary.json \
        --out emotyc_predictions_summary.html
"""

import argparse
import json
from html import escape


def fmt(value):
    """Formate proprement les valeurs pour HTML."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def html_table(headers, rows):
    """Construit une table HTML simple."""
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    for h in headers:
        html.append(f"      <th>{escape(str(h))}</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in rows:
        html.append("    <tr>")
        for cell in row:
            html.append(f"      <td>{fmt(cell)}</td>")
        html.append("    </tr>")

    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)


def build_html(summary):
    source_xlsx = summary.get("source_xlsx", "")
    n_samples = summary.get("n_samples", "")
    template = summary.get("template", "")
    threshold = summary.get("threshold", "")
    global_metrics = summary.get("global_metrics", {})
    per_label = summary.get("per_label", [])

    metadata_rows = [
        ["Fichier source", source_xlsx],
        ["Nombre d'exemples", n_samples],
        ["Template", template],
        ["Seuil", threshold],
    ]

    global_rows = [
        ["Macro-F1", global_metrics.get("macro_f1")],
        ["Micro-F1", global_metrics.get("micro_f1")],
        ["Exact match", global_metrics.get("exact_match")],
        ["Nombre d'exemples", global_metrics.get("n_samples")],
        ["Nombre de labels", global_metrics.get("n_labels")],
    ]

    per_label_headers = [
        "Label",
        "TP",
        "FP",
        "FN",
        "TN",
        "Accuracy",
        "Kappa",
        "F1",
        "Precision",
        "Recall",
        "Prévalence gold",
        "Prévalence prédite",
    ]

    per_label_rows = []
    for r in per_label:
        per_label_rows.append([
            r.get("label"),
            r.get("tp"),
            r.get("fp"),
            r.get("fn"),
            r.get("tn"),
            r.get("accuracy"),
            r.get("kappa"),
            r.get("f1"),
            r.get("precision"),
            r.get("recall"),
            r.get("prevalence_gold"),
            r.get("prevalence_pred"),
        ])

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Résumé EMOTYC</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #222;
      background: #fff;
    }}

    h1, h2 {{
      font-weight: 600;
    }}

    table {{
      border-collapse: collapse;
      margin-bottom: 32px;
      width: 100%;
      max-width: 1200px;
    }}

    th, td {{
      border: 1px solid #ccc;
      padding: 8px 10px;
      text-align: left;
      font-size: 14px;
    }}

    th {{
      background: #f2f2f2;
    }}

    td {{
      background: #fff;
    }}

    .number {{
      text-align: right;
    }}
  </style>
</head>
<body>

<h1>Résumé EMOTYC</h1>

<h2>Informations générales</h2>
{html_table(["Champ", "Valeur"], metadata_rows)}

<h2>Métriques globales</h2>
{html_table(["Métrique", "Valeur"], global_rows)}

<h2>Métriques par label</h2>
{html_table(per_label_headers, per_label_rows)}

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Convertit emotyc_predictions_summary.json en fichier HTML."
    )
    parser.add_argument("--json", required=True, help="Chemin vers le fichier JSON résumé")
    parser.add_argument("--out", required=True, help="Chemin du fichier HTML de sortie")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    html = build_html(summary)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML exporté : {args.out}")


if __name__ == "__main__":
    main()
