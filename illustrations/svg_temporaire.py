import argparse
import json
from html import escape

HEADERS = ["Label", "F1", "Précision", "Rappel", "FN", "FP", "TN", "TP"]

COL_WIDTHS = [180, 80, 90, 80, 60, 60, 60, 60]
ROW_HEIGHT = 36
RENDER_PAD_X = 34
RENDER_PAD_Y = 36


def fmt(value):
    """Formate les valeurs : 3 chiffres après la virgule pour les floats."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def extract_table_data(summary):
    """Extrait et formate les données depuis le JSON."""
    per_label = summary.get("per_label", [])

    rows = []
    for r in per_label:
        rows.append([
            r.get("label"),
            fmt(r.get("f1")),
            fmt(r.get("precision")),
            fmt(r.get("recall")),
            fmt(r.get("fn")),
            fmt(r.get("fp")),
            fmt(r.get("tn")),
            fmt(r.get("tp"))
        ])
    return rows


def generate_svg(headers, data, out_path):
    """Génère le code SVG de la table et l'écrit dans un fichier."""

    total_width = sum(COL_WIDTHS)
    total_height = ROW_HEIGHT * (len(data) + 1)

    render_width = total_width + RENDER_PAD_X * 2
    render_height = total_height + RENDER_PAD_Y * 2

    offset_x = RENDER_PAD_X
    offset_y = RENDER_PAD_Y

    svg_parts = []

    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{render_width}" height="{render_height}" '
        f'viewBox="0 0 {render_width} {render_height}">'
    )

    svg_parts.append("""
  <style>
    text {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 14px;
      fill: #333;
    }
    .header-text {
      font-weight: 600;
      fill: #111;
    }
    .bg-header {
      fill: #f3f4f6;
    }
    .bg-row-even {
      fill: #ffffff;
    }
    .bg-row-odd {
      fill: #fafafa;
    }
    .border {
      stroke: #d1d5db;
      stroke-width: 1px;
      shape-rendering: crispEdges;
    }
  </style>
""")

    svg_parts.append(f'  <g transform="translate({offset_x}, {offset_y})">')

    all_rows = [headers] + data

    for r_idx, row in enumerate(all_rows):
        y = r_idx * ROW_HEIGHT

        if r_idx == 0:
            bg_class = "bg-header"
        else:
            bg_class = "bg-row-even" if r_idx % 2 == 1 else "bg-row-odd"

        svg_parts.append(
            f'    <rect x="0" y="{y}" width="{total_width}" height="{ROW_HEIGHT}" class="{bg_class}"/>'
        )

        svg_parts.append(
            f'    <line x1="0" y1="{y + ROW_HEIGHT}" x2="{total_width}" y2="{y + ROW_HEIGHT}" class="border"/>'
        )

        current_x = 0

        for c_idx, cell_value in enumerate(row):
            w = COL_WIDTHS[c_idx]

            if c_idx == 0:
                text_anchor = "start"
                text_x = current_x + 12
            else:
                text_anchor = "middle"
                text_x = current_x + (w / 2)

            text_y = y + (ROW_HEIGHT / 2) + 5
            text_class = "header-text" if r_idx == 0 else ""

            svg_parts.append(
                f'    <text x="{text_x}" y="{text_y}" '
                f'text-anchor="{text_anchor}" class="{text_class}">'
                f'{escape(str(cell_value))}</text>'
            )

            current_x += w

    current_x = 0

    for w in COL_WIDTHS[:-1]:
        current_x += w
        svg_parts.append(
            f'    <line x1="{current_x}" y1="0" x2="{current_x}" y2="{total_height}" class="border"/>'
        )

    svg_parts.append(
        f'    <rect x="0" y="0" width="{total_width}" height="{total_height}" '
        f'fill="none" class="border"/>'
    )

    svg_parts.append('  </g>')
    svg_parts.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def main():
    parser = argparse.ArgumentParser(description="Génère un tableau SVG des métriques par label.")
    parser.add_argument("--json", required=True, help="Chemin vers emotyc_predictions_summary.json")
    parser.add_argument("--out", default="./table.svg", help="Chemin du fichier SVG de sortie")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    data = extract_table_data(summary)

    generate_svg(HEADERS, data, args.out)

    print("SVG généré")


if __name__ == "__main__":
    main()