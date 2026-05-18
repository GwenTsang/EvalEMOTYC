#!/usr/bin/env python3
"""
Inférence EMOTYC locale et comparaison au gold label.

Charge le modèle EMOTYC (TextToKids/CamemBERT-base-EmoTextToKids),
applique les prédictions sur chaque ligne du gold label, compare
avec les annotations humaines, et exporte un JSONL de résultats.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from emotyc_config import (
    EMOTYC_LABEL2ID, ALL_LABELS,
    EMOTION_LABELS, MODE_LABELS, TYPE_LABELS,
    DEFAULT_THRESHOLD, DEFAULT_MODE_THRESHOLD,
    MODEL_NAME, TOKENIZER_NAME,
)

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES DÉRIVÉES
# ═══════════════════════════════════════════════════════════════════════════

EMO_LABEL = "Emo"

# Indices dans le vecteur de 19 logits
EMOTION_INDICES = [EMOTYC_LABEL2ID[e] for e in EMOTION_LABELS]
MODE_INDICES = [EMOTYC_LABEL2ID[m] for m in MODE_LABELS]
TYPE_INDICES = [EMOTYC_LABEL2ID[t] for t in TYPE_LABELS]
EMO_INDEX = EMOTYC_LABEL2ID[EMO_LABEL]


# ═══════════════════════════════════════════════════════════════════════════
#  MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Charge le modèle EMOTYC et le tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
        .to(device)
        .eval()
    )
    print(f"Modèle EMOTYC chargé sur {device}")
    print(f"  {model.config.num_labels} labels, type={model.config.problem_type}")
    return tokenizer, model, device


def format_input(tokenizer, sentence, prev_sentence=None, next_sentence=None,
                 use_context=False, template="bca"):
    """
    Formate l'input selon le template bca.

    template='bca'        : before:{prev}</s>current:{s}</s>after:{next}</s>
    template='bca_spaced' : before:{prev}</s>current: {s}</s>after:{next}</s>

    IMPORTANT : même avec template='bca_spaced', il n'y a jamais d'espace
    après 'before:' ni après 'after:'. L'espace ne concerne que 'current:'.
    """
    eos = tokenizer.eos_token
    current_sep = " " if template == "bca_spaced" else ""
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current:{current_sep}{sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current:{current_sep}{sentence}{eos}after:{eos}"


@torch.inference_mode()
def predict_batch(tokenizer, model, device, texts, batch_size=16):
    """Inférence par batch. Retourne une matrice (N, 19) de probas sigmoid."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=False,   # aligné avec le fine-tuning d'EMOTYC
        ).to(device)
        logits = model(**encodings).logits  # (B, 19)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# ═══════════════════════════════════════════════════════════════════════════
#  GOLD LABELS
# ═══════════════════════════════════════════════════════════════════════════

def load_gold(xlsx_path):
    """
    Charge le gold label. Exige strictement les 19 colonnes EMOTYC + TEXT.
    Retourne (df, sentences).
    """
    df = pd.read_excel(xlsx_path)
    print(f"Gold labels : {len(df)} lignes chargées depuis {os.path.basename(xlsx_path)}")

    # Validation stricte : TEXT + 19 labels EMOTYC
    if "TEXT" not in df.columns:
        print("ERREUR : colonne 'TEXT' absente.")
        sys.exit(1)

    missing = [l for l in ALL_LABELS if l not in df.columns]
    if missing:
        print(f"ERREUR : colonnes EMOTYC manquantes ({len(missing)}/19) : {missing}. Fichier ignoré.")
        sys.exit(1)
    return df, df["TEXT"].astype(str).tolist()


def extract_gold_matrix(df, labels):
    """Extrait la matrice binaire (N, K) du gold pour les colonnes données."""
    gold = np.zeros((len(df), len(labels)), dtype=int)
    for j, col in enumerate(labels):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        gold[:, j] = (vals >= 0.5).astype(int)
    return gold


# ═══════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(gold, pred, label_names):
    """
    Calcule les métriques par label et globales.

    Arguments :
        gold        — matrice (N, K) binaire gold
        pred        — matrice (N, K) binaire prédictions
        label_names — liste de K noms de labels
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        cohen_kappa_score,
    )

    results = []
    for j, label in enumerate(label_names):
        g, p = gold[:, j], pred[:, j]
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum())

        acc = accuracy_score(g, p)
        try:
            kappa = cohen_kappa_score(g, p, labels=[0, 1])
        except Exception:
            kappa = float("nan")
        f1 = f1_score(g, p, zero_division=0)
        prec = precision_score(g, p, zero_division=0)
        rec = recall_score(g, p, zero_division=0)

        results.append({
            "label": label,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4) if not math.isnan(kappa) else None,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "prevalence_gold": round(g.sum() / len(g), 4),
            "prevalence_pred": round(p.sum() / len(p), 4),
        })

    macro_f1 = np.mean([r["f1"] for r in results])
    micro_f1 = f1_score(gold.ravel(), pred.ravel(), zero_division=0)
    exact_match = np.all(gold == pred, axis=1).mean()

    return results, {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "exact_match": round(float(exact_match), 4),
        "n_samples": len(gold),
        "n_labels": len(label_names),
    }


def _print_metrics_table(title, per_label, global_metrics, threshold_info=None):
    """Affiche un tableau de métriques formaté."""
    t_info = f"  (seuil: {threshold_info})" if threshold_info else ""
    print(f"\n{'—' * 75}")
    print(f"  {title}{t_info}")
    print(f"{'—' * 75}")
    print(f"  {'Label':<20s} {'Acc':>7s} {'Kappa':>7s} {'F1':>7s} "
          f"{'Prec':>7s} {'Recall':>7s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-' * 68}")
    for r in per_label:
        k_str = f"{r['kappa']:.3f}" if r['kappa'] is not None else "  N/A"
        print(f"  {r['label']:<20s} {r['accuracy']:>7.3f} {k_str:>7s} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['fp']:>5d} {r['fn']:>5d}")
    print(f"  {'-' * 68}")
    print(f"  Macro-F1    : {global_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1    : {global_metrics['micro_f1']:.4f}")
    print(f"  Exact Match : {global_metrics['exact_match']:.4f}")
    print(f"{'—' * 75}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--xlsx", required=True,
                   help="Chemin vers le fichier gold label (.xlsx)")

    p.add_argument("--out_dir", required=True,
                   help="Dossier de sortie pour les résultats")

    p.add_argument("--use-context", action="store_true",
                   help="Utiliser les phrases voisines (i-1, i+1) comme contexte")

    p.add_argument("--template", choices=["bca", "bca_spaced"], default="bca",
                   help="Format du template d'input. "
                        "'bca' = before:{prev}</s>current:{s}</s>after:{next}</s> (défaut), "
                        "'bca_spaced' = before:{prev}</s>current: {s}</s>after:{next}</s> "
                        "(espace uniquement après current:)")

    p.add_argument("--batch-size", type=int, default=32,
                   help="Taille du batch pour l'inférence (défaut: 32)")

    p.add_argument("--mode-threshold", type=float, default=DEFAULT_MODE_THRESHOLD,
                   help=f"Seuil pour les prédictions des modes d'expression (défaut: {DEFAULT_MODE_THRESHOLD})")

    return p.parse_args()


def main():
    args = parse_args()
    EMOTION_THRESHOLD = DEFAULT_THRESHOLD

    # ── 1. Chargement du gold ─────────────────────────────────────────
    xlsx_path = os.path.abspath(args.xlsx)
    df, sentences = load_gold(xlsx_path)
    N = len(sentences)

    # Gold matrices par groupe
    gold_emotion = extract_gold_matrix(df, EMOTION_LABELS)
    gold_mode = extract_gold_matrix(df, MODE_LABELS)
    gold_emo = extract_gold_matrix(df, [EMO_LABEL])
    gold_type = extract_gold_matrix(df, TYPE_LABELS)

    # ── 2. Chargement du modèle ───────────────────────────────────────
    tokenizer, model, device = load_model()

    # ── 3. Préparation des inputs ─────────────────────────────────────
    use_context = args.use_context
    formatted_texts = [
        format_input(
            tokenizer, sentences[i],
            sentences[i - 1] if (i > 0 and use_context) else None,
            sentences[i + 1] if (i < N - 1 and use_context) else None,
            use_context,
            template=args.template,
        )
        for i in range(N)
    ]

    ctx_suffix = "_context" if use_context else "_no_context"
    template_name = f"{args.template}{ctx_suffix}"
    print(f"Template : {template_name}")
    print(f"Exemple  : {formatted_texts[0][:120]}…")

    # ── 4. Inférence ──────────────────────────────────────────────────
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    all_probs = predict_batch(
        tokenizer, model, device, formatted_texts,
        batch_size=args.batch_size,
    )
    print(f"Inférence terminée — shape: {all_probs.shape}")

    # ── 5. Extraction des probas par groupe ───────────────────────────
    emotion_probs = all_probs[:, EMOTION_INDICES]
    mode_probs = all_probs[:, MODE_INDICES]
    emo_probs = all_probs[:, EMO_INDEX]
    type_probs = all_probs[:, TYPE_INDICES]

    # ── 6. Prédictions binaires ───────────────────────────────────────
    print(f"▸ Seuil émotions : {EMOTION_THRESHOLD}")
    print(f"▸ Seuil modes : {args.mode_threshold}")
    pred_emotion = (emotion_probs >= EMOTION_THRESHOLD).astype(int)
    pred_mode = (mode_probs >= args.mode_threshold).astype(int)
    pred_emo = (emo_probs >= 0.5).astype(int)
    pred_type = (type_probs >= 0.5).astype(int)

    # ── 7. Métriques (affichage terminal par groupe) ──────────────────
    per_emotion, global_emotion = compute_metrics(gold_emotion, pred_emotion, EMOTION_LABELS)
    _print_metrics_table("MÉTRIQUES PAR ÉMOTION", per_emotion, global_emotion,
                         threshold_info=f"{EMOTION_THRESHOLD}")

    per_mode, global_mode = compute_metrics(gold_mode, pred_mode, MODE_LABELS)
    _print_metrics_table("MÉTRIQUES PAR MODE D'EXPRESSION", per_mode, global_mode,
                         threshold_info=f"{args.mode_threshold}")

    per_emo, global_emo = compute_metrics(gold_emo, pred_emo.reshape(-1, 1), [EMO_LABEL])
    _print_metrics_table("MÉTRIQUES — CARACTÈRE ÉMOTIONNEL (Emo)", per_emo, global_emo)

    per_type, global_type = compute_metrics(gold_type, pred_type, TYPE_LABELS)
    _print_metrics_table("MÉTRIQUES — TYPE (Base/Complexe)", per_type, global_type)

    # ── 8. Export XLSX et JSONL ────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # XLSX des prédictions
    export_dict = {"TEXT": sentences}
    for j, emo in enumerate(EMOTION_LABELS):
        export_dict[emo] = pred_emotion[:, j]
    for j, mode in enumerate(MODE_LABELS):
        export_dict[mode] = pred_mode[:, j]
    out_xlsx = os.path.join(args.out_dir, "emotyc_predictions_output.xlsx")
    pd.DataFrame(export_dict).to_excel(out_xlsx, index=False)
    print(f"Prédictions exportées en XLSX : {out_xlsx}")

    # JSONL détaillé
    out_jsonl = os.path.join(args.out_dir, "emotyc_predictions.jsonl")
    n_divergent = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(N):
            # Divergences émotions + modes
            divergences = []
            for labels, gold_m, pred_m, probs_m, threshold, dim in [
                (EMOTION_LABELS, gold_emotion, pred_emotion, emotion_probs, EMOTION_THRESHOLD, "emotion"),
                (MODE_LABELS, gold_mode, pred_mode, mode_probs, args.mode_threshold, "mode"),
            ]:
                for j, label in enumerate(labels):
                    g, p = int(gold_m[i, j]), int(pred_m[i, j])
                    if g != p:
                        divergences.append({
                            "dimension": dim, "label": label,
                            "gold": g, "pred": p,
                            "proba": round(float(probs_m[i, j]), 6),
                            "seuil": threshold,
                            "type_divergence": "faux_positif" if p == 1 else "faux_negatif",
                        })

            if divergences:
                n_divergent += 1

            id_val = df.iloc[i].get("ID", i)
            record = {
                "idx": i,
                "id": str(id_val) if id_val is not None and not (isinstance(id_val, float) and math.isnan(id_val)) else str(i),
                "text": sentences[i],
                "text_prev": sentences[i - 1] if i > 0 else None,
                "text_next": sentences[i + 1] if i < N - 1 else None,
                "template_used": template_name,
                "emotion_threshold": EMOTION_THRESHOLD,
                "mode_threshold": args.mode_threshold,
                # Émotions
                "probas": {e: round(float(emotion_probs[i, j]), 6) for j, e in enumerate(EMOTION_LABELS)},
                "preds": {e: int(pred_emotion[i, j]) for j, e in enumerate(EMOTION_LABELS)},
                "golds": {e: int(gold_emotion[i, j]) for j, e in enumerate(EMOTION_LABELS)},
                # Modes
                "probas_mode": {m: round(float(mode_probs[i, j]), 6) for j, m in enumerate(MODE_LABELS)},
                "preds_mode": {m: int(pred_mode[i, j]) for j, m in enumerate(MODE_LABELS)},
                "golds_mode": {m: int(gold_mode[i, j]) for j, m in enumerate(MODE_LABELS)},
                # Caractère émotionnel
                "proba_emo": round(float(emo_probs[i]), 6),
                "pred_emo": int(pred_emo[i]),
                "gold_emo": int(gold_emo[i, 0]),
                # Type (Base/Complexe)
                "probas_type": {t: round(float(type_probs[i, j]), 6) for j, t in enumerate(TYPE_LABELS)},
                "preds_type": {t: int(pred_type[i, j]) for j, t in enumerate(TYPE_LABELS)},
                "golds_type": {t: int(gold_type[i, j]) for j, t in enumerate(TYPE_LABELS)},
                # Divergences
                "n_divergences": len(divergences),
                "divergences": divergences,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n Résultats exportés : {out_jsonl}")
    print(f"  {N} lignes, {n_divergent} avec ≥1 divergence")

    # ── 9. Export du résumé JSON (format normalisé) ─────────────────
    # Fusion de toutes les métriques per-label en liste plate (19 labels)
    per_label = per_emo + per_emotion + per_mode + per_type

    # Métriques globales sur les 19 labels combinés
    from sklearn.metrics import f1_score as _f1_score
    gold_all = np.hstack([gold_emo, gold_emotion, gold_mode, gold_type])
    pred_all = np.hstack([pred_emo.reshape(-1, 1), pred_emotion, pred_mode, pred_type])
    global_all = {
        "macro_f1": round(float(np.mean([r["f1"] for r in per_label])), 4),
        "micro_f1": round(float(_f1_score(gold_all.ravel(), pred_all.ravel(), zero_division=0)), 4),
        "exact_match": round(float(np.all(gold_all == pred_all, axis=1).mean()), 4),
        "n_samples": N,
        "n_labels": len(per_label),
    }

    summary = {
        "source_xlsx": os.path.basename(xlsx_path),
        "n_samples": N,
        "template": template_name,
        "threshold": EMOTION_THRESHOLD,
        "per_label": per_label,
        "global_metrics": global_all,
    }
    summary_path = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Résumé : {summary_path}")


if __name__ == "__main__":
    main()