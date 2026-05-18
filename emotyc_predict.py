#!/usr/bin/env python3
"""
Inférence EMOTYC minimale — 19 labels traités uniformément.

Charge le modèle EMOTYC, applique les prédictions sur chaque ligne
du gold label, calcule des métriques globales agrégées, et exporte
un unique fichier emotyc_predictions_summary.json.
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

from emotyc_config import (EMOTYC_LABEL2ID, ALL_LABELS)

def load_model():
    """Charge le modèle EMOTYC et le tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = (
        AutoModelForSequenceClassification
        .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
        .to(device)
        .eval()
    )
    print(f"Modèle EMOTYC chargé")
    return tokenizer, model, device


def format_input(tokenizer, sentence, prev_sentence=None, next_sentence=None,
                 use_context=False):
    """Formate l'input en template bca_spaced (espace après 'current:')."""
    eos = tokenizer.eos_token
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current: {sentence}{eos}after:{nxt}{eos}"
    return f"before:{eos}current: {sentence}{eos}after:{eos}"


@torch.inference_mode()
def predict_batch(tokenizer, model, device, texts, batch_size=32):
    """Inférence par batch. Retourne une matrice (N, 19) de probas sigmoid."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        encodings = tokenizer(
            texts[i:i + batch_size],
            return_tensors="pt", truncation=True,
            padding=True, max_length=512, add_special_tokens=False,
        ).to(device)
        probs = torch.sigmoid(model(**encodings).logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

def load_gold(xlsx_path):
    """Charge le gold. Exige TEXT + les 19 colonnes EMOTYC."""
    df = pd.read_excel(xlsx_path)
    if "TEXT" not in df.columns:
        sys.exit("ERREUR : colonne 'TEXT' absente.")
    missing = [l for l in ALL_LABELS if l not in df.columns]
    if missing:
        sys.exit(f"ERREUR : colonnes EMOTYC manquantes ({len(missing)}/19) : {missing}")

    # Matrice binaire (N, 19)
    gold = np.zeros((len(df), 19), dtype=int)
    for j, col in enumerate(ALL_LABELS):
        gold[:, j] = (pd.to_numeric(df[col], errors="coerce").fillna(0) >= 0.5).astype(int)

    return df["TEXT"].astype(str).tolist(), gold


def compute_metrics(gold, pred):
    """Calcule les métriques par label et globales sur les 19 labels."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        cohen_kappa_score,
    )

    per_label = []
    for j, label in enumerate(ALL_LABELS):
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
        per_label.append({
            "label": label,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4) if not math.isnan(kappa) else None,
            "f1": round(f1_score(g, p, zero_division=0), 4),
            "precision": round(precision_score(g, p, zero_division=0), 4),
            "recall": round(recall_score(g, p, zero_division=0), 4),
            "prevalence_gold": round(g.sum() / len(g), 4),
            "prevalence_pred": round(p.sum() / len(p), 4),
        })

    macro_f1 = np.mean([r["f1"] for r in per_label])
    micro_f1 = f1_score(gold.ravel(), pred.ravel(), zero_division=0)

    global_metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "exact_match": round(float(np.all(gold == pred, axis=1).mean()), 4),
        "n_samples": len(gold),
        "n_labels": 19,
    }
    return per_label, global_metrics


def print_metrics(per_label, global_metrics):
    """Affiche un tableau de métriques."""
    print(f"\n{'—' * 75}")
    print(f"  MÉTRIQUES — 19 LABELS EMOTYC  (seuil: {0.06})")
    print(f"{'—' * 75}")
    print(f"  {'Label':<20s} {'Acc':>7s} {'Kappa':>7s} {'F1':>7s} "
          f"{'Prec':>7s} {'Recall':>7s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-' * 68}")
    for r in per_label:
        k = f"{r['kappa']:.3f}" if r['kappa'] is not None else "  N/A"
        print(f"  {r['label']:<20s} {r['accuracy']:>7.3f} {k:>7s} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['fp']:>5d} {r['fn']:>5d}")
    print(f"  {'-' * 68}")
    print(f"  Macro-F1    : {global_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1    : {global_metrics['micro_f1']:.4f}")
    print(f"  Exact Match : {global_metrics['exact_match']:.4f}")
    print(f"{'—' * 75}")

def main():
    p = argparse.ArgumentParser(description="Inférence EMOTYC — 19 labels, métriques globales")
    p.add_argument("--xlsx", required=True, help="Fichier gold label (.xlsx)")
    p.add_argument("--out_dir", required=True, help="Dossier de sortie")
    p.add_argument("--use-context", action="store_true", help="Utiliser les phrases voisines comme contexte")
    p.add_argument("--batch-size", type=int, default=32, help="Taille du batch (défaut: 32)")
    args = p.parse_args()

    # 1. Gold
    xlsx_path = os.path.abspath(args.xlsx)
    sentences, gold = load_gold(xlsx_path)
    N = len(sentences)

    # 2. Modèle
    tokenizer, model, device = load_model()

    # 3. Inputs formatés (bca_spaced)
    use_ctx = args.use_context
    texts = [
        format_input(tokenizer, sentences[i],
                     sentences[i - 1] if (i > 0 and use_ctx) else None,
                     sentences[i + 1] if (i < N - 1 and use_ctx) else None,
                     use_ctx)
        for i in range(N)
    ]
    ctx_tag = "context" if use_ctx else "no_context"
    # 4. Inférence
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    probs = predict_batch(tokenizer, model, device, texts, batch_size=args.batch_size)
    pred = (probs >= 0.06).astype(int)
    print(f"Inférence terminée — shape: {probs.shape}")

    # 5. Métriques
    per_label, global_metrics = compute_metrics(gold, pred)
    print_metrics(per_label, global_metrics)

    # 6. Export résumé JSON uniquement
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "source_xlsx": os.path.basename(xlsx_path),
        "n_samples": N,
        "template": f"bca_spaced_{ctx_tag}",
        "threshold": 0.06,
        "per_label": per_label,
        "global_metrics": global_metrics,
    }
    out = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nRésumé exporté : {out}")


if __name__ == "__main__":
    main()