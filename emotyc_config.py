"""
Configuration partagée EMOTYC — source de vérité unique.

Centralise les labels, groupes sémantiques, seuils et tables
de normalisation utilisés par tous les scripts du repo.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  LABELS (noms canoniques, sans accents, ordre du modèle)
# ═══════════════════════════════════════════════════════════════════════════

EMOTYC_LABEL2ID = {
    "Emo": 0, "Comportementale": 1, "Designee": 2, "Montree": 3,
    "Suggeree": 4, "Base": 5, "Complexe": 6, "Admiration": 7,
    "Autre": 8, "Colere": 9, "Culpabilite": 10, "Degout": 11,
    "Embarras": 12, "Fierte": 13, "Jalousie": 14, "Joie": 15,
    "Peur": 16, "Surprise": 17, "Tristesse": 18,
}

ALL_LABELS = list(EMOTYC_LABEL2ID.keys())  # 19 labels, model order


# ═══════════════════════════════════════════════════════════════════════════
#  GROUPES SÉMANTIQUES
# ═══════════════════════════════════════════════════════════════════════════

META_LABELS = ["Emo"]

EMOTION_LABELS = [
    "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
    "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
    "Surprise", "Tristesse",
]

MODE_LABELS = ["Comportementale", "Designee", "Montree", "Suggeree"]

TYPE_LABELS = ["Base", "Complexe"]

# Dictionnaire de groupes pour itération programmatique
LABEL_GROUPS = {
    "emo":     META_LABELS,
    "emotion": EMOTION_LABELS,
    "mode":    MODE_LABELS,
    "type":    TYPE_LABELS,
}

# Mapping inverse label → groupe
LABEL_TO_GROUP = {
    label: group
    for group, labels in LABEL_GROUPS.items()
    for label in labels
}


# ═══════════════════════════════════════════════════════════════════════════
#  SOUS-CATÉGORIES D'ÉMOTIONS
# ═══════════════════════════════════════════════════════════════════════════

BASIC_EMOTIONS = ["Colere", "Degout", "Joie", "Peur", "Surprise", "Tristesse"]
COMPLEX_EMOTIONS = ["Admiration", "Culpabilite", "Embarras", "Fierte", "Jalousie", "Autre"]


# ═══════════════════════════════════════════════════════════════════════════
#  SEUILS PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_THRESHOLD = 0.5
DEFAULT_MODE_THRESHOLD = 0.06


# ═══════════════════════════════════════════════════════════════════════════
#  MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "TextToKids/CamemBERT-base-EmoTextToKids"
TOKENIZER_NAME = "camembert-base"


# ═══════════════════════════════════════════════════════════════════════════
#  NOMS D'AFFICHAGE (pour le HTML / rapports)
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_NAMES = {
    "Colere": "Colère",
    "Culpabilite": "Culpabilité",
    "Degout": "Dégoût",
    "Fierte": "Fierté",
    "Designee": "Désignée",
    "Montree": "Montrée",
    "Suggeree": "Suggérée",
    "Emo": "Émo",
}

GROUP_DISPLAY_NAMES = {
    "emo":     "Caractère émotionnel (Émo)",
    "emotion": "Émotions",
    "mode":    "Modes d'expression",
    "type":    "Types (Base / Complexe)",
}


# ═══════════════════════════════════════════════════════════════════════════
#  NORMALISATION LABELS ACCENTUÉS → CANONIQUES
# ═══════════════════════════════════════════════════════════════════════════

LABEL_NORMALIZATION = {v: k for k, v in DISPLAY_NAMES.items()}
# Ajout des fautes de frappe ou variantes courantes
LABEL_NORMALIZATION["Désignee"] = "Designee"
