import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Partie 2 — Mapping des labels
# =========================
# Le modèle travaille avec des entiers (0/1) car les réseaux de neurones
# ne peuvent manipuler que des valeurs numériques (tenseurs).
# Les logits de sortie sont des scores numériques pour chaque classe,
# il faut donc une correspondance texte <-> entier.
# =========================

def label_to_int(x: str) -> int:
    x = str(x).strip()
    if x == "Answer":
        return 1
    if x == "NoAnswer":
        return 0
    raise ValueError(f"Label inconnu: '{x}' (attendu: 'Answer' ou 'NoAnswer')")


def int_to_label(i: int) -> str:
    if i == 1:
        return "Answer"
    if i == 0:
        return "NoAnswer"
    raise ValueError(f"Entier inconnu: {i} (attendu: 0 ou 1)")


def main():
    # =========================
    # 1. Chargement des arguments
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model_mail_classifier",
                        help="Dossier du modèle fine-tuné")
    parser.add_argument("--csv", default=None,
                        help="CSV validation avec colonnes: mail;label")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--show_errors", type=int, default=10,
                        help="Nombre d'erreurs à afficher")
    # Partie 3 — Argument --text pour classifier une phrase unique
    parser.add_argument("--text", default=None,
                        help="Phrase unique à classifier (pas besoin de CSV)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # 2. Chargement du modèle et du tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()  # mode évaluation (désactive dropout, etc.)

    # =========================
    # Partie 3 — Mode phrase unique (--text)
    # Si --text est fourni, on classifie cette phrase et on s'arrête
    # =========================
    if args.text is not None:
        with torch.no_grad():
            # Tokenization de la phrase
            inputs = tokenizer(
                args.text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Passage dans le modèle -> logits -> softmax -> prédiction
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            y_pred = int(torch.argmax(probs).item())
            score = float(probs[y_pred].item())

        print(f"\nTexte   : {args.text}")
        print(f"Prédiction : {int_to_label(y_pred)}")
        print(f"Score      : {score:.4f}")
        return  # On s'arrête ici, pas besoin du CSV

    # Si pas de --text, on a besoin du CSV
    if args.csv is None:
        raise ValueError("Il faut fournir --csv ou --text")

    # =========================
    # 3. Lecture et préparation du CSV
    # =========================
    df = pd.read_csv(args.csv, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()

    # Accepter "Mail" ou "mail"
    if "mail" not in df.columns and "Mail" in df.columns:
        df = df.rename(columns={"Mail": "mail"})

    if not {"mail", "label"}.issubset(df.columns):
        raise ValueError(f"Colonnes attendues: mail;label. Trouvées: {list(df.columns)}")

    df = df.dropna(subset=["mail", "label"]).copy()
    df["y_true"] = df["label"].apply(label_to_int)

    correct = 0
    total = 0

    # Matrice de confusion : TN, FP, FN, TP
    TN = FP = FN = TP = 0

    errors = []

    # =========================
    # 4. Boucle d'inférence (tokenization + modèle + softmax)
    # =========================
    with torch.no_grad():
        for _, row in df.iterrows():
            text = str(row["mail"])
            y_true = int(row["y_true"])

            # Tokenization : texte -> input_ids + attention_mask
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Modèle : input_ids -> logits (scores bruts)
            logits = model(**inputs).logits  # shape (1, 2)

            # Softmax : logits -> probabilités
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape (2,)

            # Prédiction : argmax des probabilités
            y_pred = int(torch.argmax(probs).item())
            score = float(probs[y_pred].item())

            total += 1
            if y_pred == y_true:
                correct += 1
            else:
                errors.append((text, int_to_label(y_true), int_to_label(y_pred), score))

            # Mise à jour matrice de confusion
            if y_true == 0 and y_pred == 0:
                TN += 1
            elif y_true == 0 and y_pred == 1:
                FP += 1
            elif y_true == 1 and y_pred == 0:
                FN += 1
            elif y_true == 1 and y_pred == 1:
                TP += 1

    # =========================
    # 5. Calcul des métriques
    # =========================

    # Partie 4 — Recall au lieu de l'Accuracy
    # Recall(Answer) = TP / (TP + FN)
    # = parmi tous les vrais Answer, combien le modèle en a trouvé ?
    # Important car rater un mail important (Answer classé NoAnswer) est grave.
    recall_answer = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print(f"\nValidation: {total} exemples")
    print(f"Recall (Answer): {recall_answer:.3f}")

    print("\nMatrice de confusion (lignes=vrai, colonnes=prédit)")
    print("                 Pred NoAnswer    Pred Answer")
    print(f"True NoAnswer        {TN:4d}          {FP:4d}")
    print(f"True Answer          {FN:4d}          {TP:4d}")

    # Afficher quelques erreurs
    if errors:
        print(f"\nExemples d'erreurs (max {args.show_errors}) :")
        for i, (text, yt, yp, score) in enumerate(errors[:args.show_errors], start=1):
            short = (text[:140] + "…") if len(text) > 140 else text
            print(f"{i:02d}. Vrai={yt:<8} | Prédit={yp:<8} | score={score:.3f} | mail='{short}'")
    else:
        print("\nAucune erreur sur ce set.")


if __name__ == "__main__":
    main()