import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------------------------------
# 1) Dataset : lit mail + label, tokenize, renvoie tensors
# -------------------------------------------------
class MailDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.mails = df["mail"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.mails)

    def __getitem__(self, idx):
        text = self.mails[idx]
        label = self.labels[idx]  # 0/1 déjà encodé

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",     # padding fixe pour rester simple
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def label_to_int(x: str) -> int:
    x = str(x).strip()
    if x == "Answer":
        return 1
    if x == "NoAnswer":
        return 0
    raise ValueError(f"Label inconnu: {x} (attendu: Answer ou NoAnswer)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV avec colonnes: mail;label (label=Answer/NoAnswer)")
    parser.add_argument("--model_name", default="camembert-base", help="Modèle de base")
    parser.add_argument("--out_dir", default="./model_mail_classifier", help="Dossier de sortie")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'epochs (80 phrases -> 3 à 8 ok)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (petit dataset -> petit batch)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (fine-tuning)")
    parser.add_argument("--max_length", type=int, default=128, help="Longueur max tokens")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------------------------------
    # 2) Lire CSV ; et vérifier colonnes
    # -------------------------------------------------
    df = pd.read_csv(args.csv, sep=";", encoding="latin-1")
    expected_cols = {"mail", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Colonnes attendues: {expected_cols}. Colonnes trouvées: {list(df.columns)}")

    # Nettoyage minimal transforme des NoAnswer en 0 etc.
    df = df.dropna(subset=["mail", "label"]).copy()
    df["label"] = df["label"].apply(label_to_int)

    print(f"Nombre d'exemples: {len(df)}")
    print("Répartition labels:", df["label"].value_counts().to_dict())

    # -------------------------------------------------
    # 3) Tokenizer + Modèle (CamemBERT + tête)
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    # -------------------------------------------------
    # 4) Dataset + DataLoader
    # -------------------------------------------------
    dataset = MailDataset(df, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # -------------------------------------------------
    # 5) Optimiseur
    # -------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------------------------------------
    # 6) Entraînement (pipeline visible)
    # -------------------------------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # (7) Forward -> logits (+ loss car labels fournis)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # (8) Loss
            total_loss += loss.item()

            # (Option pédagogique) softmax/argmax pour calculer accuracy
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # (9) Backprop
            optimizer.zero_grad()
            loss.backward()

            # (10) Update
            optimizer.step()

        avg_loss = total_loss / len(loader)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Accuracy (train): {acc:.3f}")

    # -------------------------------------------------
    # 7) Sauvegarde
    # -------------------------------------------------
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Modèle sauvegardé dans:", args.out_dir)


if __name__ == "__main__":
    main()

