# tokenizer.py – Version élève (squelette)

import re
import sys

# -----------------------------------------------------------
# 1) MODE 1 : découpe naïve par espaces
# -----------------------------------------------------------

def tokenize_whitespace(text):
    return text.split()


# -----------------------------------------------------------
# 2) MODE 2 : découpe avec regex
# -----------------------------------------------------------

# Regex fournie (simplifiée)
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+|[.,;:!?()\"]")

def tokenize_regex(text):
    return TOKEN_RE.findall(text)


# -----------------------------------------------------------
# 3) Construction du vocabulaire avec IDs
# -----------------------------------------------------------

def build_vocab(tokens):
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def display_vocab(vocab):
    print("\n--- VOCABULAIRE ---")
    print(f"{'ID':<5} | {'TOKEN':<15}")
    print("-" * 25)
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"{idx:<5} | {token:<15}")
    print(f"\nTotal: {len(vocab)} tokens uniques")


# -----------------------------------------------------------
# 4) Interface en ligne de commande (CLI)
# -----------------------------------------------------------

# Usage attendu :
# python tokenizer.py whitespace "Bonjour le monde !"
# python tokenizer.py regex "J'aime les maths !"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tokenizer.py <mode> <texte>")
        print("Modes: whitespace, regex")
        sys.exit(1)
    
    mode = sys.argv[1]
    text = " ".join(sys.argv[2:])
    
    if mode == "whitespace":
        tokens = tokenize_whitespace(text)
    elif mode == "regex":
        tokens = tokenize_regex(text)
    else:
        print(f"Mode inconnu: {mode}")
        sys.exit(1)
    
    print("--- TOKENS ---")
    for i, token in enumerate(tokens):
        print(f"{i}: {token}")
    
    vocab = build_vocab(tokens)
    display_vocab(vocab)