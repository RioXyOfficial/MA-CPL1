import math
import random

TRAIN = [
    (4.8, 1.0, 1), (4.2, 0.5, 1), (3.9, 1.1, 1), (3.5, 0.8, 1),
    (2.9, 1.0, 1), (4.5, 2.0, 1), (3.8, 2.5, 1), (2.5, 0.4, 1),
    (0.5, 4.5, 0), (1.0, 4.2, 0), (0.8, 3.9, 0), (1.5, 3.5, 0),
    (1.2, 2.9, 0), (2.0, 4.5, 0), (2.5, 3.8, 0), (0.4, 2.5, 0),
    (4.6, 0.5, 0),
    (4.0, 1.0, 0),
    (3.7, 1.2, 0),
    (0.6, 4.7, 1),
    (1.0, 4.4, 1),
    (1.5, 3.9, 1),
]

VAL = [
    (4.9, 1.0, 1), (4.1, 0.7, 1), (3.6, 1.2, 1), (3.2, 0.9, 1),
    (4.4, 2.1, 1), (2.8, 1.5, 1), (3.9, 0.8, 1), (4.7, 2.0, 1),
    (3.3, 1.1, 1), (4.0, 1.9, 1),
    (0.7, 4.8, 0), (1.2, 4.1, 0), (0.9, 3.6, 0), (1.1, 3.2, 0),
    (2.0, 4.4, 0), (1.4, 2.8, 0), (0.8, 3.5, 0), (1.6, 4.0, 0),
    (2.2, 3.7, 0), (0.5, 2.6, 0),
]

ETA = 0.01
EPOCHS = 300

def sigmoid(z: float) -> float:
    """Retourne 1 / (1 + exp(-z))"""
    if z < -700:
        return 0
    return 1.0 / (1.0 + math.exp(-z))

def forward(x1: float, x2: float, w1: float, w2: float, b: float) -> float:
    """Retourne p = sigmoid(w1*x1 + w2*x2 + b)"""
    z = w1 * x1 + w2 * x2 + b
    return sigmoid(z)

def mse_loss(y: int, p: float) -> float:
    """
    Loss MSE pour classification binaire.
    L = 1/2 * (y - p)^2
    """
    return 0.5 * (y - p)**2

def predict(x1: float, x2: float, w1: float, w2: float, b: float) -> int:
    p = forward(x1, x2, w1, w2, b)
    return 1 if p >= 0.5 else 0

def accuracy(dataset, w1: float, w2: float, b: float) -> float:
    """Calcule le pourcentage de bonnes réponses sur un dataset donné"""
    correct = 0
    total = len(dataset)
    for x1, x2, y in dataset:
        pred = predict(x1, x2, w1, w2, b)
        if pred == y:
            correct += 1
    return correct / total

def train():
    random.seed(42)
    w1 = random.uniform(-0.5, 0.5)
    w2 = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)

    print(f"{'Epoch':^5} | {'Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        
        random.shuffle(TRAIN)

        for x1, x2, y in TRAIN:
            p = forward(x1, x2, w1, w2, b)
            
            total_loss += mse_loss(y, p)
            
            dz = (p - y) * p * (1 - p)
            
            dw1 = dz * x1
            dw2 = dz * x2
            db = dz

            w1 -= ETA * dw1
            w2 -= ETA * dw2
            b -= ETA * db

        train_acc = accuracy(TRAIN, w1, w2, b)
        val_acc = accuracy(VAL, w1, w2, b)

        if epoch % 25 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"{epoch:5d} | "
                f"{total_loss:10.4f} | "
                f"{train_acc*100:9.2f}% | "
                f"{val_acc*100:9.2f}%"
            )

    print("\nParamètres finaux:")
    print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")

if __name__ == "__main__":
    train()
