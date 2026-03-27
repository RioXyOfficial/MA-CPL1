import math

W1 = 1.3
W2 = -0.7
B = -0.2

DATA = [
    (1.0, 5.0, 0), (5.0, 1.0, 1), (0.0, 5.0, 0),
    (3.0, 2.0, 1), (1.0, 4.0, 0), (5.0, 0.0, 1),
    (2.0, 4.0, 0), (3.0, 1.0, 1), (1.0, 3.0, 0),
    (4.0, 1.0, 1)
]

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def predict_proba(x1: float, x2: float) -> float:
    z = W1 * x1 + W2 * x2 + B
    return sigmoid(z)

def predict(x1: float, x2: float) -> int:
    return 1 if predict_proba(x1, x2) >= 0.5 else 0

def main():
    correct = 0
    print("x1\tx2\tz\t\tp\t\ty_pred\ty_true")
    
    for x1, x2, y_true in DATA:
        z = W1 * x1 + W2 * x2 + B
        p = predict_proba(x1, x2)
        y_pred = predict(x1, x2)
        
        if y_pred == y_true:
            correct += 1
            
        print(f"{x1}\t{x2}\t{z:.2f}\t{p:.4f}\t{y_pred}\t{y_true}")
    
    accuracy = correct / len(DATA)
    print(f"\nAccuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()