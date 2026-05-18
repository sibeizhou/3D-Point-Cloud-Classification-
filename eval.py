import numpy as np
from collections import Counter

def evaluate_accuracy(y_true, y_pred):
    """
    Computes overall accuracy and per-class accuracy.

    Parameters:
    - y_true: List or NumPy array of ground truth labels
    - y_pred: List or NumPy array of predicted labels

    Returns:
    - overall_accuracy: Overall classification accuracy
    - per_class_accuracy: Dictionary with per-class accuracy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute overall accuracy
    overall_accuracy = np.mean(y_true == y_pred)

    # Compute per-class accuracy
    class_counts = Counter(y_true)  # Count occurrences of each class
    correct_counts = Counter(y for y, p in zip(y_true, y_pred) if y == p)

    per_class_accuracy = {
        cls: correct_counts[cls] / class_counts[cls]
        for cls in class_counts
    }

    return overall_accuracy, per_class_accuracy

# Example usage:
y_true = [0, 1, 2, 0, 1, 2, 2, 1, 0, 2]
y_pred = [0, 1, 2, 1, 1, 2, 0, 1, 0, 2]

overall_acc, per_class_acc = evaluate_accuracy(y_true, y_pred)
print(f"Overall Accuracy: {overall_acc:.2%}")
print("Per-Class Accuracy:", per_class_acc)

