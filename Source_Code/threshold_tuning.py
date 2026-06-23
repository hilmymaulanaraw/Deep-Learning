import numpy as np

from sklearn.metrics import (
    precision_recall_curve
)


def tune_threshold(
        model,
        dataset):

    y_true = []
    y_scores = []

    # Mengumpulkan label asli dan skor probabilitas prediksi
    for imgs, labels in dataset:

        preds = model.predict(
            imgs,
            verbose=0
        )

        y_true.extend(
            labels.numpy()
        )

        y_scores.extend(
            preds.flatten()
        )

    # Menghitung Precision-Recall Curve untuk seluruh threshold
    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_true,
            y_scores
        )
    )

    # Menghitung nilai F1-Score pada setiap threshold
    f1_scores = (
        2 *
        (precisions * recalls)
    ) / (
        precisions +
        recalls +
        1e-10
    )

    # Menentukan indeks threshold dengan F1-Score tertinggi
    best_idx = np.argmax(
        f1_scores
    )

    # Mengambil threshold optimal
    best_threshold = (
        thresholds[
            best_idx
        ]
    )

    print(
        f"Optimal Threshold found: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Best F1-Score: "
        f"{f1_scores[best_idx]:.4f}"
    )

    return best_threshold