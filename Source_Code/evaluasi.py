import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


def evaluate_model(
        model,
        test_ds):

    # Prediksi probabilitas pada seluruh data uji
    y_pred_prob = model.predict(
        test_ds
    )

    # Konversi probabilitas menjadi label kelas
    # Threshold default = 0.5
    y_pred = (
        y_pred_prob > 0.5
    ).astype(int).flatten()

    # Mengambil label asli dari dataset uji
    y_true = np.concatenate(
        [
            y
            for x, y
            in test_ds
        ],
        axis=0
    )

    # Membentuk Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # Mengambil nilai:
    # TN = True Negative
    # FP = False Positive
    # FN = False Negative
    # TP = True Positive
    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # Sensitivity (Recall Positif)
    # Mengukur kemampuan model mendeteksi kasus positif
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # Laporan evaluasi lengkap
    # Berisi:
    # Precision
    # Recall
    # F1-Score
    # Support
    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )

    # Menyimpan seluruh hasil evaluasi
    results = {

        "y_true":
        y_true,

        "y_pred":
        y_pred,

        "y_pred_prob":
        y_pred_prob,

        "confusion_matrix":
        cm,

        "sensitivity":
        sensitivity,

        "specificity":
        specificity,

        "classification_report":
        report
    }

    return results