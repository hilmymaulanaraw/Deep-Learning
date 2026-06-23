import tensorflow as tf

from tensorflow.keras import optimizers

from config import (
    LEARNING_RATE,
    EPOCHS
)


def compile_model(model):

    # Konfigurasi optimizer, loss function, dan metrik evaluasi
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            )
        ]
    )

    return model


def train_model(
        model,
        train_ds,
        val_ds,
        epochs=EPOCHS):

    # Proses pelatihan model menggunakan data training
    # dan validasi menggunakan data validation/test
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history