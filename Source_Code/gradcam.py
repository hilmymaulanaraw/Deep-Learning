import tensorflow as tf
import numpy as np


def get_gradcam(
        img_array,
        model,
        last_conv_layer_name=None):

    # Mencari layer konvolusi terakhir secara otomatis
    # jika nama layer tidak diberikan
    if last_conv_layer_name is None:

        for layer in reversed(
                model.layers):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                last_conv_layer_name = (
                    layer.name
                )

                break

    # Membuat model sementara yang mengembalikan:
    # 1. Output feature map layer konvolusi terakhir
    # 2. Output prediksi model
    grad_model = tf.keras.models.Model(
        inputs=[
            model.inputs
        ],
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    # Menghitung gradien prediksi terhadap feature map
    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(img_array)
        )

        loss = tf.reduce_mean(
            predictions
        )

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    # Global Average Pooling pada gradien
    # untuk memperoleh bobot setiap channel feature map
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    # Menggabungkan feature map dengan bobot gradien
    # untuk menghasilkan heatmap Grad-CAM
    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
    )


    heatmap = tf.squeeze(
        heatmap
    )

    # ReLU dan normalisasi heatmap ke rentang 0-1
    heatmap = tf.maximum(
        heatmap,
        0
    ) / (
        tf.reduce_max(
            heatmap
        ) + 1e-10
    )

    return heatmap.numpy()