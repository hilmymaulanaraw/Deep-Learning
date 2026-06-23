import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from gradcam import get_gradcam


def plot_history(history):

    # Visualisasi performa model selama proses training
    acc = history.history["accuracy"]

    val_acc = history.history[
        "val_accuracy"
    ]

    auc = history.history["auc"]

    val_auc = history.history[
        "val_auc"
    ]

    epochs_range = range(
        len(acc)
    )

    plt.figure(
        figsize=(15,5)
    )

    plt.subplot(1,2,1)

    plt.plot(
        epochs_range,
        acc,
        label=
        "Training Accuracy"
    )

    plt.plot(
        epochs_range,
        val_acc,
        label=
        "Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(
        epochs_range,
        auc,
        label=
        "Training AUC"
    )

    plt.plot(
        epochs_range,
        val_auc,
        label=
        "Validation AUC"
    )

    plt.title(
        "Training and Validation AUC"
    )

    plt.legend()

    plt.show()


def visualize_gradcam(
        model,
        dataset):

    # Mengambil satu sampel citra untuk interpretasi model
    for images, labels in dataset.take(1):

        sample_img = images[0:1]

        sample_label = labels[0]

        break

    # Membuat heatmap Grad-CAM
    heatmap = get_gradcam(
        sample_img,
        model
    )

    img_np = sample_img[
        0
    ].numpy()

    plt.figure(
        figsize=(12,6)
    )

    plt.subplot(1,2,1)

    plt.imshow(
        img_np
    )

    plt.title(
        f"Original Image "
        f"(Label: "
        f"{'Malignant' if sample_label==1 else 'Benign'})"
    )

    plt.axis("off")

    plt.subplot(1,2,2)

    plt.imshow(
        img_np
    )

    # Menyesuaikan ukuran heatmap dengan ukuran citra asli
    resized_heatmap = (
        tf.image.resize(
            heatmap[
                ...,
                np.newaxis
            ],
            (224,224)
        ).numpy()
    )

    # Overlay heatmap pada citra mammogram
    plt.imshow(
        resized_heatmap,
        alpha=0.5,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis("off")

    plt.show()


def visualize_attention_unet_internals(
        model,
        dataset):

    # Mengambil satu sampel untuk visualisasi internal model
    for images, labels in dataset.take(1):

        img = images[0:1]

        label = labels[0]

        break

    layer_outputs = []

    layer_names = []

    # Mengambil feature map dan attention map dari layer tertentu
    for layer in model.layers:

        if len(
            layer.output.shape
        ) == 4:

            if (
                "conv2d_7"
                in layer.name
                or
                "last_conv_layer"
                in layer.name
            ):

                layer_outputs.append(
                    layer.output
                )

                layer_names.append(
                    layer.name
                )

            if (
                "activation"
                in layer.name
                and
                "sigmoid"
                in layer.name
                and
                "final"
                not in layer.name
            ):

                layer_outputs.append(
                    layer.output
                )

                layer_names.append(
                    "Attention Gate"
                )

    if not layer_outputs:

        print(
            "Tidak ditemukan layer yang cocok untuk visualisasi."
        )

        return

    vis_model = (
        tf.keras.models.Model(
            inputs=model.input,
            outputs=layer_outputs
        )
    )

    activations = (
        vis_model.predict(img)
    )

    if not isinstance(
        activations,
        list
    ):

        activations = [
            activations
        ]

    plt.figure(
        figsize=(24,6)
    )

    # Menampilkan citra asli
    plt.subplot(1,4,1)

    plt.imshow(
        img[0].numpy()
    )

    plt.title(
        f"Original Image\n"
        f"Label: "
        f"{'Malignant' if label==1 else 'Benign'}"
    )

    plt.axis("off")

    # Visualisasi feature map hasil ekstraksi fitur
    plt.subplot(1,4,2)

    feat_map = (
        activations[0][0]
    )

    if len(
        feat_map.shape
    ) == 3:

        feat_map = np.mean(
            feat_map,
            axis=-1
        )

    plt.imshow(
        feat_map,
        cmap="viridis"
    )

    plt.title(
        f"Feature Map: "
        f"{layer_names[0]}"
    )

    plt.axis("off")

    # Visualisasi attention map dari Attention Gate
    plt.subplot(1,4,3)

    attn_map = (
        activations[-1][0]
    )

    if len(
        attn_map.shape
    ) == 3:

        attn_map = np.mean(
            attn_map,
            axis=-1
        )

    plt.imshow(
        attn_map,
        cmap="hot"
    )

    plt.title(
        "Attention Gate Map"
    )

    plt.axis("off")

    # Visualisasi area fokus model menggunakan Grad-CAM
    plt.subplot(1,4,4)

    heatmap = get_gradcam(
        img,
        model
    )

    resized_heatmap = (
        tf.image.resize(
            heatmap[
                ...,
                np.newaxis
            ],
            (224,224)
        ).numpy()
    )

    plt.imshow(
        img[0].numpy()
    )

    plt.imshow(
        resized_heatmap,
        alpha=0.5,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Focus"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()