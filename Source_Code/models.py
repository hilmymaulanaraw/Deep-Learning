import tensorflow as tf

from tensorflow.keras import (
    layers,
    Model
)

from tensorflow.keras.applications import (
    DenseNet121,
    ResNet50,
    EfficientNetB0
)


def build_densenet121():

    # Memuat model DenseNet121 pretrained ImageNet
    # tanpa layer klasifikasi bawaan
    base_model = DenseNet121(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze bobot pretrained
    # agar hanya classifier yang dilatih
    base_model.trainable = False

    x = base_model.output

    # Mengubah feature map menjadi vektor fitur
    x = layers.GlobalAveragePooling2D()(x)

    # Menstabilkan distribusi fitur
    x = layers.BatchNormalization()(x)

    # Fully Connected Layer
    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    # Mengurangi overfitting
    x = layers.Dropout(
        0.4
    )(x)

    # Output binary classification
    # 0 = Benign
    # 1 = Malignant
    predictions = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="DenseNet121_Classifier"
    )

    return model

def build_resnet50():

    # Memuat model ResNet50 pretrained ImageNet
    base_model = ResNet50(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze seluruh layer pretrained
    base_model.trainable = False

    x = base_model.output

    # Feature Extraction Layer
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    # Classification Head
    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.4
    )(x)

    # Output binary classification
    predictions = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="ResNet50_Classifier"
    )

    return model

def build_efficientnetb0():

    # Memuat model EfficientNetB0 pretrained ImageNet
    base_model = EfficientNetB0(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze seluruh layer pretrained
    base_model.trainable = False

    x = base_model.output

    # Feature Extraction Layer
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    # Classification Head
    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.4
    )(x)

    # Output binary classification
    predictions = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="EfficientNetB0_Classifier"
    )

    return model