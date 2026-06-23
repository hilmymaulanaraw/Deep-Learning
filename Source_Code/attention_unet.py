import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
    backend as K
)


def attention_block(
        x,
        gating,
        inter_shape):

    # Mengambil dimensi feature map encoder
    shape_x = K.int_shape(x)

    # Transformasi feature map decoder
    phi_g = layers.Conv2D(
        inter_shape,
        (1, 1),
        padding="same"
    )(gating)

    # Transformasi feature map encoder
    theta_x = layers.Conv2D(
        inter_shape,
        (1, 1),
        strides=(1, 1),
        padding="same"
    )(x)

    # Menggabungkan informasi encoder dan decoder
    concat_xg = layers.add(
        [
            theta_x,
            phi_g
        ]
    )

    # Aktivasi ReLU
    act_xg = layers.Activation(
        "relu"
    )(concat_xg)

    # Menghasilkan attention coefficient
    psi_xg = layers.Conv2D(
        1,
        (1, 1),
        padding="same"
    )(act_xg)

    # Mengubah nilai attention menjadi 0-1
    sigmoid_xg = (
        layers.Activation(
            "sigmoid"
        )(psi_xg)
    )

    # Mengalikan attention map dengan feature map encoder
    y = layers.multiply(
        [
            sigmoid_xg,
            x
        ]
    )

    # Menyesuaikan jumlah channel output
    result = layers.Conv2D(
        shape_x[3],
        (1, 1),
        padding="same"
    )(y)

    # Batch Normalization untuk stabilitas training
    result_bn = (
        layers.BatchNormalization()
        (result)
    )

    return result_bn


def conv_block(
        x,
        filters):

    x = layers.Conv2D(
        filters,
        (3,3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.Conv2D(
        filters,
        (3,3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    return x


def build_attention_unet_classifier(
        input_shape=(224,224,3)):

    inputs = layers.Input(
        input_shape
    )

    c1 = conv_block(
        inputs,
        64
    )

    p1 = layers.MaxPooling2D(
        (2,2)
    )(c1)

    c2 = conv_block(
        p1,
        128
    )

    p2 = layers.MaxPooling2D(
        (2,2)
    )(c2)

    c3 = conv_block(
        p2,
        256
    )

    p3 = layers.MaxPooling2D(
        (2,2)
    )(c3)

    c4 = conv_block(
        p3,
        512
    )

    p4 = layers.MaxPooling2D(
        (2,2)
    )(c4)

    b1 = conv_block(
        p4,
        1024
    )

    u1 = layers.Conv2DTranspose(
        512,
        (2,2),
        strides=(2,2),
        padding="same"
    )(b1)

    att1 = attention_block(
        c4,
        u1,
        512
    )

    c5 = conv_block(
        layers.concatenate(
            [u1, att1]
        ),
        512
    )

    u2 = layers.Conv2DTranspose(
        256,
        (2,2),
        strides=(2,2),
        padding="same"
    )(c5)

    att2 = attention_block(
        c3,
        u2,
        256
    )

    c6 = conv_block(
        layers.concatenate(
            [u2, att2]
        ),
        256
    )

    u3 = layers.Conv2DTranspose(
        128,
        (2,2),
        strides=(2,2),
        padding="same"
    )(c6)

    att3 = attention_block(
        c2,
        u3,
        128
    )

    c7 = conv_block(
        layers.concatenate(
            [u3, att3]
        ),
        128
    )

    u4 = layers.Conv2DTranspose(
        64,
        (2,2),
        strides=(2,2),
        padding="same"
    )(c7)

    att4 = attention_block(
        c1,
        u4,
        64
    )

    c8 = conv_block(
        layers.concatenate(
            [u4, att4]
        ),
        64
    )

    last_conv = layers.Conv2D(
        64,
        (3,3),
        padding="same",
        activation="relu",
        name="last_conv_layer"
    )(c8)

    gap = layers.GlobalAveragePooling2D()(
        last_conv
    )

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="final_output"
    )(gap)

    model = models.Model(
        inputs=[inputs],
        outputs=[output],
        name=
        "Attention_UNet_Classifier"
    )

    return model


def dice_loss(
        y_true,
        y_pred):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    numerator = (
        2 *
        tf.reduce_sum(
            y_true * y_pred
        )
    )

    denominator = tf.reduce_sum(
        y_true + y_pred
    )

    return 1 - (
        numerator + 1e-6
    ) / (
        denominator + 1e-6
    )