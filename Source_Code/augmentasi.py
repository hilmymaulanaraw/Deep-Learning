import tensorflow as tf

from tensorflow.keras import (
    layers,
    Sequential
)

data_augmentation = Sequential([

    # Membalik gambar secara horizontal dan vertikal
    # untuk mensimulasikan variasi posisi objek
    layers.RandomFlip(
        "horizontal_and_vertical"
    ),

    # Rotasi acak hingga ±20%
    # membantu model mengenali objek dari berbagai orientasi
    layers.RandomRotation(
        0.2
    ),

    # Zoom acak sebesar 10%
    # membantu model lebih toleran terhadap variasi ukuran objek
    layers.RandomZoom(
        0.1
    )

])


def apply_augmentation(train_ds):

    augmented_train_ds = train_ds.map(
        lambda x, y:
        (
            data_augmentation(
                x,
                training=True
            ),
            y
        ),

        # Mengoptimalkan pemrosesan data secara paralel
        num_parallel_calls=
        tf.data.AUTOTUNE
    )

    return augmented_train_ds