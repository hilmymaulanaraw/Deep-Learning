import tensorflow as tf
import cv2
import numpy as np

from config import IMG_SIZE
from config import BATCH_SIZE


def remove_artifacts_and_extract_roi(image):
    # Konversi citra RGB menjadi grayscale
    gray = cv2.cvtColor(
        np.uint8(image * 255),
        cv2.COLOR_RGB2GRAY
    )
    # Threshold untuk memisahkan objek utama dari background
    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )
    # Mencari contour pada citra biner
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Jika tidak ditemukan contour, kembalikan citra asli
    if not contours:
        return image
    # Mengambil contour terbesar sebagai area payudara
    largest_cnt = max(
        contours,
        key=cv2.contourArea
    )
    # Membuat mask ROI
    mask = np.zeros_like(gray)

    cv2.drawContours(
        mask,
        [largest_cnt],
        -1,
        255,
        -1
    )
    # Menghilangkan area di luar ROI
    result = cv2.bitwise_and(
        np.uint8(image * 255),
        np.uint8(image * 255),
        mask=mask
    )

    # Mendapatkan bounding box ROI
    x, y, w, h = cv2.boundingRect(
        largest_cnt
    )

    roi = result[
        y:y+h,
        x:x+w
    ]

    # Menyamakan ukuran citra
    roi_resized = cv2.resize(
        roi,
        IMG_SIZE
    )

    # Normalisasi piksel ke rentang 0–1
    return (
        np.float32(
            roi_resized
        ) / 255.0
    )


def preprocess_pipeline(image):

    image_roi = (
        remove_artifacts_and_extract_roi(
            image
        )
    )

    img_uint8 = np.uint8(
        image_roi * 255
    )

    denoised = cv2.medianBlur(
        img_uint8,
        3
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    channels = cv2.split(
        denoised
    )

    clahe_channels = [
        clahe.apply(c)
        for c in channels
    ]

    result = cv2.merge(
        clahe_channels
    )

    return (
        np.float32(result)
        / 255.0
    )


def load_and_preprocess_image(
        path,
        label):

    # Membaca file gambar
    img = tf.io.read_file(path)

    # Decode JPG menjadi tensor RGB
    img = tf.image.decode_jpeg(
        img,
        channels=3
    )

    # Resize sesuai ukuran input model
    img = tf.image.resize(
        img,
        IMG_SIZE
    )

    # Normalisasi awal
    img = img / 255.0

        # Menjalankan preprocessing OpenCV
    [img] = tf.py_function(
        preprocess_pipeline,
        [img],
        [tf.float32]
    )

    # Menentukan bentuk tensor secara eksplisit
    img.set_shape(
        (*IMG_SIZE, 3)
    )

    return img, label


def create_dataset(
        df,
        shuffle=True):

    # Mengambil path gambar
    paths = df[
        "full_path"
    ].values

    # Mengambil label
    labels = df[
        "label"
    ].values

    # Membuat TensorFlow Dataset
    ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (paths, labels)
        )
    )

    # Load dan preprocessing otomatis
    ds = ds.map(
        load_and_preprocess_image,
        num_parallel_calls=
        tf.data.AUTOTUNE
    )

    # Mengacak urutan data saat training
    if shuffle:
        ds = ds.shuffle(
            buffer_size=1000
        )

    # Membagi data menjadi batch
    ds = ds.batch(
        BATCH_SIZE
    )

    # Optimasi loading data
    ds = ds.prefetch(
        buffer_size=
        tf.data.AUTOTUNE
    )

    return ds