import os
import pandas as pd

def load_metadata(base_path):

    csv_dir = os.path.join(base_path, "csv")

    calc_train = pd.read_csv(
        os.path.join(
            csv_dir,
            "calc_case_description_train_set.csv"
        )
    )

    calc_test = pd.read_csv(
        os.path.join(
            csv_dir,
            "calc_case_description_test_set.csv"
        )
    )

    mass_train = pd.read_csv(
        os.path.join(
            csv_dir,
            "mass_case_description_train_set.csv"
        )
    )

    mass_test = pd.read_csv(
        os.path.join(
            csv_dir,
            "mass_case_description_test_set.csv"
        )
    )

    train_df = pd.concat(
        [calc_train, mass_train],
        ignore_index=True
    )

    test_df = pd.concat(
        [calc_test, mass_test],
        ignore_index=True
    )

    return train_df, test_df


def map_images_to_metadata(df, base_path):

    def get_potential_uids(p):
        if isinstance(p, str):
            return set(p.split('/'))
        return set()

    jpeg_dir = os.path.join(
        base_path,
        "jpeg"
    )

    uid_map = {}

    for root, dirs, files in os.walk(jpeg_dir):

        for f in files:

            if f.endswith(".jpg"):

                uid_map[
                    os.path.basename(root)
                ] = os.path.join(root, f)

    def match_path(p):

        uids = get_potential_uids(p)

        for uid in uids:

            if uid in uid_map:

                return uid_map[uid]

        return None

    df = df.copy()

    df["full_path"] = (
        df["image file path"]
        .apply(match_path)
    )
    # Membuat label biner
    df["label"] = (
        df["pathology"]
        .apply(
            lambda x:
            1 if "MALIGNANT" in str(x) # 1 = MALIGNANT
            else 0 # 0 = BENIGN
        )
    )

    # Menghapus data yang tidak memiliki file gambar
    cleaned_df = (
        df
        .dropna(subset=["full_path"])
        .copy()
    )

    return cleaned_df