import kagglehub

from dataset_loader import (
    load_metadata,
    map_images_to_metadata
)

from preprocessing import (
    create_dataset
)

from augmentasi import (
    apply_augmentation
)

from models import (
    build_densenet121
)

from train_models import (
    compile_model,
    train_model
)

from evaluasi import (
    evaluate_model
)

from threshold_tuning import (
    tune_threshold
)

from visualisasi import (
    plot_history,
    visualize_gradcam
)


path = kagglehub.dataset_download(
    "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
)

print(
    "Path to dataset files:",
    path
)

train_df, test_df = (
    load_metadata(path)
)

train_df_cleaned = (
    map_images_to_metadata(
        train_df,
        path
    )
)

test_df_cleaned = (
    map_images_to_metadata(
        test_df,
        path
    )
)

train_ds = create_dataset(
    train_df_cleaned,
    shuffle=True
)

test_ds = create_dataset(
    test_df_cleaned,
    shuffle=False
)

augmented_train_ds = (
    apply_augmentation(
        train_ds
    )
)

model = (
    build_densenet121()
)

model = compile_model(
    model
)

history = train_model(
    model,
    augmented_train_ds,
    test_ds
)

plot_history(
    history
)

results = evaluate_model(
    model,
    test_ds
)

print(
    "\nSensitivity:",
    results["sensitivity"]
)

print(
    "\nSpecificity:",
    results["specificity"]
)

print(
    "\nClassification Report:\n"
)

print(
    results[
        "classification_report"
    ]
)

best_threshold = (
    tune_threshold(
        model,
        test_ds
    )
)

print(
    "\nBest Threshold:",
    best_threshold
)

visualize_gradcam(
    model,
    test_ds
)