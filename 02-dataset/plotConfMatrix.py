import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

cache_dir = Path(tempfile.gettempdir()) / "gesture_confusion_matrix_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

IMG_SIZE = 64
COLOR_CHANNELS = 3
DEFAULT_LABEL_NAMES = ["like", "rock", "peace", "no_gesture"]
OUTPUT = "confusion_matrix_samuel_scheer.png"
MODEL = "gesture_recognition.keras"
JSON = "samuel_scheer_images/annot-samuel_scheer.json"


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Predict gesture labels from an annotation JSON and plot a confusion matrix."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / MODEL,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=script_dir / JSON,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABEL_NAMES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / OUTPUT,
    )
    return parser.parse_args()


def bbox_center_to_crop(image, bbox):
    """Convert normalized center-x/center-y/width/height bbox to a clamped crop."""
    image_height, image_width = image.shape[:2]
    x_center, y_center, width, height = bbox

    x1 = int(round((x_center - width / 2) * image_width))
    y1 = int(round((y_center - height / 2) * image_height))
    x2 = int(round((x_center + width / 2) * image_width))
    y2 = int(round((y_center + height / 2) * image_height))

    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clamping: {bbox}")

    return image[y1:y2, x1:x2]


def preprocess_image(image):
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return resized.astype("float32") / 255.0


def image_path_for_id(images_dir, image_id):
    for extension in (".png", ".jpg", ".jpeg"):
        path = images_dir / f"{image_id}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for '{image_id}' in {images_dir}")


def load_annotated_images(annotations_path, images_dir, label_names):
    with annotations_path.open() as f:
        annotations = json.load(f)

    images = []
    true_labels = []
    sample_names = []

    for image_id, annotation in annotations.items():
        image_path = image_path_for_id(images_dir, image_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        for hand_index, bbox in enumerate(annotation["bboxes"]):
            label = annotation["labels"][hand_index]
            if label not in label_names:
                raise ValueError(
                    f"Label '{label}' is not in --labels {label_names}. "
                    "Pass the model's training labels in output order."
                )

            crop = bbox_center_to_crop(image, bbox)
            images.append(preprocess_image(crop))
            true_labels.append(label_names.index(label))
            sample_names.append(f"{image_path.name} hand {hand_index + 1}")

    return np.array(images), np.array(true_labels), sample_names


def predict(model, images):
    output_classes = model.output_shape[-1]
    batch_size = model.input_shape[0] or len(images)

    if output_classes is not None and output_classes <= 0:
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")

    if batch_size and len(images) < batch_size:
        padding = np.repeat(images[-1:], batch_size - len(images), axis=0)
        padded_images = np.concatenate([images, padding], axis=0)
        return model.predict(padded_images, verbose=0)[: len(images)]

    return model.predict(images, verbose=0)


def main():
    args = parse_args()
    images_dir = args.images_dir or args.annotations.parent

    if len(args.labels) == 0:
        raise ValueError("At least one label name is required.")

    model = load_model(args.model)
    output_classes = model.output_shape[-1]
    if output_classes != len(args.labels):
        raise ValueError(
            f"The model outputs {output_classes} classes, but --labels has "
            f"{len(args.labels)} entries: {args.labels}"
        )

    X_test, y_test, sample_names = load_annotated_images(
        args.annotations, images_dir, args.labels
    )

    probabilities = predict(model, X_test)
    y_predictions = np.argmax(probabilities, axis=1)

    print("Labels:", args.labels)
    print("Probabilities:")
    for sample_name, true_index, predicted_index, row in zip(
        sample_names, y_test, y_predictions, probabilities
    ):
        true_label = args.labels[true_index]
        predicted_label = args.labels[predicted_index]
        confidence = row[predicted_index]
        rounded = np.round(row, 4).tolist()
        print(
            f"  {sample_name}: true={true_label}, predicted={predicted_label}, "
            f"confidence={confidence:.4f}, probabilities={rounded}"
        )

    print("Predicted label indices:", y_predictions.tolist())

    conf_matrix = confusion_matrix(
        y_test, y_predictions, labels=list(range(len(args.labels)))
    )
    print("Confusion matrix:")
    print(conf_matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(conf_matrix, display_labels=args.labels).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved confusion matrix plot to: {args.output}")


if __name__ == "__main__":
    main()
