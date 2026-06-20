import json
import os
import cv2
import numpy as np

ANNOT_PATH = "02-dataset/combined_images/annot-combined.json"
IMAGE_FOLDER = "02-dataset/combined_images"
IMG_EXT = ".png"


def denormalize_bbox(bbox, img_w, img_h):
    x1, y1, w, h = bbox

    x1 = int(x1 * img_w)
    y1 = int(y1 * img_h)
    x2 = int((x1 / img_w + w) * img_w)
    y2 = int((y1 / img_h + h) * img_h)

    return x1, y1, x2, y2


def main():
    with open(ANNOT_PATH, "r") as f:
        data = json.load(f)

    for img_id, entry in data.items():

        img_path = os.path.join(IMAGE_FOLDER, img_id + IMG_EXT)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        for i, bbox in enumerate(entry["bboxes"]):
            x1, y1, x2, y2 = denormalize_bbox(bbox, img_w, img_h)

            crop = img[y1:y2, x1:x2]

            label = entry["labels"][i] if i < len(entry["labels"]) else "unknown"

            window_name = f"{img_id} - {label} - {i}"

            cv2.imshow(window_name, crop)

            print(f"{img_id} | {label} | bbox: {bbox}")

            key = cv2.waitKey(0)  # press space key to go to next
            if key == 27:  # ESC to quit
                cv2.destroyAllWindows()
                return

            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
