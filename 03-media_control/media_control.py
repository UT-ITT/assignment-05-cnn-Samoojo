import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from pynput.keyboard import Controller, Key

from keras.models import load_model

IMG_SIZE = 64
COLOR_CHANNELS = 3
SIZE = (IMG_SIZE, IMG_SIZE)
LABEL_NAMES = ["like", "no_gesture", "stop", "peace"]
MODEL = "gesture_recognition.keras"

GESTURE_TO_ACTION = {
    "stop": "pause",
    "like": "play",
    "peace": "next",
}


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Use webcam hand gestures to control media playback."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / MODEL,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Minimum prediction confidence required to trigger a gesture.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Number of consecutive equal predictions before triggering.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between media key presses.",
    )
    parser.add_argument(
        "--min-hand-area",
        type=float,
        default=0.015,
        help="Minimum hand contour area as a fraction of the webcam frame.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=35,
        help="How far a pixel must be from white to count as foreground.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without showing the webcam preview window.",
    )
    return parser.parse_args()


def foreground_mask_from_white_background(frame, white_threshold):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    distance_from_white = np.linalg.norm(255 - blurred.astype("int16"), axis=2).astype(
        "float32"
    )
    _, mask = cv2.threshold(
        distance_from_white, white_threshold, 255, cv2.THRESH_BINARY
    )
    return mask.astype("uint8")


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(mask, kernel, iterations=1)


def best_hand_contour(mask, frame_shape, min_area_fraction):
    frame_area = frame_shape[0] * frame_shape[1]
    min_area = frame_area * min_area_fraction
    max_area = frame_area * 0.75

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / float(height)
        if aspect_ratio < 0.25 or aspect_ratio > 3.5:
            continue

        extent = area / float(width * height)
        if extent < 0.12:
            continue

        candidates.append((area, contour))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def crop_with_margin(frame, rect):
    x, y, width, height = rect
    margin = int(0.05 * max(width, height))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + width + margin)
    y2 = min(frame.shape[0], y + height + margin)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def crop_hand(frame, min_area_fraction, white_threshold):
    """Crop the largest foreground object from a virtual white background. (or well lit cardboard)"""
    mask = foreground_mask_from_white_background(frame, white_threshold)
    contour = best_hand_contour(clean_mask(mask), frame.shape, min_area_fraction)
    if contour is None:
        return None, None

    return crop_with_margin(frame, cv2.boundingRect(contour))


def preprocess_image(image):
    resized = cv2.resize(image, SIZE)
    if COLOR_CHANNELS == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    resized = resized.astype("float32") / 255.0
    return resized.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)


def predict(model, images):
    batch_size = model.input_shape[0] or len(images)

    if batch_size and len(images) < batch_size:
        padding = np.repeat(images[-1:], batch_size - len(images), axis=0)
        padded_images = np.concatenate([images, padding], axis=0)
        return model.predict(padded_images, verbose=0)[: len(images)]

    return model.predict(images, verbose=0)


def predict_gesture(
    model, frame, confidence_threshold, min_area_fraction, white_threshold
):
    crop, bbox = crop_hand(frame, min_area_fraction, white_threshold)
    if crop is None:
        return "no_gesture", 0.0, bbox

    probabilities = predict(model, preprocess_image(crop))[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    label = LABEL_NAMES[predicted_index]

    if confidence < confidence_threshold:
        return "no_gesture", confidence, bbox
    return label, confidence, bbox


class MediaController:
    def __init__(self, cooldown):
        self.keyboard = Controller()
        self.cooldown = cooldown
        self.last_action_at = 0.0

    def press_media_key(self, key):
        self.keyboard.press(key)
        self.keyboard.release(key)

    def trigger(self, gesture):
        action = GESTURE_TO_ACTION.get(gesture)
        if action is None:
            return False

        now = time.monotonic()
        if now - self.last_action_at < self.cooldown:
            return False

        if action == "play":
            self.press_media_key(Key.media_play_pause)
        elif action == "pause":
            self.press_media_key(Key.media_play_pause)
        elif action == "next":
            self.press_media_key(Key.media_next)

        self.last_action_at = now
        return True


def draw_preview(frame, label, confidence, bbox, triggered):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

    status = f"{label} ({confidence:.2f})"
    if triggered:
        status += " -> media key"

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    model = load_model(args.model)
    if model.output_shape[-1] != len(LABEL_NAMES):
        raise ValueError(
            f"The model outputs {model.output_shape[-1]} classes, but this script "
            f"expects {len(LABEL_NAMES)} labels: {LABEL_NAMES}"
        )

    cap = cv2.VideoCapture(args.camera)
    time.sleep(1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {args.camera}")

    media = MediaController(args.cooldown)
    previous_label = None
    stable_count = 0
    armed = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame from webcam.")
                break

            label, confidence, bbox = predict_gesture(
                model,
                frame,
                args.confidence,
                args.min_hand_area,
                args.white_threshold,
            )

            if label == previous_label:
                stable_count += 1
            else:
                previous_label = label
                stable_count = 1

            if label == "no_gesture":
                armed = True

            triggered = False
            if armed and label != "no_gesture" and stable_count >= args.stable_frames:
                triggered = media.trigger(label)
                if triggered:
                    armed = False

            if not args.no_preview:
                draw_preview(frame, label, confidence, bbox, triggered)
                cv2.imshow("Gesture Media Control - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
