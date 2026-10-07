import argparse
import os
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test pitch detection at one-second intervals."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO model (.pt)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input cricket video"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="YOLO confidence threshold (default: 0.40)"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # --------------------------------------------------
    # Validate arguments
    # --------------------------------------------------

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    if not 0.0 <= args.conf <= 1.0:
        print("ERROR: confidence must be between 0 and 1")
        sys.exit(1)

    print("=" * 70)
    print("🏏 WICKETTRACKER — PITCH DETECTION EVERY SECOND TEST")
    print("=" * 70)

    print(f"Model      : {args.model}")
    print(f"Video      : {args.input}")
    print(f"Confidence : {args.conf}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO(args.model)

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Open video
    # --------------------------------------------------

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERROR: Could not determine video FPS.")
        cap.release()
        sys.exit(1)

    duration = frame_count / fps

    print("\nVideo information")
    print("-" * 70)
    print(f"FPS       : {fps:.2f}")
    print(f"Frames    : {frame_count}")
    print(f"Duration  : {duration:.2f} seconds")
    print("-" * 70)

    # --------------------------------------------------
    # Test one frame every second
    # --------------------------------------------------

    second = 0
    detected_count = 0
    tested_count = 0

    while second <= int(duration):

        frame_number = int(second * fps)

        # Prevent seeking beyond the video
        if frame_number >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            print(
                f"{second:4d}s → ⚠ Could not read frame"
            )
            second += 1
            continue

        results = model.predict(
            source=frame,
            conf=args.conf,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        tested_count += 1

        # --------------------------------------------------
        # Check for pitch detection
        # --------------------------------------------------

        if boxes is not None and len(boxes) > 0:

            confidences = boxes.conf.cpu().numpy()

            best_conf = float(max(confidences))

            detected_count += 1

            print(
                f"{second:4d}s → ✅ PITCH DETECTED "
                f"| Confidence: {best_conf:.3f}"
            )

        else:

            print(
                f"{second:4d}s → ❌ NO PITCH DETECTED"
            )

        second += 1

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()

    detection_rate = (
        detected_count / tested_count * 100
        if tested_count > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)

    print(f"Frames tested       : {tested_count}")
    print(f"Pitch detected      : {detected_count}")
    print(f"No pitch detected  : {tested_count - detected_count}")
    print(f"Detection rate      : {detection_rate:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()