import argparse
import os
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO-based pitch detection on a cricket video."
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
        "--output",
        required=True,
        help="Path where the annotated output video will be saved"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="YOLO confidence threshold (default: 0.40)"
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Run YOLO every Nth frame (default: 5)"
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

    if args.frame_skip < 1:
        print("ERROR: frame-skip must be >= 1")
        sys.exit(1)

    if not 0.0 <= args.conf <= 1.0:
        print("ERROR: confidence must be between 0 and 1")
        sys.exit(1)

    # Create output directory if necessary
    output_dir = os.path.dirname(os.path.abspath(args.output))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🏏 WICKETTRACKER — PITCH DETECTION")
    print("=" * 60)

    print(f"Model       : {args.model}")
    print(f"Input       : {args.input}")
    print(f"Output      : {args.output}")
    print(f"Confidence  : {args.conf}")
    print(f"Frame skip  : {args.frame_skip}")

    # --------------------------------------------------
    # Load YOLO model
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo information")
    print("-" * 40)
    print(f"FPS         : {fps:.2f}")
    print(f"Resolution  : {width} x {height}")
    print(f"Frames      : {frame_count}")
    print(f"YOLO frames : ~{frame_count // args.frame_skip}")

    # --------------------------------------------------
    # Create output video
    # --------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print(f"ERROR: Could not create output video: {args.output}")
        cap.release()
        sys.exit(1)

    # --------------------------------------------------
    # Process video
    # --------------------------------------------------

    frame_number = 0
    processed_frames = 0
    detections_found = 0

    # Previous annotated frame
    last_annotated_frame = None

    print("\nStarting pitch detection...")
    print("-" * 60)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------
        # Run YOLO every Nth frame
        # ----------------------------------------------

        if frame_number % args.frame_skip == 0:

            results = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False
            )

            result = results[0]

            # Count detections
            if result.boxes is not None and len(result.boxes) > 0:
                detections_found += 1

            # Draw detections
            annotated_frame = result.plot()

            last_annotated_frame = annotated_frame

            processed_frames += 1

        else:

            # Reuse previous annotated frame
            if last_annotated_frame is not None:
                annotated_frame = last_annotated_frame
            else:
                annotated_frame = frame

        # Write frame
        out.write(annotated_frame)

        # ----------------------------------------------
        # Progress information
        # ----------------------------------------------

        if frame_number % 3000 == 0:

            percentage = (
                frame_number / frame_count * 100
                if frame_count > 0
                else 0
            )

            print(
                f"Processed: {frame_number}/{frame_count} "
                f"({percentage:.1f}%) | "
                f"YOLO frames: {processed_frames} | "
                f"Detections: {detections_found}"
            )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()
    out.release()

    print("\n" + "=" * 60)
    print("✅ PITCH DETECTION COMPLETE")
    print("=" * 60)

    print(f"Total frames       : {frame_number}")
    print(f"YOLO frames        : {processed_frames}")
    print(f"Frames with pitch  : {detections_found}")
    print(f"Output             : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()