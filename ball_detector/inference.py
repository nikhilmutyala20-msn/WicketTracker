
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="WicketTracker cricket ball detector"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to ball_detector.pt")
    parser.add_argument("--source", required=True,
                        help="Path to an image or video")
    parser.add_argument("--output", default="outputs/ball_detection",
                        help="Output directory")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    output = Path(args.output)

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(output),
        name="run",
        exist_ok=True,
        verbose=False,
    )

    print(f"Processed {len(results)} result(s).")
    print(f"Results saved under: {output / 'run'}")


if __name__ == "__main__":
    main()
