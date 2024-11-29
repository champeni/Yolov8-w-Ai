import cv2
import argparse
import numpy as np
import torch

from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live detection")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int,
        help="Webcam resolution (width height)"
    )
    args = parser.parse_args()
    return args

def plot_boxes(results, frame):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box details
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(xyxy[0]), int(xyxy[1])), 
                (int(xyxy[2]), int(xyxy[3])),
                (0, 255, 0), 
                2
            )
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(
                frame, label, 
                (int(xyxy[0]), int(xyxy[1] - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1
            )
    return frame

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8x.pt")
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA not available. Running on CPU.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        results = model(frame, show=False)  # Disable automatic display
        frame = plot_boxes(results, frame)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(30) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
