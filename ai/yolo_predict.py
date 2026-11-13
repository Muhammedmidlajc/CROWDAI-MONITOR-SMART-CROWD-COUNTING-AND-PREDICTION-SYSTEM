import os
import cv2

from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")


def yolo_predict(image_path):

    results = model(image_path)

    person_count = 0
    total_box_area = 0

    image_area = 0

    annotated_image = None

    for result in results:

        annotated_image = result.plot()

        h, w = result.orig_shape
        image_area = h * w

        for box in result.boxes:

            if int(box.cls) != 0:
                continue

            person_count += 1

            x1, y1, x2, y2 = box.xyxy[0]

            area = (x2 - x1) * (y2 - y1)

            total_box_area += float(area)

    average_box_area = 0

    if person_count > 0:
        average_box_area = total_box_area / person_count

    # -----------------------------
    # Save YOLO Detection Image
    # -----------------------------

    base, _ = os.path.splitext(image_path)

    save_path = base + "_yolo.png"

    cv2.imwrite(save_path, annotated_image)

    return (
        person_count,
        average_box_area,
        image_area,
        save_path
    )