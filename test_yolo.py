from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Image path
image_path = "media/test.jpg"   # Change this to your image

# Run prediction
results = model(image_path)

person_count = 0

for result in results:

    boxes = result.boxes

    for box in boxes:

        cls = int(box.cls)

        # COCO class 0 = Person
        if cls == 0:
            person_count += 1

    annotated = result.plot()

print("Detected Persons :", person_count)

cv2.imshow("YOLO Detection", annotated)

cv2.waitKey(0)

cv2.destroyAllWindows()