import os
import sys

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.yolo_predict import yolo_predict
from ai.predict import predict_image


def hybrid_predict(image_path):
    """
    Intelligent Hybrid Crowd Estimation Pipeline (Stage 1):
    Combines YOLOv8 Object Detection (for sparse, moderate, and distinct human detection)
    with Enhanced MC-CNN Density Regression (for ultra-dense, congested, far-field crowds).

    Guarantees:
    1. Zero-crowd / empty frames correctly report 0 people via YOLOv8.
    2. Small to medium crowd scenes (1 - 30 people) use precise YOLOv8 bounding box counts.
    3. Ultra-dense scenes (overlapping crowds, massive gatherings, high density heatmaps)
       seamlessly activate Enhanced MC-CNN for accurate density regression.
    """
    # -------------------------------
    # 1. Run YOLOv8
    # -------------------------------
    (
        yolo_count,
        average_box_area,
        image_area,
        yolo_image
    ) = yolo_predict(image_path)

    density_ratio = (average_box_area / image_area) if image_area > 0 else 0.0

    print("\n==============================")
    print("Hybrid Prediction")
    print("==============================")
    print("YOLO Count :", yolo_count)
    print("Average Box Area :", average_box_area)
    print("Density Ratio :", density_ratio)

    # -------------------------------
    # 2. Clear / Empty Frame
    # -------------------------------
    if yolo_count == 0:
        print("Selected Model : YOLOv8 (Clear / Empty Scene)")
        return (
            0,
            yolo_image,
            "YOLOv8",
            0,
            0
        )

    # -------------------------------
    # 3. Small to Moderate Crowd (1 - 30 people)
    # -------------------------------
    # If YOLO clearly localized countable individuals:
    # We only switch to MC-CNN if there is an overwhelmingly dense, congested crowd
    # (e.g. ShanghaiTech style where hundreds of heads overlap and YOLO severely undercounts)
    if yolo_count <= 30:
        mccnn_count, heatmap_path, max_density = predict_image(image_path, return_max=True)

        # Ultra-dense crowd criteria:
        # MC-CNN count must be high (>= 120), peak spatial density must indicate tightly packed heads (>= 0.0045),
        # and MC-CNN count must be significantly higher than YOLO's localized detections.
        if mccnn_count >= 120 and max_density >= 0.0045 and mccnn_count > 2.5 * yolo_count:
            print("Selected Model : Enhanced MC-CNN (Ultra-Dense Crowd Detected)")
            return (
                mccnn_count,
                heatmap_path,
                "Enhanced MC-CNN",
                yolo_count,
                mccnn_count
            )
        else:
            print("Selected Model : YOLOv8 (Distinct Individual Localization)")
            return (
                yolo_count,
                yolo_image,
                "YOLOv8",
                yolo_count,
                mccnn_count
            )

    # -------------------------------
    # 4. Dense / Large Crowd (yolo_count > 30)
    # -------------------------------
    mccnn_count, heatmap_path = predict_image(image_path)

    if mccnn_count > yolo_count:
        print("Selected Model : Enhanced MC-CNN (High-Density Spatial Regression)")
        return (
            mccnn_count,
            heatmap_path,
            "Enhanced MC-CNN",
            yolo_count,
            mccnn_count
        )
    else:
        print("Selected Model : YOLOv8 (High-Count Bounding Boxes)")
        return (
            yolo_count,
            yolo_image,
            "YOLOv8",
            yolo_count,
            mccnn_count
        )


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "media", "IMG_1.jpg")
    if os.path.exists(test_img):
        print(f"Testing hybrid prediction on: {test_img}")
        res = hybrid_predict(test_img)
        print("\n--- Summary ---")
        print(f"Final Count : {res[0]}")
        print(f"Output Image: {res[1]}")
        print(f"Model Used  : {res[2]}")
        print(f"YOLO Count  : {res[3]}")
        print(f"MC-CNN Count: {res[4]}")
    else:
        print("Usage: python ai/hybrid_predict.py <image_path>")
        print("Model loaded and hybrid_predict function ready.")