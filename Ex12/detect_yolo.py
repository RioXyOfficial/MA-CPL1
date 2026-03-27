import json
import os
import sys
from ultralytics import YOLO


def detect_objects(image_path):
    if not os.path.isfile(image_path):
        print(f"Erreur : image '{image_path}' introuvable.")
        sys.exit(1)

    print("Chargement du modèle YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print(f"Détection sur : {image_path}")
    results = model(image_path, save=True, project="runs/detect", name="predict")

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]

        confidence = round(float(box.conf[0].item()), 4)

        bbox = box.xyxy[0].tolist()
        bbox = [round(coord, 2) for coord in bbox]

        detection = {
            "classe": class_name,
            "confiance": confidence,
            "bounding_box": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            },
        }
        detections.append(detection)

        print(f"  -> {class_name} (confiance: {confidence}) | bbox: {bbox}")

    output_dir = os.path.join("runs", "detect", "predict")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "detections.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)

    print(f"\nNombre d'objets détectés : {len(detections)}")
    print(f"Image annotée           : {output_dir}/")
    print(f"Fichier JSON            : {json_path}")

    return detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python detect_yolo.py <chemin_image>")
        sys.exit(1)

    image_file = sys.argv[1]
    detect_objects(image_file)