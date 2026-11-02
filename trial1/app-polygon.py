import json
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best_3.pt")

# Load config 4 titik (polygon)
with open("slots_config.json", "r") as f:
    SLOTS = json.load(f)


@app.get("/api/parking-status")
def get_parking_status():
    image_path = "cropgedq.jpg"
    results = model.predict(source=image_path, conf=0.35, iou=0.45, save=False)[0]

    car_boxes = []
    class_names = results.names

    # for box in results.boxes:
    #     cls_id = int(box.cls[0].item())
    #     label_name = class_names.get(cls_id, "").lower()
    #     coords = box.xyxy[0].cpu().numpy().tolist()

    #     if "car" in label_name:
    #         x1, y1, x2, y2 = coords
    #         # Ambil titik tengah bawah mobil (footprint) sebagai acuan parkir
    #         footprint_x = int((x1 + x2) / 2)
    #         footprint_y = int(y2)
    #         car_boxes.append((footprint_x, footprint_y))

    for box in results.boxes:
      cls_id = int(box.cls[0].item())
      label_name = class_names.get(cls_id, "").lower()
      coords = box.xyxy[0].cpu().numpy().tolist()

      if "car" in label_name:
        x1, y1, x2, y2 = coords

        # Gunakan titik pusat (center point) dari mobil agar lebih akurat menagkap bodi mobil
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        car_boxes.append((center_x, center_y))

    slot_results = []
    occupied_count = 0
    anomaly_count = 0

    for slot in SLOTS:
        # Ubah list points dari JSON menjadi NumPy array format OpenCV
        pts = np.array(slot["pts"], dtype=np.int32)
        matching_cars = 0

        for car_pt in car_boxes:
            # cv2.pointPolygonTest mengembalikan:
            # > 0 jika titik di dalam polygon, == 0 di garis, < 0 di luar
            result = cv2.pointPolygonTest(pts, car_pt, False)
            if result >= 0:
                matching_cars += 1

        if matching_cars == 0:
            status = "empty"
        elif matching_cars == 1:
            status = "occupied"
            occupied_count += 1
        else:
            status = "anomaly"
            anomaly_count += 1

        slot_results.append({
            "id": slot["id"],
            "label": slot["label"],
            "status": status,
            "pts": slot["pts"],
        })

    total_slots = len(SLOTS)
    empty_count = total_slots - occupied_count - anomaly_count

    return {
        "summary": {
            "total": total_slots,
            "occupied": occupied_count,
            "empty": empty_count,
            "anomaly": anomaly_count,
        },
        "slots": slot_results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)