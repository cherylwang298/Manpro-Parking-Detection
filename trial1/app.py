import json
import cv2
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


model = YOLO("best_1.pt")
with open("slots_config.json", "r") as f:
  SLOTS = json.load(f)


def calculate_slot_overlap(car_box, slot_box):
  xA = max(car_box[0], slot_box[0])
  yA = max(car_box[1], slot_box[1])
  xB = min(car_box[2], slot_box[2])
  yB = min(car_box[3], slot_box[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)
  if interArea == 0:
    return 0.0

  slotArea = (slot_box[2] - slot_box[0]) * (slot_box[3] - slot_box[1])
  if slotArea == 0:
    return 0.0

  return interArea / float(slotArea)


@app.get("/api/parking-status")
def get_parking_status():
  
  image_path = "test1.jpg"
  results = model.predict(source=image_path, conf=0.35, save=False)[0]

  car_boxes = []
  class_names = results.names

  for box in results.boxes:
    cls_id = int(box.cls[0].item())
    label_name = class_names.get(cls_id, "").lower()
    coords = box.xyxy[0].cpu().numpy().tolist()

    if "car" in label_name:
      car_boxes.append(coords)

  slot_results = []
  occupied_count = 0
  anomaly_count = 0

  for slot in SLOTS:
    slot_bbox = slot["bbox"]
    matching_cars = 0

    for car_bbox in car_boxes:
      overlap = calculate_slot_overlap(car_bbox, slot_bbox)
      if overlap > 0.30:
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
        "bbox": slot_bbox,
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