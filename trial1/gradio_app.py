import json
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# =========================
# LOAD MODEL & KONFIGURASI
# =========================
model = YOLO("best_3.pt")

with open("slots_config.json", "r") as f:
    SLOTS = json.load(f)

# Warna BGR untuk OpenCV
COLOR_EMPTY   = (0, 200, 0)     # hijau  -> kosong
COLOR_OCCUPIED = (0, 0, 255)    # merah  -> terisi
COLOR_ANOMALY  = (0, 165, 255)  # oranye -> anomali

OVERLAP_THRESHOLD = 0.30
CONF_THRESHOLD = 0.35


# =========================
# FUNGSI UTILITAS
# =========================
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


def detect_frame(frame):
    """Deteksi pada satu frame (numpy BGR). Mengembalikan frame beranotasi & summary."""
    results = model.predict(source=frame, conf=CONF_THRESHOLD, save=False, verbose=False)[0]

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
            if overlap > OVERLAP_THRESHOLD:
                matching_cars += 1

        if matching_cars == 0:
            status = "empty"
            color = COLOR_EMPTY
        elif matching_cars == 1:
            status = "occupied"
            color = COLOR_OCCUPIED
            occupied_count += 1
        else:
            status = "anomaly"
            color = COLOR_ANOMALY
            anomaly_count += 1

        slot_results.append({
            "id": slot["id"],
            "label": slot.get("label", str(slot["id"])),
            "status": status,
            "bbox": slot_bbox,
        })

        # Gambar slot
        x1, y1, x2, y2 = map(int, slot_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{slot.get('label', slot['id'])}: {status.upper()}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    total_slots = len(SLOTS)
    empty_count = total_slots - occupied_count - anomaly_count

    summary = {
        "total": total_slots,
        "occupied": occupied_count,
        "empty": empty_count,
        "anomaly": anomaly_count,
    }
    return frame, summary, slot_results


def make_summary_text(summary):
    return (
        f"### Ringkasan Parkir\n"
        f"- **Total slot**: {summary['total']}\n"
        f"- 🟢 **Kosong**: {summary['empty']}\n"
        f"- 🔴 **Terisi**: {summary['occupied']}\n"
        f"- 🟠 **Anomali**: {summary['anomaly']}\n"
    )


def make_slot_table(slot_results):
    rows = []
    for s in slot_results:
        emoji = {"empty": "🟢", "occupied": "🔴", "anomaly": "🟠"}.get(s["status"], "")
        rows.append([s["id"], s["label"], f"{emoji} {s['status']}"])
    return rows


# =========================
# HANDLER: GAMBAR
# =========================
def process_image(image):
    if image is None:
        return None, "Silakan unggah gambar terlebih dahulu.", []

    # Gradio memberi RGB, konversi ke BGR untuk OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    annotated, summary, slot_results = detect_frame(frame)

    # Konversi balik ke RGB untuk ditampilkan Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return (
        annotated_rgb,
        make_summary_text(summary),
        make_slot_table(slot_results),
    )


# =========================
# HANDLER: VIDEO
# =========================
def process_video(video_path):
    if not video_path:
        return None, "Silakan unggah video terlebih dahulu.", []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Gagal membuka video.", []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "output_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    last_summary = None
    last_slots = []

    # Batasi jumlah frame agar tidak terlalu lama (opsional)
    MAX_FRAMES = 300
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= MAX_FRAMES:
            break

        annotated, summary, slot_results = detect_frame(frame.copy())
        writer.write(annotated)

        last_summary = summary
        last_slots = slot_results
        frame_idx += 1

    cap.release()
    writer.release()

    if last_summary is None:
        return None, "Tidak ada frame yang diproses.", []

    return (
        out_path,
        make_summary_text(last_summary) + f"\n_Frame diproses: {frame_idx}_",
        make_slot_table(last_slots),
    )


# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="Deteksi Parkir (YOLO)") as demo:
    gr.Markdown(
        "# 🚗 Deteksi Parkir Otomatis\n"
        "Deteksi slot parkir melalui **gambar** atau **video**.\n"
        "Keterangan warna: 🟢 Kosong · 🔴 Terisi · 🟠 Anomali (tidak rapi / >1 mobil)"
    )

    with gr.Tabs():
        # ---------- TAB GAMBAR ----------
        with gr.Tab("📸 Gambar"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Unggah Gambar Parkir", type="numpy")
                    btn_img = gr.Button("🔍 Deteksi Gambar", variant="primary")
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Hasil Deteksi")
                    img_summary = gr.Markdown()
                    img_table = gr.Dataframe(
                        headers=["ID", "Label", "Status"],
                        label="Status Slot",
                        interactive=False,
                    )

            btn_img.click(
                process_image,
                inputs=img_input,
                outputs=[img_output, img_summary, img_table],
            )

        # ---------- TAB VIDEO ----------
        with gr.Tab("🎬 Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Unggah Video Parkir")
                    btn_vid = gr.Button("🎥 Deteksi Video", variant="primary")
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Hasil Deteksi Video")
                    vid_summary = gr.Markdown()
                    vid_table = gr.Dataframe(
                        headers=["ID", "Label", "Status"],
                        label="Status Slot (frame terakhir)",
                        interactive=False,
                    )

            btn_vid.click(
                process_video,
                inputs=vid_input,
                outputs=[vid_output, vid_summary, vid_table],
            )

    gr.Markdown(
        "---\n"
        "**Catatan**: Untuk video panjang, hanya 300 frame pertama yang diproses agar tetap responsif. "
        "Threshold overlap slot–mobil = `0.30`, confidence YOLO = `0.35`."
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)