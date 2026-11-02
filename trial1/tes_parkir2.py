import os
from ultralytics import YOLO
from collections import Counter

model = YOLO('best_3.pt')

results = model.predict(
    source='tes_gedq1.jpg',
    conf=0.4,
    show=False,
    save=True,
    project='hasil_prediksi', # Membuat folder 'hasil_prediksi' langsung di dalam direktori utama
    name='output',            # Subfolder tempat menyimpan foto
    exist_ok=True,            # Memastikan tidak membuat folder berulang (output2, output3, dst.)
    imgsz=640
)

res = results[0]
boxes = res.boxes.cls.cpu().numpy()
class_names = res.names

print("\n" + "="*35)
print(" 📊 LAPORAN STATUS PARKIR (LOCAL VS CODE) ")
print("="*35)

total_detected = len(boxes)
if total_detected == 0:
    print("Tidak ada objek yang terdeteksi.")
else:
    counts = Counter(boxes)
    for class_id, count in counts.items():
        label_name = class_names[int(class_id)]
        print(f"• Total {label_name}: {count} buah")

print("="*35)
# Menampilkan path lengkap folder hasil
print(f"Gambar tersimpan di: {os.path.abspath(res.save_dir)}")