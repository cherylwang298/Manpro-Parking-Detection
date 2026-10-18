import os
from ultralytics import YOLO
from collections import Counter

# 1. Load model terbaik (best.pt) yang sudah didownload dari Colab
# Taruh file best.pt di dalam folder yang sama atau tentukan path-nya
model = YOLO('best.pt')

# 2. Lakukan prediksi pada foto uji di komputer lokal
results = model.predict(
    source='test2.jpg',  # Pastikan file foto ada di folder proyek atau isi path lengkapnya
    conf=0.4,            # Batas minimum confidence score (40%)
    show=False,          # Di VS Code lokal, biasanya 'show=True' butuh jendela GUI terpisah, jadi lebih aman pakai False lalu simpan hasilnya
    save=True            # Menyimpan hasil gambar ber-bounding box
)

# 3. Menampilkan Laporan / Rekap Status Parkir yang Rapi
res = results[0]
boxes = res.boxes.cls.cpu().numpy() # Mengambil label kelas objek
class_names = res.names             # Nama kelas dari model

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
print("Hasil gambar ber-bounding box tersimpan di folder: runs/detect/predict/")