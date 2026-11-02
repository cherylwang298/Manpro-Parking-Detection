import json
import cv2
import numpy as np

# Konfigurasi file
IMAGE_PATH = "cropgedq.jpg"  # Pastikan menggunakan gambar acuan yang sama
CONFIG_PATH = "slots_config.json"

current_pts = []
slots = []

# Coba load config yang sudah ada jika ada
try:
    with open(CONFIG_PATH, "r") as f:
        slots = json.load(f)
    print(f"Berhasil memuat {len(slots)} slot dari {CONFIG_PATH}")
except FileNotFoundError:
    slots = []


def mouse_callback(event, x, y, flags, param):
    global current_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_pts) < 4:
            current_pts.append([x, y])
            print(f"Titik {len(current_pts)} ditambahkan: ({x}, {y})")


def main():
    global current_pts, slots

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {IMAGE_PATH}")
        return

    clone = img.copy()
    cv2.namedWindow("Make Config - 4 Points")
    cv2.setMouseCallback("Make Config - 4 Points", mouse_callback)

    print("\n--- PANDUAN KONTROL ---")
    print("1. Klik 4 titik berurutan untuk 1 slot:")
    print("   (1: Kiri-Atas, 2: Kanan-Atas, 3: Kanan-Bawah, 4: Kiri-Bawah)")
    print("2. Tombol 'u' : Batalkan titik terakhir pada slot saat ini")
    print("3. Tombol 'z' : Hapus slot terakhir yang sudah disimpan")
    print("4. Tombol 's' : Simpan ke slots_config.json")
    print("5. Tombol 'q' : Keluar tanpa menyimpan")
    print("------------------------\n")

    slot_id = len(slots) + 1

    while True:
        display_img = clone.copy()

        # Gambar slot yang sudah tersimpan sebelumnya (Warna Hijau)
        for slot in slots:
            pts = np.array(slot["pts"], dtype=np.int32)
            cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                display_img,
                slot["label"],
                (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Gambar titik-titik yang sedang diklik untuk slot aktif (Warna Merah & Oranye)
        for pt in current_pts:
            cv2.circle(display_img, (pt[0], pt[1]), 4, (0, 0, 255), -1)

        if len(current_pts) > 1:
            pts_np = np.array(current_pts, dtype=np.int32)
            cv2.polylines(display_img, [pts_np], False, (0, 165, 255), 2)

        # Jika sudah 4 titik, otomatis buat slot baru
        if len(current_pts) == 4:
            label_name = f"Slot A{slot_id}"
            slots.append(
                {"id": slot_id, "label": label_name, "pts": current_pts}
            )
            print(f"Berhasil menambahkan {label_name}!")
            slot_id += 1
            current_pts = []

        cv2.imshow("Make Config - 4 Points", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Keluar.")
            break
        elif key == ord("s"):
            with open(CONFIG_PATH, "w") as f:
                json.dump(slots, f, indent=4)
            print(
                f"Konfigurasi berhasil disimpan ke {CONFIG_PATH} dengan total {len(slots)} slot."
            )
        elif key == ord("u"):
            if len(current_pts) > 0:
                removed = current_pts.pop()
                print(f"Titik {removed} dibatalkan.")
        elif key == ord("z"):
            if len(slots) > 0:
                removed_slot = slots.pop()
                slot_id -= 1
                print(f"Slot terakhir ({removed_slot['label']}) dihapus.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()