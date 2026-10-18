import json
import cv2

# Load gambar latar belakang area parkir kampusmu
img_path = 'test1.jpg'  # Pastikan foto ini ada di folder proyek
img = cv2.imread(img_path)

if img is None:
  print(f"❌ Error: Foto '{img_path}' tidak ditemukan!")
  exit()

slots = []
current_box = []


# Fungsi callback saat mouse diklik
def draw_rectangle(event, x, y, flags, param):
  global current_box, img

  # Klik kiri untuk mulai menggambar / menandai sudut
  if event == cv2.EVENT_LBUTTONDOWN:
    current_box.append([x, y])
    print(f"Titik dipilih: [{x}, {y}]")

    # Jika sudah terkumpul 2 titik (Sudut kiri-atas dan kanan-bawah)
    if len(current_box) == 2:
      slot_id = len(slots) + 1
      x1, y1 = current_box[0]
      x2, y2 = current_box[1]

      # Pastikan x1<x2 dan y1<y2
      box_coords = [
          min(x1, x2),
          min(y1, y2),
          max(x1, x2),
          max(y1, y2),
      ]

      # Simpan ke list slots
      slots.append({
          'id': slot_id,
          'label': f'Slot A{slot_id}',
          'bbox': box_coords,
      })

      # Gambar kotak permanen di layar
      cv2.rectangle(
          img,
          (box_coords[0], box_coords[1]),
          (box_coords[2], box_coords[3]),
          (0, 255, 0),
          2,
      )
      cv2.putText(
          img,
          f'A{slot_id}',
          (box_coords[0], box_coords[1] - 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )
      cv2.imshow('Mapping Slot Parkir', img)

      print(f"✅ Slot A{slot_id} berhasil dibuat: {box_coords}")
      current_box = []  # Reset untuk slot berikutnya


# Tampilkan jendela gambar
cv2.imshow('Mapping Slot Parkir', img)
cv2.setMouseCallback('Mapping Slot Parkir', draw_rectangle)

print(
    '--- INSTRUKSI ---'
    '\n1. Klik MOUSE KIRI di sudut KIRI-ATAS kotak slot.'
    '\n2. Klik MOUSE KIRI sekali lagi di sudut KANAN-BAWAH kotak slot.'
    '\n3. Ulangi untuk semua slot.'
    '\n4. Tekan tombol [S] di keyboard untuk SIMPAN ke slots_config.json.'
    '\n5. Tekan tombol [Q] untuk KELUAR.'
)

while True:
  key = cv2.waitKey(1) & 0xFF

  # Tombol 's' untuk menyimpan file JSON
  if key == ord('s'):
    with open('slots_config.json', 'w') as f:
      json.dump(slots, f, indent=4)
    print(f'💾 Berhasil! {len(slots)} slot parkir disimpan ke slots_config.json')
    break

  # Tombol 'q' untuk keluar tanpa simpan
  elif key == ord('q'):
    print('Keluar tanpa menyimpan.')
    break

cv2.destroyAllWindows()