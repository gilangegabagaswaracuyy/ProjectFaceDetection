import cv2
import os

# Path untuk file haarcascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Mengecek apakah file haarcascade ada
if not os.path.exists(cascade_path):
    print("Error: File cascade tidak ditemukan di path yang benar.")
    print("Pastikan Anda memiliki file haarcascade_frontalface_default.xml di direktori yang tepat.")
    exit()

print(f"File cascade ditemukan di: {cascade_path}")

# Menggunakan cascade classifier standar OpenCV
face_ref = cv2.CascadeClassifier(cascade_path)
camera = cv2.VideoCapture(0)

# Pastikan kamera terbuka
if not camera.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()
else:
    print("Kamera berhasil dibuka.")

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Menambahkan parameter minNeighbors dan minSize untuk optimasi
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def drawer_box(frame):
    faces = face_detection(frame)
    if len(faces) == 0:
        print("Tidak ada wajah yang terdeteksi")
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = camera.read()

        # Cek apakah frame berhasil dibaca
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
        
        print("Membaca frame...")  # Debugging output
        drawer_box(frame)
        
        cv2.imshow("LangFace AI", frame)

        # Cek jika pengguna menekan tombol 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()
