from google.colab import files
from ultralytics import YOLO
import cv2
import glob
import os

print("📥 Mời ní chọn file 'best.pt' để nạp vào model:")
uploaded_weights = files.upload()

weight_path = 'best.pt'
for filename in uploaded_weights.keys():
    if filename.endswith('.pt'):
        weight_path = filename

print(f"🎯 Đang load mô hình từ: {weight_path}")
model = YOLO(weight_path)

print("📷 Mời ní chọn ảnh lá lúa cần nhận diện:")
uploaded_img = files.upload()

for filename in uploaded_img.keys():
    print(f"🔍 Đang nhận diện sâu bệnh trên ảnh: {filename}...")
    results = model.predict(source=filename, conf=0.1, save=True)
    
    list_of_files = glob.glob('runs/detect/predict*/*')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"🖼️ Đã tìm thấy ảnh kết quả: {latest_file}")
        img = cv2.imread(latest_file)
        if img is not None:
            output_path = "/content/ket_qua_nhan_dien.png"
            cv2.imwrite(output_path, img)
            print("💾 Đang tự động tải ảnh kết quả về máy tính...")
            files.download(output_path)
        else:
            print("❌ Lỗi: Không thể đọc ảnh kết quả!")
    else:
        print("❌ Lỗi: Không tìm thấy ảnh kết quả sau khi nhận diện!")
