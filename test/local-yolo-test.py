# local-yolo-test.py

from ultralytics import YOLO

model_path = "../models/best-2025-11-17.pt"
model = YOLO(model_path)

# batch folder
images_dir = "../images/"
results = model.predict(f'{images_dir}*.jpg', imgsz=640, conf=0.25)

