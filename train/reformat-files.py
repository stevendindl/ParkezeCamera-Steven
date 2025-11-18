# reformat-files.py

import os
from pathlib import Path

test = "2025-11-17_17-39-15_cam4_snapshot.jpg"

test = test.split('_')

if len(test) > 2:
    date = test[0]
    time = test[1]
    cam_part = test[2]
    cam_id = cam_part.replace("cam", "").replace(".jpg", "")
    new_filename = f"{cam_part}_{date}_{time}.jpg"
    print("New filename:", new_filename)
    print("Camera ID:", cam_id)

input_dir = "../images"
output_dir = "../images"

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            parts = file.split('_')
            if len(parts) > 2:
                date = parts[0]
                time = parts[1]
                cam_part = parts[2]
                cam_id = cam_part.replace("cam", "").replace(".jpg", "")
                new_filename = f"{cam_part}_{date}_{time}.jpg"
                
                old_path = Path(root, file)
                new_path = Path(root, new_filename)
                
                print(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)
