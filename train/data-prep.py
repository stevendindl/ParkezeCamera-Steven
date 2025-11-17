# data-prep.py

import os
from pathlib import Path

"""
first training data preparation script for yolov5 parking lot model
two datasets: PKLot and CNRPark+EXT

model 1: all normal images
model 2: augmented images with radial distortion and fisheye effect
    75% normal images,
    15% mild fisheye augmentation,
    10% strong fisheye augmentation
"""

# convert CNR-EXT_FULL_IMAGE_1000x750 to yolov5 format

'''
data/CNR-EXT_FULL_IMAGE_1000x750/images

txt line example from all.txt = RAINY/2016-02-12/camera1/R_2016-02-12_09.10_C01_191.jpg 1
                                               2016-02-12_0710.jpg

format = {weather_subfolder}/{date_subfolder}/{camera_subfoler}/{filename}
so path = RAINY/2016-02-12/camera1/{filename_pt1}{filename_pt2}.jpg
filename_pt1 = 2016-02-12_07
filename_pt2 = .10 -> _10
filename_pt3 = .jpg

csv lookup "09.10_C01_191.jpg" -> C01 = ?, 191 = parking spot id 

CNRPark+EXT.csv - main csv with metadata 
camera,datetime,day,hour,image_url,minute,month,occupancy,slot_id,weather,year,occupant_changed
A,20150703_0805,3,8,CNRPark/A/free/20150703_0805_1.jpg,5,7,0,1,S,2015,

csv's with locations (camera[x].csv) 
SlotId,X,Y,W,H
603,1034,1640,240,240

'''
cnr_dir = Path("../data/CNR-EXT_FULL_IMAGE_1000x750/")
images = Path(cnr_dir, "images")
txts = Path(cnr_dir, "txt")
csvs = Path(cnr_dir, "csv")

print(images)

image_path = "RAINY/2016-02-12/camera1/R_2016-02-12_09.10_C01_191.jpg"

# Extract components
path_parts = image_path.split('/')
weather = path_parts[0]
date = path_parts[1]
camera_folder = path_parts[2]
filename_dummy = path_parts[3]

filename_dummy = filename_dummy[2:].replace(".", "").split("_C0")
image_name = filename_dummy[0] + ".jpg"
cam_id, slot_id = filename_dummy[1].replace("jpg", "").split("_")

print(image_name)
print(cam_id)
print(slot_id)

"2016-02-12_0910.jpg"

# print dataset size 
tr1 = "../data/CNRPark-EXT-YOLO/train"
tr2 = "../data/PKLot.v2-640.yolov5pytorch/train"

v1 = "../data/CNRPark-EXT-YOLO/val"
v2 = "../data/PKLot.v2-640.yolov5pytorch/val"

tt1 = "../data/CNRPark-EXT-YOLO/test"
tt2 = "../data/PKLot.v2-640.yolov5pytorch/test"

from collections import Counter
from pathlib import Path
import glob

def count_files_in_folder(folder_path):
    total_files = 0
    file_types = Counter()
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
        for file in files:
            ext = Path(file).suffix
            file_types[ext] += 1
    print(f'File types in {folder_path}:')
    for ext, count in file_types.items():
        print(f'  {ext}: {count}')
    return total_files

count_files_in_folder(tr1)
count_files_in_folder(tr2)

count_files_in_folder(v1)
count_files_in_folder(v2)

count_files_in_folder(tt1)
count_files_in_folder(tt2)
