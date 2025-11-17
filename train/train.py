# train.py - yolov5
from ultralytics import YOLO

# Load a model
model = YOLO(model='yolov5su.pt')  

'''
# Model training params
# init learning rate = 0.01
# batch size = 16
# img size = 640
# epochs = 100
# SGD optimizer w/ learning rate of 0.1 and weight decay of 0.0005
# '''

cnrpark_data = 'data/CNRPark-EXT.v3i.yolov5pytorch/data.yaml'
pklot_data = 'data/PKLot.v2-640.yolov5pytorch/data.yaml'


# model.train(data='data/coco128.yaml', epochs=100, imgsz=640, batch=16, lr0=0.01, optimizer='SGD', weight_decay=0.0005)