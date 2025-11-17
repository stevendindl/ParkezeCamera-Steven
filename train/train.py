# train.py - yolov5
from ultralytics import YOLO
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='path to data yaml')
parser.add_argument('--output', type=str, default='runs/train', help='output dir')
args = parser.parse_args()

# Load a model
model = YOLO(model='yolov5su.pt')  

'''
Model training params
init learning rate = 0.01
batch size = 16
img size = 640
epochs = 100
SGD optimizer w/ and weight decay of 0.0005

File types in ../data/CNRPark-EXT-YOLO/train: (1000x750)
  .jpg: 2055
  .txt: 2055
File types in ../data/PKLot.v2-640.yolov5pytorch/train: (640x640)
  .jpg: 8691
  .txt: 8691
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# model.train(data='../data/data.yaml', epochs=100, imgsz=640, batch=16, device=device) # defaults
# model.train(data='../data/data.yaml', epochs=100, imgsz=640, batch=16, lr0=0.01, optimizer='SGD', weight_decay=0.0005, momentum=0.937, device=device) 
## ^ adjustments made based on Chaoyang University paper - https://www.sciencedirect.com/org/science/article/pii/S1546221825009531

# robust config (with elements from Chaoyang University paper)
# Training settings
'''
If overfitting (train loss low, val loss high): increase augmentation
If underfitting (both high): reduce augmentation or train longer
Good sign if val loss decreases then plateaus and train loss continues to decrease
  # 'blur': 0.01,        # failed. not supported in yolov5?
'''
config = {
    'data': '../data/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,        # Adjust based on GPU memory
    'device': device,
    'optimizer': 'SGD',  # Stochastic Gradient Descent (Adam may be simpler, but research uses SGD)

    'momentum': 0.937,
    # 'lr0': 0.01,         # Base learning rate 
    'lr0': 0.001,          # Updated option for learning reate
    'lrf': 0.01,           # Final LR multiplier
    'weight_decay': 0.0005,  # Regularization 

    'patience': 50,     # Early stopping patience (epochs without improvement)
    'save': True,       # Save checkpoints
    'save_period': 10,  # Save checkpoint every N epochs
    'cache': False,     # Cache images (True for faster training if you have RAM)
    'workers': 8,       # Dataloader workers

    # Output settings
    'project': 'runs/train',  # Save results to project/name
    'name': 'parking-detection',  # Experiment name
    'exist_ok': False,  # Overwrite existing project/name
    'pretrained': True,  # Use pretrained weights
    'verbose': True,  # Verbose output

    # Augmentation settings
    'hsv_h': 0.015,         # Day/night/weather variations
    'hsv_s': 0.7,        
    'hsv_v': 0.4,           # Shadows, lighting

    # Geometric (conservative changes because parking lots are grid aligned)
    'degrees': 5.0,      # Small rotation only
    'translate': 0.1,    
    'scale': 0.5,
    'flipud': 0.0,       # NO vertical flip
    'fliplr': 0.5,       # YES horizontal flip
    'shear': 2.0,           #  Camera angle variations
    'perspective': 0.0003,  # Lens distortion

    # Advanced
    'mosaic': 1.0,       # Great for variety
    'mixup': 0.1,        # Slight blending for robustness
    'close_mosaic': 10,  # Disable mosaic last 10 epochs for stability
    
    # Validation
    'val': True,    # Validate during training
    'plots': True,  # Save plots
    'rect': False,  # Rectangular training (may be faster)
}

# update config['data'] and output paths from args
if args.config:
    config['data'] = args.config
config['project'] = args.output

# Train the model
results = model.train(**config)

# Validation
print("\n" + "="*80)
print("RUNNING VALIDATION")
print("="*80)
metrics = model.val()

print("\nValidation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Save final model
fname = f"parkdetect_{time.strftime('%Y-%m-%d')}_{time.strftime('%H-%M')}.pt"
model.save(fname)