#!/usr/bin/env python3
"""
Minimal YOLO training script for canopy detection.
"""
import torch
from ultralytics import YOLO

# Load YOLOv11n-seg model
model = YOLO("yolo11n-seg.pt")

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'

# Train the model
results = model.train(
    data="yolo_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=device
)
save_dir = results.save_dir
best = save_dir / 'weights' / 'best.pt'
model = YOLO(best)
metrics = model.val(
    data="yolo_dataset/data.yaml",
    imgsz=640,
    device=device,
    split='val',
    iou=0.7,
    conf=0.001,
)
to_optimize = metrics.results_dict['metrics/mAP50(M)']
print(to_optimize)
