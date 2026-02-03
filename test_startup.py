"""Minimal startup test — run this to check imports work."""
import sys
print(f"Python: {sys.version}")

print("Importing numpy...", end=" ")
import numpy as np
print(f"OK ({np.__version__})")

print("Importing cv2...", end=" ")
import cv2
print(f"OK ({cv2.__version__})")

print("Importing onnxruntime...", end=" ")
import onnxruntime as ort
print(f"OK ({ort.__version__})")

print("Loading ONNX model...", end=" ")
import os
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.onnx")
if not os.path.exists(model_path):
    model_path = "/app/yolov8n-pose.onnx"
print(f"path={model_path} exists={os.path.exists(model_path)}")

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
inp = session.get_inputs()[0]
print(f"Model loaded: {inp.name} {inp.shape}")

print("Importing FastAPI...", end=" ")
from fastapi import FastAPI
print("OK")

print("\nAll imports OK! Server should start fine.")
