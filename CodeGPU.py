import cv2
import torch
import time
import psutil
import GPUtil
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import pandas as pd

# Set expected label index (Pomeranian)
expected_label = 259

#Now load and process model 
model_name = "facebook/deit-base-patch16-224"
device = torch.device("cuda")
model = ViTForImageClassification.from_pretrained(model_name).to(device)
processor = ViTImageProcessor.from_pretrained(model_name)

# Video input
video_path = "5877829-hd_1080_1920_30fps.mp4"
gif = cv2.VideoCapture(video_path)

# Metrics
total_frames = 0
correct_predictions = 0
prev_time = time.time()

frame_data = []
i=0

while True:
    ret, frame = gif.read()
    if not ret or frame is None or frame.size == 0:
        break

    # Preprocessing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = outputs.logits.argmax(-1).item()
        label_name = model.config.id2label[pred_label]

    # Accuracy
    total_frames += 1
    if pred_label == expected_label:
        correct_predictions += 1
    accuracy = (correct_predictions / total_frames) * 100

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # GPU usage
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].memoryUsed
    
    #storing all the values of variables
    frame_data.append({
        "frame" : i,
        "fps": fps,
        "accuracy": accuracy,
        "gpu_usage": gpu_usage
    })

    i = i+1
        
    # Overlay the indexes on frame
    cv2.putText(frame, f"Prediction: {label_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"GPU Memory: {gpu_usage:.1f} MB", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Display
    cv2.namedWindow("ViT Inference", cv2.WINDOW_NORMAL)
    cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow("ViT Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df = pd.DataFrame(frame_data)   
df.to_csv("frame_statistics1.csv",index=False)
gif.release()
cv2.destroyAllWindows()
