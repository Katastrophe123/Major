import cv2
import pandas as pd
import psutil
import torch
from psutil import time 
import torchvision.transforms as T
from transformers import ViTForImageClassification,ViTImageProcessor

from PIL import Image

expected_label = 259

model_name = "facebook/deit-base-patch16-224"

model = ViTForImageClassification.from_pretrained(model_name).to("cpu")

extractor = ViTImageProcessor.from_pretrained(model_name)

device = torch.device("cpu")

gif = cv2.VideoCapture("5877829-hd_1080_1920_30fps.mp4")

prev_time = time.time()#For calculating FPS

#For Tracking accuracy
total_frames = 0
correct_predictions = 0

frame_data = []
i = 0

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=extractor.image_mean, std=extractor.image_std)
])#For transforming function

while gif.isOpened():#Inference Loop
    ret, frame = gif.read()
    if not ret or frame is None:
        break
    
    # Preprocessing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = extractor(images=image, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k,v in inputs.items()}

    
    #Here Resize and Transform Frames
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = inputs["pixel_values"].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
    
    label = model.config.id2label[pred.item()]
    confidence = conf.item()
    
    #For Finding FPS here
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = outputs.logits.argmax(-1).item()
        label_name = model.config.id2label[pred_label]

    
    #For Tracking Accuracy
    total_frames += 1
    if pred_label == expected_label:
        correct_predictions += 1
    accuracy = (correct_predictions / total_frames) * 100
    
    #For System resources
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    
    #storing all the values of variables
    frame_data.append({
        "frame" : i,
        "fps": fps,
        "accuracy": accuracy,
        "cpu_usage": cpu_usage,
        "memory_usage": ram_usage
    })

    i = i+1
        
    #Overlay on Frames includes these following Syntax
    
    cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame,f"Accuracy:{accuracy:.2f}%",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame, f"CPU: {cpu_usage}%", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"RAM: {ram_usage}%", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    

    #For Displaying the frame
    cv2.namedWindow("ViT Inference", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("ViT Inference", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("ViT Inference",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df = pd.DataFrame(frame_data)   
df.to_csv("frame_statistics.csv",index=False) 
gif.release()
cv2.destroyAllWindows()
