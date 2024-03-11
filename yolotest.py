from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolov8x.pt")
src = "https://www.youtube.com/watch?v=M8QoC6iMO40"
#result = model.predict(source=src,conf=0.5, show=True)
#print(result.boxes)
results = model(source=src,conf=0.5, show=True)
print(results)

