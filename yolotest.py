from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("best.pt")
src = "0"
#result = model.predict(source=src,conf=0.5, show=True)
#print(result.boxes)
results = model(source=src,conf=0.5, show=True)
print(results)

