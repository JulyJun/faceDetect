@echo off
conda activate jun
cls
set msg=GPU Current Usage
echo today: %DATE%, %TIME%
echo %msg%

nvidia-smi

echo ****YOLO version check****
yolo checks

pip install roboflow
python
from roboflow import Roboflow
rf = Roboflow(api_key="GtFDSq2OGauuFtVjHR8w")
project = rf.workspace("mohamed-traore-2ekkp").project("face-detection-mik1i")
version = project.version(18)
dataset = version.download("yolov8")
exit()

yolo detect train yolov8s.pt data={dataset.location}/data.yaml source=/content/Face-Detection-18/valid/images epochs=40 imgsz=800