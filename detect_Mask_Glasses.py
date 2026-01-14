from ultralytics import YOLO
model = YOLO("mask-glasses-yolov8m\\weights\\best.pt")


result = model('img1.jpg')

print(result)

