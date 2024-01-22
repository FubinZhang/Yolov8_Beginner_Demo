from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('./runs/segment/train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model('test.png', save=True, save_conf=True, conf=0.60, show=True)  # predict on an image