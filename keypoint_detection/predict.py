from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model
model = YOLO('./runs/pose/train5/weights/best.pt')  # load a custom model

# Predict with the model
results = model('./6.jpg', device=0, save=True, save_conf=True, conf=0.90, show=True, show_labels= True)  # predict on an image