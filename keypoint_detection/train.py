from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8s-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s-pose.yaml').load('yolov8s-pose.pt')  # build from YAML and transfer weights
                
# Train the model
results = model.train(data='./apple_keypoint/dataset.yaml', epochs=500, batch=16, imgsz=640, device=0, workers=0, save_period=50)