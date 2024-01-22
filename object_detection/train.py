from ultralytics import YOLO

# 加载一个模型
# 会自动从库中下载你选中的预训练模型
model = YOLO('yolov8n.yaml')  # 从YAML文件构建一个新模型
model = YOLO('yolov8n.pt')  # 加载一个预训练模型（推荐用于训练）
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML文件构建模型并加载权重


# 训练模型 设置参数注意/workers线程数/device/epochs/
results = model.train(data='./fruits_set.yaml', epochs=50, batch=16, imgsz=640, device=0, workers=0, save_period=50)