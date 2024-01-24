from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model
model = YOLO('./runs/pose/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('./test.jpg', device=0, save=True, save_conf=True, conf=0.90)  # predict on an image

for r in results:
    im_array = r.plot(conf=0.9, kpt_radius=20)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image