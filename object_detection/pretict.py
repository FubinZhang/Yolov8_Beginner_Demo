from ultralytics import YOLO
from PIL import Image

# 导入模型
model = YOLO('./runs/detect/train2/weights/best.pt') 
# 加载一个预训练的YOLOv8n模型，在训练结束后会提示最终/最好模型的存放位置


# 可以找一个测试图片来测试模型的预测情况
model.predict('oranges_4.png', save=True, save_conf=True, conf=0.90, show=True) 
# Show the results


# 对目标列表进行批量推理
# results = model(['oranges_4.png'], save=True, imgsz=320, conf=0.2)  # 返回一个Results对象列表
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image
