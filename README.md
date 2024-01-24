# Yolov8_Beginner_Demo

一份关于yolov8的训练以及预测的代码demo（目标检测/实例分割/关键点检测........）

## 1.install

安装torch+torchvision，torch官网有可选配置的安装指令提供，以及Previous versions提供

`https://pytorch.org/`

安装ultralytics

`pip install ultralytics`

安装习惯的图像处理安装包处理result

`pip install 。。。。。`

## 2.数据集标注

工具：labelme

`pip install labelme`

安装后在环境终端输入指令labelme即可使用

### （1）object_detection

在labelme中create rectangle即可,

每一张image都会生成一个 `.json`文件，

转换格式可通过开源项目转化

`https://github.com/rooneysh/Labelme2YOLO`（同样适用于 `目标检测`/`实例分割`）

or

`https://github.com/TommyZihao/Label2Everything`

（同样适用于 `目标检测`/`实例分割`/`关键点检测`）

最终数据集文件格式如下：

```
└─fruits_sets
    │  dataset.yaml
    ├─images
    │  ├─train
    │  └─val
    └─labels
        ├─train
        └─val

```

其中 `dataset.yaml`格式如下：

```
train: path/images/train/
val: path/images/val/

nc: 3

names: ['orange', 'tomato', 'apple']
```

ps：train和val最好使用绝对路径

补ps：或者在最开始加一个path：.......之后train和val使用相对于path的相对路径

### （2）instance_segmentation

在labelme中create polygon即可

方法和文件目录格式同目标检测相同

### （3）keypoint_detection

在labelme中create rectangle和create point即可

注意由于关键点检测也是基于目标检测的，所以标注的时候要有rectangle和point两种

方法和文件目录格式同上，需要注意的是 `dataset.yaml`的格式略有不一样

```
# 训练集、验证集、测试集的路径（一般不用区分验证集和测试集，统称测试集）
train: path\images/train
val: path\images/val
test: path\images/val

# 3 种关键点，每个关键点有 X Y 是否可见 三个参数
# 是否可见：2-可见不遮挡 1-遮挡 0-没有点
kpt_shape: [3, 3]

# 框的类别（对于关键点检测，只有一类）
names:
  0: apple
```

## 2.训练（train.py）

三种任务的训练代码都非常简单。

首先都是载入模型，yolov8+n/s/m/l/x 是不同级别的目标检测预训练模型，后面+‘-seg’是实例分割模型，后面+‘-pose’是关键点检测模型，因为后两者都是基于目标检测的所以都会自动先加载目标检测模型。

```
from ultralytics import YOLO

# 加载一个模型
# 会自动从库中下载你选中的预训练模型
model = YOLO('yolov8n.yaml')  # 从YAML文件构建一个新模型
model = YOLO('yolov8n.pt')  # 加载一个预训练模型（推荐用于训练）
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML文件构建模型并加载权重
```

其次是训练，训练的这个接口中其实包含很多可调参数，这里用到的是epochs(训练次数)，batch(每一批的数据量)，imgsz(图片大小)，device(使用的训练设备，是gpu还是cpu)，workers(线程数量)，save_period(多少次训练会保存一次结果)，

```
results = model.train(data='dataset.yaml', epochs=500, batch=16, imgsz=640, device=0, workers=0, save_period=50)
```

参数在官方文档中有详细说明 `https://docs.ultralytics.com/modes/train/`

## 3.预测（predict.py）

```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('./best.pt')  # load a custom model

# Predict with the model
results = model('test.png', save=True, save_conf=True, conf=0.60, show=True)  # predict on an image
```

照例首先是载入模型，然后是预测图片，同样预测函数中也有很多可调参数包括置信度(conf),是否保存,是否展示,是否按照种类筛选等等......

在官方文档中有详细介绍 `https://docs.ultralytics.com/modes/predict/`

后续待更。。。。。
