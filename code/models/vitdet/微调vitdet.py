from transformers import VitDetBackbone, VitDetModel
from transformers.models import vitdet
from PIL import Image
import requests

# 初始化特征提取器和模型
feature_extractor = VitDetFeatureExtractor.from_pretrained('facebook/vitdet-base')
model = VitDetModel.from_pretrained('facebook/vitdet-base')

# 下载一个示例图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 提取图像特征
inputs = feature_extractor(images=image, return_tensors="pt")

# 获取模型输出
outputs = model(**inputs)

# 获取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)

