from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# 模型名称，可以从Hugging Face模型库中选择
model_name = '../../weight/vit'

# 加载特征提取器和模型
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 加载图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 预处理图像
inputs = feature_extractor(images=image, return_tensors="pt")

# 使用模型进行预测
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# 打印预测结果
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Predicted class:", model.config.id2label[predicted_class_idx])
print(model.config.id2label.get(predicted_class_idx, "Unknown"))