"""
name：code & 测试detr的效果
time：2024/11/14 19:19
author：yxy
content：
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("../../data/img/apple/apple.png")

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("../../weight/detr", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("../../weight/detr", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
    # 绘制矩形框
    ax.add_patch(
        plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=2))
    # 绘制类别标签
    plt.text(box[0], box[1], f'{model.config.id2label[label.item()]} {round(score.item(), 3)}', color='red',
             fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

plt.show()