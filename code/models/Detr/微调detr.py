"""
name：code & 微调detr
time：2024/11/14 19:07
author：yxy
content：
"""

from transformers import DetrForObjectDetection, DetrConfig
from datasets import load_dataset, DatasetDict
import torch

# 1. 加载数据集
dataset = load_dataset("coco", split="train2017")

# 2. 配置DETR模型
model_config = DetrConfig.from_pretrained("../../weight/detr", num_labels=91)  # 根据需要调整类别数
model = DetrForObjectDetection.from_pretrained("../../weight/detr", config=model_config)

# 3. 准备数据集
def preprocess(examples):
    return model.preprocess(examples)

encoded_dataset = dataset.map(preprocess, batched=True)

# 4. 创建DataLoader
from torch.utils.data import DataLoader

def collate_fn(batch):
    return model.prepare_inputs(batch)

dataloader = DataLoader(encoded_dataset, batch_size=8, collate_fn=collate_fn)

# 5. 设置优化器和损失函数
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# 6. 微调模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(3):  # 训练3个epoch
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 7. 保存模型
model.save_pretrained("detr_finetuned")