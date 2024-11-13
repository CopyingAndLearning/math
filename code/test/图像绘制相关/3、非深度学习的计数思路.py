import cv2
import numpy as np

# 读取图像
image = cv2.imread('../../data/img/apple/apple.png')

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 使用阈值分割生成二值化图像
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary_image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 连通区域分析
_, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# 统计苹果数量
apple_count = len(labels) - 1  # 减去背景标签

# 输出结果
print(f"苹果数量：{apple_count}")

# 可视化结果（可选）
for i in range(1, len(stats)):
    x, y, w, h, area = stats[i]