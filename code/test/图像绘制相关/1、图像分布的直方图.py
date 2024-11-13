import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_images_histogram(image_paths, bins=256):
    """
    绘制多张图像的灰度直方图。
    :param image_paths: 包含图像文件路径的列表。
    :param bins: 直方图的柱数。
    """
    # 创建一个画布，根据图像数量动态调整大小
    plt.figure(figsize=(15, 5 * len(image_paths)))

    for i, image_path in enumerate(image_paths, 1):
        # 加载图像并转换为灰度
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)

        # 计算直方图
        histogram, bin_edges = np.histogram(image_np, bins=bins, range=(0, 255))

        # 绘制直方图
        plt.plot(bin_edges[0:-1], histogram, label=f"Image {i}")
    plt.title(f"Grayscale Histogram for Image {i}")
    plt.xlabel("Grayscale Value")
    plt.ylabel("Pixels")
    plt.xlim([0, 255])
    plt.legend()
    plt.tight_layout()
    plt.show()

# 使用示例
image_paths = ['../../data/img/desktop.png', '../../data/img/desktop2.png']  # 替换为你的图像文件路径列表
plot_images_histogram(image_paths)