import matplotlib.pyplot as plt

# 绘制一些数据
## 这些轴构成了一个坐标系
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# 获取当前轴对象
ax = plt.gca()

# 在当前轴上添加文本
ax.text(2, 8, 'This is a text', fontsize=12, color='blue')

# 显示图表
plt.show()