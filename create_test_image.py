import cv2
import numpy as np

# 创建一个黑色背景的图像
img = np.zeros((400, 600, 3), dtype=np.uint8)

# 绘制几个椭圆
cv2.ellipse(img, (200, 200), (100, 50), 0, 0, 360, (255, 255, 255), 2)
cv2.ellipse(img, (400, 200), (80, 80), 45, 0, 360, (255, 255, 255), 2)
cv2.ellipse(img, (300, 150), (60, 30), 90, 0, 360, (255, 255, 255), 2)
cv2.ellipse(img, (300, 250), (70, 40), 135, 0, 360, (255, 255, 255), 2)

# 保存图像到 images 文件夹
cv2.imwrite('images/test.jpg', img)

print("测试图像已创建: images/test.jpg")