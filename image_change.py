import cv2
import numpy as np
import matplotlib.pyplot as plt


def logarithmic_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * np.log(1 + img)
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed


img = cv2.imread('datasets/before/1.jpg', 0)

log_transformed = logarithmic_transform(img)

plt.imshow(log_transformed, cmap='gray')
plt.title('Logarithmic Transformed Image')
plt.show()
