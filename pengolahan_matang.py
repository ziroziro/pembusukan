# # nama kelompok :
# # Kevin Gantama                (2215061133)
# # Muhammad Haikal Batubara     (2255061010)
# # Muhammad Dzikri Rofa         (2255061022)

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('matang.png')

k = 3
pixels = image.reshape((image.shape[0] * image.shape[1], 3))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented_image = labels.reshape((image.shape[0], image.shape[1]))

centers = np.uint8(centers)

dominant_color = centers[labels.flatten()]

colored_segmented_image = centers[labels.flatten()].reshape(image.shape)

gray_image = cv2.cvtColor(colored_segmented_image, cv2.COLOR_BGR2GRAY)
range_values = np.arange(0, 145)
pixel_count = np.sum(np.isin(gray_image, range_values))

print(f"Jumlah piksel hitam: {pixel_count}")

if pixel_count > 200000:
    print("Pisang Belum Matang")
elif pixel_count < 100000:
    print("Pisang Matang")
else:
    print("Pisang Busuk")

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Citra\nAsli'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(cv2.cvtColor(colored_segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmentasi\nWarna Utama'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(gray_image, cmap='gray')
plt.title('Segmentasi\nGrayscale'), plt.xticks([]), plt.yticks([])
plt.show()