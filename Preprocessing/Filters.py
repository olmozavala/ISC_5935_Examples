import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy import signal

## ------- Reading Image
# Read images
input_file = "./Data/Preproc/FiltersExampleLarge.png"
img = iio.imread(input_file)[:,:,0]

fig, ax = plt.subplots(1,1)
ax.imshow(img, cmap='Greys_r')
plt.show()

## ---------- Mean filter ---------
# mean_filter = np.ones((3,3)) * 1/9
mean_filter = np.ones((5,5)) * 1/25
img_mean= signal.convolve2d(img, mean_filter, boundary='symm', mode='same')

fig, axs = plt.subplots(1,2, figsize=(10,5))
im1 = axs[0].imshow(img, cmap='Greys_r', vmin=0, vmax=250)
im2 = axs[1].imshow(img_mean, cmap='Greys_r', vmin=0, vmax=250)
plt.colorbar(im1, ax=axs[0])
plt.colorbar(im2, ax=axs[1])
plt.tight_layout()
plt.show()

## --------- Edge detector --------
sobelx = [[-1, 1],[-1, 1]]
sobely = [[-1, -1],[1, 1]]
img_x= signal.convolve2d(img_mean, sobelx, boundary='symm', mode='same')
img_y= signal.convolve2d(img_mean, sobely, boundary='symm', mode='same')
img_sobel = img_x+img_y

fig, axs = plt.subplots(1,2, figsize=(10,5))
im1 = axs[0].imshow(img, cmap='Greys_r', vmin=0, vmax=250)
im2 = axs[1].imshow(img_sobel, cmap='Greys_r',)
plt.colorbar(im1, ax=axs[0])
plt.colorbar(im2, ax=axs[1])
plt.tight_layout()
plt.show()
print("Done")

## ------------- Dilation -------------
image = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

image = np.array(image)

kernel = [[1, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]

kernel = np.array(kernel)

def dilation(image, kernel):
    m, n = image.shape
    a, b = kernel.shape
    output = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if image[i, j] == 1:
                for k in range(a):
                    for l in range(b):
                        if kernel[k, l] == 1:
                            p = i + (k - 1)
                            q = j + (l - 1)
                            if p >= 0 and p < m and q >= 0 and q < n:
                                output[p, q] = 1
    return output

def erosion(image, kernel):
    m, n = image.shape
    a, b = kernel.shape
    output = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if image[i, j] == 1:
                flag = 1
                for k in range(a):
                    for l in range(b):
                        if kernel[k, l] == 1:
                            p = i + (k - 1)
                            q = j + (l - 1)
                            if p >= 0 and p < m and q >= 0 and q < n:
                                if image[p, q] == 0:
                                    flag = 0
                if flag == 1:
                    output[i, j] = 1
    return output

dilated = dilation(image, kernel)
dilated_scipy = scipy.ndimage.binary_dilation(image, kernel)
eroded = erosion(image, kernel)

fig, axs = plt.subplots(2,2, figsize=(10,5))
axs[0][0].imshow(image, cmap='Greys_r')
axs[0][0].set_title(f"Original {image.shape}")
axs[0][1].imshow(kernel)
axs[0][1].set_title(f"Structure {kernel.shape}")
# axs[1][0].imshow(dilated, cmap='Greys_r')
axs[1][0].imshow(dilated_scipy, cmap='Greys_r')
axs[1][0].set_title("Dilated")
axs[1][1].imshow(eroded, cmap='Greys_r')
axs[1][1].set_title("Eroded")
plt.tight_layout()
plt.show()
print("Done")

##

