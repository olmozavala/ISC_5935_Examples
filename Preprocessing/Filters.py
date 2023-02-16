import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
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
##

