import pytest
import itk
import matplotlib.pyplot as plt
import numpy as np
# API C++: https://itk.org/Doxygen/html/namespaceitk.html
# Python https://itkpythonpackage.readthedocs.io/en/master/
import matplotlib as mpl
mpl.rc('image', cmap='gray')

##
input_file = "./Data/MRI/t2_prostate_coronal.nii"

## ------- Reading Images data ------
# Read images
image_original = itk.imread(input_file)
# -- Read metadata as a dictionary
metadata_dict = itk.dict_from_image(image_original)  # Fewer attributes, better names ('origin','spacing',etc.)
print(metadata_dict)
metadata_dict = dict(image_original)  # All attributes, raw names (dim, pixdim, qoffset,...)
print(metadata_dict)
# -- Access data as np arrays
imge_np = itk.array_view_from_image(image_original)
dims = imge_np.shape
print(f"Dimensions of the image are: {dims}")

## Plotting all body planes
print("Plotting all body planes")
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].imshow(imge_np[int(dims[0]/2), :, :])  # Trans
axs[0].set_title('Transversal')
axs[1].imshow(imge_np[:, int(dims[1]/2), :], aspect=8)  # Coronal
axs[1].set_title('Coronal')
axs[2].imshow(imge_np[:, :, int(dims[2]/2)], aspect=8)  # Sag
axs[2].set_title('Saggital')
plt.show()

# ----------------- Filters ---------
## Median filter
print("Computing Median filter...")
smoothed = itk.mean_image_filter(image_original, radius=3)
mean_filter = itk.MeanImageFilter.New(image_original)
mean_filter.SetRadius(4)
mean_filter.Update()
smoothed2 = mean_filter.GetOutput()

fig, axs = plt.subplots(1, 3, figsize=(12,5))
axs[0].imshow(image_original[int(dims[0]/2), :, :])  # Trans
axs[0].set_title('Original')
axs[1].imshow(smoothed[int(dims[0]/2), :, :])  # Trans
axs[1].set_title('Smoothed (mean filter)')
axs[2].imshow(smoothed2[int(dims[0]/2), :, :])  # Trans
axs[2].set_title('Smoothed (mean filter Radius 4)')
plt.tight_layout()
plt.show()

## Sobel filter
print("Computing Sobel filter...")
smoothed = itk.sobel_edge_detection_image_filter(image_original)

fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].imshow(image_original[int(dims[0]/2), :, :])  # Trans
axs[0].set_title('Original')
axs[1].imshow(smoothed[int(dims[0]/2), :, :], cmap='Greys_r')  # Trans
axs[1].set_title('Edge detction (sobel filter)')
plt.tight_layout()
plt.show()


## Resample
print("Original spacing: ", image_original.GetSpacing())
print("Original origin: ", image_original.GetOrigin())
print("Original direction: ", image_original.GetDirection())
print("Resampling...")
new_spacing = [0.15, 0.15, 3.0]  # spacing in mm

##
interpolator = itk.LinearInterpolateImageFunction.New(image_original)
resampler = itk.ResampleImageFilter.New(image_original)

# Set the resampling parameters
resampler.SetOutputSpacing(new_spacing)
print(itk.size(image_original))
# resampler.SetSize(itk.size(image_original))
resampler.SetSize((720*2,720*2,25))
resampler.SetOutputDirection(image_original.GetDirection())
# resampler.SetOutputOrigin(image_original.GetOrigin())
resampler.SetOutputOrigin((-99, 6.80609, 227.443))

# Set the interpolation method
resampler.SetInterpolator(interpolator)

# Execute the resampling
resampler.Update()

# Get the resampled output image
resampled = resampler.GetOutput()

print("Done!")

print(f"Dimensions of the original image are: {image_original.shape}")
print(f"Dimensions of the image are: {resampled.shape}")
print(f"Spacing of the original image are: {image_original.GetSpacing()}")
print(f"Spacing of the image are: {resampled.GetSpacing()}")

fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].imshow(image_original[int(dims[0]/2), :, :])  # Trans
axs[0].set_title('Original')
axs[1].imshow(resampled[int(dims[0]/2), :, :], cmap='Greys_r')  # Trans
axs[1].set_title('Resampled image')
plt.tight_layout()
plt.show()
##

