import pytest
import itk
import matplotlib.pyplot as plt
import numpy as np
# API C++: https://itk.org/Doxygen/html/namespaceitk.html
# Python https://itkpythonpackage.readthedocs.io/en/master/

##
# input_file = "./TestData/Breast_MRI.nii"
# input_file = "./TestData/10647_1000663_cor.nii"
input_file = "./TestData/Reg_2.nii"

## ------- Reading Image
# Read images
img = itk.imread(input_file)
# -- Read metadata as a dictionary
metadata_dict = itk.dict_from_image(img)  # Fewer attributes, better names ('origin','spacing',etc.)
metadata_dict = dict(img)  # All attributes, raw names (dim, pixdim, qoffset,...)
# -- Access data as np arrays
imge_np = itk.array_view_from_image(img)
dims = imge_np.shape
print(dims)

## Showing planes
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].imshow(imge_np[int(dims[0]/2), :, :])  # Trans
axs[1].imshow(imge_np[:, int(dims[1]/2), :])  # Coronal
axs[2].imshow(imge_np[:, :, int(dims[2]/3)])  # Sag
plt.show()

# ----------------- Filters ---------
## Median filter
smoothed = itk.mean_image_filter(img, radius=1)
mean_filter = itk.MeanImageFilter.New(img)
mean_filter.SetRadius(4)
mean_filter.Update()
smoothed2 = mean_filter.GetOutput()

fig, axs = plt.subplots(1, 3, figsize=(22,5))
axs[0].imshow(img[int(dims[0]/2), :, :])  # Trans
axs[1].imshow(smoothed[int(dims[0]/2), :, :])  # Trans
axs[2].imshow(smoothed2[int(dims[0]/2), :, :])  # Trans
plt.tight_layout()
plt.show()
##

