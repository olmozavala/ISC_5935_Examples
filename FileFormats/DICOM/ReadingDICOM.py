import pytest
import itk
import matplotlib.pyplot as plt
import numpy as np
# API C++: https://itk.org/Doxygen/html/namespaceitk.html
# Python https://itkpythonpackage.readthedocs.io/en/master/

##
# dicom_path = "../../Data/DCE_DICOM_Prostate/"
dicom_path = "./Data/DCE_DICOM_Prostate/"

## ------- Reading Images data ------
# Read images
img = itk.imread(dicom_path)
# -- Read metadata as a dictionary
metadata_dict = itk.dict_from_image(img)  # Fewer attributes, better names ('origin','spacing',etc.)
print(metadata_dict)
metadata_dict_2 = dict(img)  # All attributes, raw names (dim, pixdim, qoffset,...)
print(metadata_dict)
# -- Access data as np arrays
imge_np = itk.array_view_from_image(img)
dims = imge_np.shape
slides = dims[0]
print(f"Dimensions of the image are: {dims}")

## Plotting all body planes
print("Plotting all slides ...")
for slide in range(slides):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.imshow(imge_np[slide, :, :])  # Trans
    axs.set_title(f'Slide {slide}')
    plt.show()
print("Done!")
##

