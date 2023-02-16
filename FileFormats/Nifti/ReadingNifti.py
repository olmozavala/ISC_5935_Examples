import pytest
import itk
import matplotlib.pyplot as plt
import numpy as np
# API C++: https://itk.org/Doxygen/html/namespaceitk.html
# Python https://itkpythonpackage.readthedocs.io/en/master/

##
input_file = "./Data/MRI/t2_prostate_coronal.nii"

## ------- Reading Images data ------
# Read images
img = itk.imread(input_file)
# -- Read metadata as a dictionary
metadata_dict = itk.dict_from_image(img)  # Fewer attributes, better names ('origin','spacing',etc.)
print(metadata_dict)
metadata_dict = dict(img)  # All attributes, raw names (dim, pixdim, qoffset,...)
print(metadata_dict)
# -- Access data as np arrays
imge_np = itk.array_view_from_image(img)
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
##

