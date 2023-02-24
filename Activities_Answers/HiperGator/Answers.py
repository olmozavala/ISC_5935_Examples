import itk
import matplotlib.pyplot as plt

##
file = "../..//Data/OTHERS/BrainT1_OriginalSize.nii"
img = itk.imread(file)
data = itk.array_from_image(img)

##
metadata = itk.dict_from_image(img)
print(f"Space {metadata['spacing']}")
print(f"Size {metadata['size']}")
print(f"Origin {metadata['origin']}")

## Resampling to .5, .5, .5
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(data[80,:,:], cmap='gray')
plt.tight_layout()
plt.show()

##
new_spacing = (0.5, 0.5, 0.5)

interpolator = itk.LinearInterpolateImageFunction.New(img)
resampler = itk.ResampleImageFilter.New(img)
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(itk.size(img)*2)
resampler.SetOutputDirection(img.GetDirection())
resampler.SetOutputOrigin(img.GetOrigin())
resampler.SetInterpolator(interpolator)
resampler.Update()
resampled = resampler.GetOutput()
meta_res = itk.dict_from_image(resampled)
##

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img[80,:,:], cmap='gray')
axs[0].set_title(f"Original pixel spacing: {metadata['spacing']}")
axs[1].imshow(resampled[160,:,:], cmap='gray')
axs[1].set_title(f"Resampled pixel spacing: {meta_res['spacing']}")
plt.tight_layout()
plt.show()

##

