import numpy as np
from skimage import io
from skimage.feature import canny
from scipy import ndimage as ndi

img = io.imread('/home/jon-ross/Desktop/bee-wing/enhanced_image/1 Lasioglossum leucozonium f left 3.2x.jpg')[:,:,0]
img = io.imread('/home/jon-ross/Desktop/bee-wing/enhanced_image/001 Osmia lignaria m right 4x.jpg')[:,:,0]
img = io.imread('/home/jon-ross/Desktop/bee-wing/enhanced_image/1 Lasioglossum leucozonium f right 3.2x.jpg')[:,:,0]
img = io.imread('/home/jon-ross/Desktop/bee-wing/enhanced_image/1080 Lasioglossum rohweri f left 4x.jpg')[:,:,0]
"""
edges = canny(img/255.)
fill_regions = ndi.binary_fill_holes(edges)

io.imshow(edges)
io.show()

label_objects, nb_labels = ndi.label(fill_regions)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
regions_cleaned = mask_sizes[label_objects]

io.imshow(regions_cleaned)
io.show()
"""


"""
from skimage.morphology import watershed
from skimage.filters import sobel


markers = np.zeros_like(img)
markers[img < 200] = 1
markers[img > 150] = 2
elevation_map = sobel(img)

segmentation = watershed(elevation_map, markers)

#segmentation = ndi.binary_fill_holes(segmentation - 1)
io.imshow(segmentation)
io.show()
"""

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte

denoised = rank.median(img, disk(2))

markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

gradient = rank.gradient(denoised, disk(2))

labels = watershed(gradient, markers)

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (8,8), sharex = True, sharey = True)
ax = axes.ravel()

print('denoised')
print(denoised)
print('markers')
print(markers)
print('gradient')
print(gradient)
print('labels')
print(labels)


ax[0].imshow(img, cmap = plt.cm.gray, interpolation = 'nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap = plt.cm.nipy_spectral, interpolation = 'nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap = plt.cm.nipy_spectral, interpolation = 'nearest')
ax[2].set_title("Markers")

ax[3].imshow(img, cmap = plt.cm.gray, interpolation = 'nearest')
ax[3].imshow(labels, cmap = plt.cm.nipy_spectral, interpolation = 'nearest', alpha = 0.7)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
