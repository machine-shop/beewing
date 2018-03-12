import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from skimage import data
from skimage import color
from skimage.util import view_as_blocks

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from skimage.exposure import rescale_intensity
from skimage.morphology import reconstruction

from skimage.feature import corner_harris, corner_subpix, corner_peaks


# Load an color image in grayscale
%matplotlib inline
for index in range(1,12):
    image = mpimg.imread("./raw_image/" + str(index) + ".jpg")
    plt.imshow(image)
    plt.show()

    img_gray = rgb2gray(image)
    plt.imshow(img_gray)
    plt.show()


    thresh = threshold_otsu(img_gray)
    binary = img_gray > thresh

    # fig, axes = plt.subplots(ncols=3)
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
    # ax[1] = plt.subplot(1, 3, 2)
    # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

    # plt.imshow(img_gray, cmap=plt.cm.gray)
    # plt.set_title('Original')
    # ax[0].axis('off')

    # plt.hist(img_gray.ravel(), bins=256)
    # plt.set_title('Histogram')
    # plt.axvline(thresh, color='r')

    plt.imshow(binary, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


    # step one for otsu thresholding with appropriate nbins number 
    thresh = threshold_otsu(img_gray, nbins = 60)
    binary = img_gray > thresh
    plt.imshow(binary, cmap=cm.Greys_r)
    plt.title('Otsu threshold with nbins = 60')
    plt.axis('off')
    plt.show()

    binary = resize(binary, (1600, 2000))

    # size of blocks
    block_shape = (10, 10)

    # see astronaut as a matrix of blocks (of shape block_shape)
    view = view_as_blocks(binary, block_shape)

    # collapse the last two dimensions in one
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

    # resampling the image by taking either the `mean`,
    # the `max` or the `median` value of each blocks.
    mean_view = np.mean(flatten_view, axis=2)

    plt.title("Block view with\n local mean pooling")
    plt.imshow(mean_view, cmap=cm.Greys_r)
    plt.axis('off')  
    plt.show()

    image = mean_view
    seed = np.copy(mean_view)
    seed[1:-1, 1:-1] = image.max()
    mask = image

    #  fill holes (i.e. isolated, dark spots) in an image using morphological reconstruction by erosion
    # plt.title("Filled dark spots by morphological reconstruction")
    filled = reconstruction(seed, mask, method='erosion')
    plt.axis('off')
    plt.imshow(filled, cmap=cm.Greys_r)
    plt.savefig("enhanced_image/" + str(index) + ".jpg")
    plt.show()

    # step two for label and extract cell
    denoised = denoise_wavelet(filled, multichannel=True)
    plt.imshow(denoised, cmap=cm.Greys_r)
    plt.axis('off')
    plt.show()

    image = filled 
    coords1 = corner_peaks(corner_harris(image), min_distance=5)
    coords_subpix1 = corner_subpix(image, coords1, window_size=13)

    image = denoised
    coords2 = corner_peaks(corner_harris(image), min_distance=5)
    coords_subpix2 = corner_subpix(image, coords2, window_size=13)

    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords_subpix1[:, 1], coords_subpix1[:, 0], '+r', markersize=15)
    ax.plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r', markersize=15)
    plt.axis('off')
    plt.savefig("preprocessed_image/" + str(index) + ".jpg")
    plt.show()