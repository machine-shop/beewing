import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import matplotlib.cm as cm
from skimage import color
from skimage.util import view_as_blocks
from skimage.transform import resize
from skimage.morphology import reconstruction
from os import listdir

"""
Function for preprocessing images

Put raw image in the folder path one level above the current image 
"""
def preprocess_raw_image(path):
    files = listdir()
    for f in files:
        image = mpimg.imread(path + f)
        img_gray = rgb2gray(image)
        thresh = threshold_otsu(img_gray)
        binary = img_gray > thresh

        # step one for otsu thresholding with appropriate nbins number 
        thresh = threshold_otsu(img_gray, nbins = 60)
        binary = img_gray > thresh
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

        image = mean_view
        seed = np.copy(mean_view)
        seed[1:-1, 1:-1] = image.max()
        mask = image

        #  fill holes (i.e. isolated, dark spots) in an image using morphological reconstruction by erosion
        filled = reconstruction(seed, mask, method='erosion')
        plt.figure()
        plt.axis('off')
        plt.imshow(filled, cmap=cm.Greys_r)
        plt.savefig("preprocessed_image/" + f)