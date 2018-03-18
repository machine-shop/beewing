import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import csv
import os

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.io import imread


class wing_photo():
    def __init__(self, file_name):
        """Binds the object to a two dimensional array"""
        self.information = file_name.split(' ')
        self.species = self.information[1]
        self.image = imread(file_name)[:,:,0]

    def identify_region(self, min_area = 0):
        """Approximates the regions in the image and adds attributes to the image object
        Takes in MIN_AREA as the cutoff for a region to be relevant enough to analyze
        These attributes include the number of regions, the area of all regions, 
        and the mean of the area of regions. Returns the feature vector"""
        #apply threshold
        thresh = threshold_otsu(self.image)
        bw = closing(self.image > thresh, square(3))

        #remove artifacts connected to image border
        cleared = clear_border(bw)

        #label image regions
        label_image = label(cleared)

        #Gets areas that exceed MIN_AREA
        self.num_regions, self.areas = 0, []
        for region in regionprops(label_image):
            #take regions with large enough areas
            if region.area >= min_area:
                self.areas.append(region.area)
                self.num_regions += 1

        #Calculates features
        if self.num_regions == 0:
            self.mean_area = 0
            self.top = 0
        else:
            self.mean_area = np.mean(self.areas)
            self.areas.sort(reverse=True)
            self.top = self.areas[0]

        #final vector of things to be returned
        self.feature_vector = [self.num_regions, self.mean_area, self.top, self.species]
        return self.feature_vector

def image_file_scanner(file_name):
    """takes the location of a file on a computer and returns a list of image paths"""
    img_list = []
    for img in os.scandir(file_name):
        img_list += [img.path]
    return img_list

def proprocess_image(image_path):
    files = listdir(image_path)
    for f in files:
        image = mpimg.imread(image_path + f)
        
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
        # plt.title("Filled dark spots by morphological reconstruction")
        filled = reconstruction(seed, mask, method='erosion')
        plt.figure()
        plt.axis('off')
        plt.imshow(filled, cmap=cm.Greys_r)
        plt.savefig("enhanced_image/" + f)

        # step two for label and extract cell
        denoised = denoise_wavelet(filled, multichannel=True)

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
        plt.savefig(path + "preprocessed_image/" + f)

def create_csv(csv_name, img_list):
    """Takes in a list of images, creates a csv (CSV_NAME)
    where each row is a feature vector for the corresponding image"""
    data = [[]]
    for img in img_list:
        #creates photo_class, runs identify_region, appends the list to the data list
        data.append(wing_photo(img).identify_region())

    #writes the data to a csv
    myFile = open(csv_name + '.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)
