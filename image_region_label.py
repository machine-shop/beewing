import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import csv

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.io import imread


#img = imread('C://Users//Jon-Ross//Desktop//bees//pictures//001 Osmia lignaria m right 4x.jpg')
class wing_photo():
    def __init__(self, file_name):
        """Binds the object to a two dimensional array"""
        self.image = imread(file_name)[:,:,0]

    def identify_region(self, min_area = 100):
        """Approximates the regions in the image and adds attributes to the image object
        Takes in MIN_AREA as the cutoff for a region to be relevant enough to analyze
        These attributes include the number of regions, the area of all regions, the top five largest areas
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
        self.mean_area = np.mean(self.areas)
        self.areas.sort(reverse=True)
        self.top5 = self.areas[:5]

        #final vector of things to be returned
        self.feature_vector = [self.num_regions, self.areas, self.mean_area]
        return self.feature_vector

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
