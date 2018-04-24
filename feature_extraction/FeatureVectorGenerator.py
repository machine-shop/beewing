from skimage.io import imshow, imread
from skimage.filters import rank
from skimage.morphology import disk, watershed
from skimage import segmentation
from scipy import ndimage as ndi
from skimage.measure import regionprops
from scipy.spatial import distance
import numpy as np
import csv
import os

def watershed_segmentation(img):
    """Takes an image as input
    Returns the segmented image"""
    denoised = rank.median(img, disk(2))
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(2))
    labels = watershed(gradient, markers)
    return segmentation.clear_border(labels)

def filter_regions(regions, min, max):
    """Takes in a list of regions, the output from regionprops
    Filters back the ones that have area within the given range"""
    filtered_regions = []
    for region in regions:
        area = region.area
        if area > min and area < max:
            filtered_regions.append(region)
    return filtered_regions

def central_cell_finder(filtered_regions):
    """Takes in a list of filtered regions
    Outputs the cell that is determined to be the center of these regions"""
    centroids, centroids_x, centroids_y, filtered_regions_area, weighted_values = [], [], [], [], []
    distance_weight, size_weight = 1, 1

    for region in filtered_regions:
        filtered_regions_area.append(region.area)
        y0, x0 = region.centroid
        region_centroid = (x0, y0)
        centroids.append(region_centroid)
        centroids_x.append(x0)
        centroids_y.append(y0)

    avg_x = np.mean(centroids_x)
    avg_y = np.mean(centroids_y)

    distances_from_center = []
    for centroid in centroids:
        distances_from_center.append(distance.euclidean(centroid, (avg_x, avg_y)))

    distance_sd = np.std(distances_from_center)
    distance_mean = np.mean(distances_from_center)

    dist_from_center_norm = [dist_from_center / distance_mean for dist_from_center in distances_from_center]
    area_sd = np.std(filtered_regions_area)
    area_mean = np.mean(filtered_regions_area)
    area_norm = [area / area_mean for area in filtered_regions_area]

    for i in range(len(filtered_regions)):
        weighted_values.append(size_weight * area_norm[i] - distance_weight * dist_from_center_norm[i])

    max_index = weighted_values.index(max(weighted_values))
    return filtered_regions[max_index]

def neighbor_finder(region, regions):
    """Takes in a REGION and a list of REGIONS
    Returns which REGIONS are neighbors of the REGION"""
    surrounding_regions = set()
    for coord in region.coords:
        for i in range(-2, 3):
            for j in range(-2, 3):
                surrounding_regions.add((coord[0] + i, coord[1] + j))
    region_tuple = [tuple(k) for k in region.coords]
    for element in region_tuple:
        surrounding_regions.remove(element)

    neighbors = []
    for region in regions:
        region_tuple = [tuple(k) for k in region.coords]
        if any(([coord in surrounding_regions for coord in region_tuple])):
            neighbors.append(region)
    return neighbors


def feature_vector_generator(path):
    """Takes in the path of an enhanced image
    Returns a vector of features for classification"""
    img = imread(path)[:,:,0]
    species = [path.split()[1]]
    labeled_wing = watershed_segmentation(img)

    regions = filter_regions(regionprops(labeled_wing), 1000, 400000)
    central_cell = central_cell_finder(regions)
    neighbors = neighbor_finder(central_cell, regions)

    star_features = [central_cell.area, central_cell.eccentricity]
    adjacent_features = [len(neighbors), np.mean([region.area for region in neighbors]), np.std([region.area for region in neighbors])]
    return star_features + adjacent_features + species
    #Center cell area, eccentricity of center cell, number of neighbors, mean of neighbor area, std of neighbor area

def main():
    i = 0
    directory, data = '/home/jon-ross/Desktop/bee-wing/data/enhanced_image', []
    for filename in os.listdir(directory):
        percent = round((i / 938), 3) * 100
        i += 1
        print(str(percent) + '%')
        if filename != '.DS_Store':
            data.append(feature_vector_generator(directory + '/' + filename))
    myFile = open('bee_wing_features.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)
    print("Writing Complete")

main()
