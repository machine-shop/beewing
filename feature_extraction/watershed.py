from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plot
from skimage.util import invert
import skimage

bee_image = skimage.io.imread("raw_image/2.jpg")
inverted_bee_image = invert(bee_image)
skimage.io.imshow(inverted_bee_image)
