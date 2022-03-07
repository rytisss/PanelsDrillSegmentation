from sklearn.cluster import KMeans
import cv2
import numpy as np


def kmeans(image, n_clusters=2, iteration=100):
    # for greyscale
    pixel_values = image.reshape((-1, 1))
    # convert to float
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, n_clusters, None, criteria, iteration, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = np.uint8(labels.reshape(image.shape))
    segmented_image[segmented_image == 1] = 255 # make class one as pixel with 255 intensity in 8 bit depth
    return segmented_image
