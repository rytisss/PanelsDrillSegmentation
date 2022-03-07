import cv2
import numpy as np
from utilities import gather_image_from_dir, get_file_name
from conventional_methods import kmeans
from EnFCM import EnFCM
from FCM import FCM
from MFCM import MFCM

# Test images directory
test_images = '../data/image/'


def predict():
    image_paths = gather_image_from_dir(test_images)
    # Load and predict on all images from directory
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # kmeans
        kmeans_image = kmeans(image)
        cv2.imshow('kmeans', kmeans_image)
        # Enhanced Fuzzy C-Means Algorithm
        cluster = EnFCM(image, image_bit=8, n_clusters=2, m=2,
                        neighbour_effect=0.2, epsilon=0.1, max_iter=100,
                        kernel_size=3)
        cluster.form_clusters()
        result = np.uint8(cluster.result)
        result[result == 1] = 255
        cv2.imshow('Enhanced Fuzzy C-Means Algorithm', result)
        # Standard Fuzzy C-Means Algorithm
        cluster = FCM(image, image_bit=8, n_clusters=2, m=2, epsilon=0.02,
                      max_iter=100)
        cluster.form_clusters()
        result = np.uint8(cluster.result)
        result[result == 1] = 255
        cv2.imshow('Fuzzy C-Means Algorithm', result)
        # Modified Fuzzy C-Means Algorithm
        cluster = MFCM(image, image_bit=8, n_clusters=2, m=2,
                        neighbour_effect=0.2, epsilon=0.1, max_iter=100,
                        kernel_size=3)

        # OTSU
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        cv2.imshow('OTSU', otsu)
        adaptive_th_meanc = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 15, 2)
        cv2.imshow('adaptive_th_meanc', adaptive_th_meanc)
        adaptive_th_gaussianc = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 15, 2)
        cv2.imshow('adaptive_th_gaussianc', adaptive_th_gaussianc)

        # TODO

        #https: // scikit - image.org / docs / dev / auto_examples / developers / plot_threshold_li.html

        cv2.imshow("image", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    predict()
