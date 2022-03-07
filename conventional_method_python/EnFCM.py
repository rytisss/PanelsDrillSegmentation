# https://github.com/jeongHwarr/various_FCM_segmentation/blob/master/EnFCM.py

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def get_mean_image_in_window(image, kernel):
    '''Get image consisting of mean values ​​of neighboring pixels in a window '''
    neighbor_sum = convolve2d(
        image, kernel, mode='same',
        boundary='fill', fillvalue=0)

    num_neighbor = convolve2d(
        np.ones(image.shape), kernel, mode='same',
        boundary='fill', fillvalue=0)

    return neighbor_sum / num_neighbor


class EnFCM():
    def __init__(self, image, image_bit, n_clusters, m, neighbour_effect, epsilon, max_iter, kernel_size):
        '''Modified Fuzzy C-means clustering
        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <kernel_size>: int >= 1, size of neighborhood.
        <neighbour_effect>: float, parameter which controls the influence extent of neighbouring pixels.
        <max_iter>: int, max number of iterations.
        '''

        # -------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if kernel_size <= 0 or kernel_size != int(kernel_size):
            raise Exception("<kernel_size> needs to be positive integer.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters
        self.m = m
        self.neighbour_effect = neighbour_effect
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.kernel_size = kernel_size

        self.shape = image.shape  # image shape
        self.X = image.flatten().astype('float')  # flatted image shape: (number of pixels,1)
        self.numPixels = image.size

    # ---------------------------------------------
    def initial_U(self):
        U = np.zeros((self.num_gray, self.n_clusters))
        idx = np.arange(self.num_gray)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            U[idxii, ii] = 1
        return U

    def update_U(self):
        '''Compute weights'''
        idx = np.arange(self.num_gray)
        c_mesh, idx_mesh = np.meshgrid(self.C, idx)
        power = -2. / (self.m - 1)
        numerator = abs(idx_mesh - c_mesh) ** power
        denominator = np.sum(abs(idx_mesh - c_mesh) ** power, axis=1)
        return numerator / denominator[:, None]

    def update_C(self):
        '''Compute centroid of clusters'''
        idx = np.arange(self.num_gray)
        idx_reshape = idx.reshape(len(idx), 1)
        numerator = np.sum(self.histogram * idx_reshape * pow(self.U, self.m), axis=0)
        denominator = np.sum(self.histogram * pow(self.U, self.m), axis=0)
        return numerator / denominator

    # ---------------------------------------------
    def get_filtered_image(self):

        # Create padding image
        print("Getting filtered image..")

        # mask to ignore the center pixel
        mask = np.ones((self.kernel_size, self.kernel_size))
        mask[int(self.kernel_size / 2), int(self.kernel_size / 2)] = 0

        a = self.neighbour_effect  # alpha
        mean_image = get_mean_image_in_window(self.image, mask)
        # median_image = ndimage.generic_filter(self.image, np.nanmean, footprint=mask, mode='constant', cval=np.NaN) # too slow
        filtered_image = (self.image + a * mean_image) / (1 + a)  # linearly-weighted sum image
        dtype = self.image.dtype
        self.filtered_image = filtered_image.reshape(self.shape).astype(dtype)

    def calculate_histogram(self):
        hist_max_value = (1 << self.image_bit)
        hist = cv2.calcHist([self.filtered_image], [0], None, [hist_max_value], [0, hist_max_value])
        self.num_gray = len(hist)
        self.histogram = hist

    def form_clusters(self):
        self.get_filtered_image()
        self.calculate_histogram()

        '''Iterative training'''
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" % (i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i += 1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" % (i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i += 1
        self.segmentImage()

    def deFuzzify(self):
        return np.argmax(self.U, axis=1)

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()

        self.result = np.array(self.image, copy=True)
        for i in range(len(result)):
            self.result[self.result == i] = result[i]

        self.result = self.result.reshape(self.shape).astype('int')

        return self.result