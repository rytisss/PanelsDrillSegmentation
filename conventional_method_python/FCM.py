# https://github.com/jeongHwarr/various_FCM_segmentation/blob/master/FCM.py

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class FCM():
    def __init__(self, image, image_bit, n_clusters, m, epsilon, max_iter):
        '''Modified Fuzzy C-means clustering
        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <max_iter>: int, max number of iterations.
        '''

        # -------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.shape = image.shape  # image shape
        self.X = image.flatten().astype('float')  # flatted image shape: (number of pixels,1)
        self.numPixels = image.size

    # ---------------------------------------------
    def initial_U(self):
        U = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            U[idxii, ii] = 1
        return U

    def update_U(self):
        '''Compute weights'''
        c_mesh, idx_mesh = np.meshgrid(self.C, self.X)
        power = 2. / (self.m - 1)
        p1 = abs(idx_mesh - c_mesh) ** power
        p2 = np.sum((1. / abs(idx_mesh - c_mesh)) ** power, axis=1)

        return 1. / (p1 * p2[:, None])

    def update_C(self):
        '''Compute centroid of clusters'''
        numerator = np.dot(self.X, self.U ** self.m)
        denominator = np.sum(self.U ** self.m, axis=0)
        return numerator / denominator

    def form_clusters(self):
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
        self.result = result.reshape(self.shape).astype('int')

        return self.result
