import numpy as np
import cv2
import os
import glob

def GatherImagesFromDirectory(directory):
    # define supproted types
    types = ['*.png', '*.bmp', '*.jpg']
    # define image list
    imageList = []
    # check if directory exist
    if not os.path.exists(directory):
        print('Directory doesn\'t exist, return empty image path array')
        return imageList #exit method here
    # gather images by types
    for type in types:
        print('Gathering type: ' + type)
        specificTypeImages = glob.glob(directory + type)
        # add one type images list to main array
        imageList.extend(specificTypeImages)
    return imageList

def main():
    labelsDir = r'C:\Users\Rytis\Desktop\drilled holes data for training\freda holes data 2020-08-24\Label_rois_/'
    labels = GatherImagesFromDirectory(labelsDir)

    sum = np.zeros([320, 320], dtype=np.float64)

    #for i in range(0, len(labels)):
    for i in range(0, 10000):
        image = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image_float64 = image.astype(np.float64)
        image_float64 /= 255.0
        sum += image_float64

    norm_sum = np.zeros([320, 320], dtype=np.float64)
    cv2.normalize(sum, norm_sum, 0.0, 255.0, cv2.NORM_MINMAX)
    norm_sum.astype(np.uint8)
    cv2.imshow("sum", norm_sum)
    cv2.waitKey(0)
    print('end!')

if __name__ == "__main__":
    main()