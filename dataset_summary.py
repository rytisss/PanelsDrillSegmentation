import numpy as np
import cv2
import os
import glob
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html

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
    labelsDir = r'C:\Users\Rytis\Desktop\hole detection\drilled holes data for training\freda holes data 2020-08-24\Label_rois_/'
    labels = GatherImagesFromDirectory(labelsDir)
    sum = np.zeros([320, 320], dtype=np.float64)

    count_only_positive = True

    sample_count = 150#len(labels)
    counter = 0
    for i in range(0, sample_count):
        image = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image_float64 = image.astype(np.float64)
        image_float64 /= 255.0
        sum += image_float64
        if count_only_positive:
            non_zero_pixels = cv2.countNonZero(image)
            if non_zero_pixels > 0:
                counter += 1
        else:
            counter+=1

    norm_sum = np.zeros([320, 320], dtype=np.float64)
    #cv2.normalize(sum, norm_sum, 0.0, 255.0, cv2.NORM_MINMAX)

    sum /= counter
    sum *= 100.0 # to percents
    max_value = np.amax(sum)
    min_value = np.amin(sum)

    tick_values = []
    tick_labels = []
    range_len = max_value - min_value
    tick_distance = range_len / 5.0
    tickCount = 5
    for i in range(0, tickCount + 1):
        value = min_value + (float)(i) * tick_distance
        tick_values.append(value)
        value = float("{:.2f}".format(value))
        tick_labels.append(str(value) + "%")

    # make colormaps
    viridis = cm.get_cmap('viridis', 256)
    tenPercentColors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    tenPercentColors[:25, :] = pink
    tenPercentCmp = ListedColormap(tenPercentColors)

    twentyFivePercentColors = viridis(np.linspace(0, 1, 256))
    twentyFivePercentColors[:64, :] = pink
    twentyFivePercentCmp = ListedColormap(twentyFivePercentColors)

    fiftyFivePercentColors = viridis(np.linspace(0, 1, 256))
    fiftyFivePercentColors[:128, :] = pink
    fiftyFivePercentCmp = ListedColormap(fiftyFivePercentColors)

    cms = [viridis, tenPercentCmp, twentyFivePercentCmp, fiftyFivePercentCmp]

    print("Min value " + str(min_value))
    print("Max value " + str(max_value))

    fig, axs = plt.subplots(1, 4, figsize=(16, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(sum, cmap=cmap, rasterized=True, vmin=min_value, vmax=max_value)
        #ax.set_yticklabels(tick_labels)  # vertically oriented colorb
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(psm, ax=ax, cax=cax, ticks=tick_values, shrink=1.0)
        cbar.ax.set_yticklabels(tick_labels)
        ax.set_aspect(aspect=1.0)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end+1, 80))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end+1, 80))
    #fig.tight_layout()
    fig.suptitle("Labels Overlay Percentage in Each Image Pixel")
    plt.show()

    #cax = ax.imshow(sum, vmin=min_value, vmax=max_value)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #cbar = fig.colorbar(cax, ticks=tick_values)
    #cbar.ax.set_yticklabels(tick_labels)  # vertically oriented colorb

    #plt.colorbar()
    #plt.show()
    norm_sum = sum.astype(np.uint8)
    print('end!')

if __name__ == "__main__":
    main()