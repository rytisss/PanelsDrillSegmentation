import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import DBSCAN


def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def make_output_directory(output_dir):
    if not os.path.exists(output_dir):
        print('Making output directory: ' + output_dir)
        os.makedirs(output_dir)


def get_image_name(path):
    image_name_with_ext = path.rsplit('\\', 1)[1]
    image_name, image_extension = os.path.splitext(image_name_with_ext)
    return image_name


def find_image_with_name(name, image_paths):
    paths = []
    for image_path in image_paths:
        if name in image_path:
            paths.append(image_path)
    return paths


def draw_image_boarders(image):
    width, height = image.shape[:2]
    # draw border
    border_color = 0
    # invert image
    cv2.rectangle(image, (0, 0), (height - 1, width - 1), border_color, 1)
    return image


def invert_image(image):
    return abs(255 - image)


def threshold_feature_map(image, th=0.75):
    if th < 0.0:
        th = 0.0
    if th > 1.0:
        th = 1.0
    max_val = image.max()
    limit = int(max_val * th)
    _, th_image = cv2.threshold(image, limit, 255, cv2.THRESH_BINARY)
    cv2.imshow('th', th_image)
    return th_image


def cluster_and_filter(image):
    distance_between_neighbours = 5
    Z = np.float32(image.reshape((-1, 1)))
    db = DBSCAN(eps=distance_between_neighbours, min_samples=10, metric='euclidean', algorithm='auto').fit(Z[:, :2])
    plt.imshow(np.uint8(db.labels_.reshape(image.shape[:2])))
    plt.show()


def check_conventional_methods(image_path, label_path, output_dir):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    label_name = get_image_name(label_path)

    # use conventional methods and make diagrams
    # Sobel x + y
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow('sobel', sobel)
    th_sobel = threshold_feature_map(sobel, th=0.25)
    cluster_and_filter(th_sobel)

    # laplace
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # laplacian = invert_image(laplacian)
    laplacian = cv2.convertScaleAbs(laplacian)
    th_laplacian = threshold_feature_map(laplacian, th=0.25)
    cv2.imshow('laplacian', laplacian)

    # canny edge
    canny = cv2.Canny(image, 100, 200)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)


def main():
    images_directory = r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\output\interesting parts\images/'
    label_directory = r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\output\interesting parts\labels/'

    image_paths = gather_image_from_dir(images_directory)
    label_paths = gather_image_from_dir(label_directory)

    for image_path, label_paths in zip(image_paths, label_paths):
        diagrams_output_dir = r'C:\Users\Rytis\Desktop\straipsnis\major_revision\preview/conventional_methods/'
        check_conventional_methods(image_path, label_paths, diagrams_output_dir)

    h = 0


if __name__ == '__main__':
    main()
