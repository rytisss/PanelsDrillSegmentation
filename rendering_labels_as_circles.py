import numpy as np
import cv2
import glob
import os


def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def get_image_name(path):
    image_name_with_ext = path.rsplit('\\', 1)[1]
    image_name, image_extension = os.path.splitext(image_name_with_ext)
    return image_name


def fit_image_to_screen(image, screen_width=1920, screen_height=1080, scale=1.0):
    """Returns resized, fit to the screen image"""
    height, width = image.shape[:2]
    width_scale = float(screen_width) / float(width)
    height_scale = float(screen_height) / float(height)
    # if image fits to desired screen size, do not resize
    if width_scale > 1.0:
        width_scale = 1.0
    if height_scale > 1.0:
        height_scale = 1.0
    image_scale = height_scale if width_scale > height_scale else width_scale
    image_scale *= scale
    resized_image = cv2.resize(image, (0, 0), fx=image_scale, fy=image_scale)
    return resized_image


def main():
    images_directory = r'C:\Users\Rytis\Desktop\straipsnis\major_revision\images_for_preview/image/'
    label_directory = r'C:\Users\Rytis\Desktop\straipsnis\major_revision\images_for_preview/label/'

    image_paths = gather_image_from_dir(images_directory)
    label_paths = gather_image_from_dir(label_directory)

    for image_path, label_paths in zip(image_paths, label_paths):
        # take out the name from the path
        name = get_image_name(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_paths, cv2.IMREAD_GRAYSCALE)
        label_fit = fit_image_to_screen(label)
        cv2.imshow('label', label_fit)
        cv2.waitKey(1)

        _, label = cv2.threshold(label, 50, 255, cv2.THRESH_BINARY)
        height, width = image.shape[:2]
        thickness = max(height // 500, width // 500)
        contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            radius = max(w, h) // 2 + 2 * thickness
            cv2.circle(image, (center_x, center_y), radius, (209, 245, 66), thickness, lineType=cv2.LINE_AA)

        image_fit = fit_image_to_screen(image)
        cv2.imshow('image', image_fit)
        cv2.waitKey(1)
        cv2.imwrite(name + '.png', image)
        g = 4

if __name__ == '__main__':
    main()
