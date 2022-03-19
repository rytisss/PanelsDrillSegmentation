import os
import glob
import cv2
import numpy as np
from evaluation.statistics import Statistics
from utilities import get_file_name, gather_image_from_dir, get_file_name_with_ext, make_directory
from processing import split_image_to_tiles, crop_image_from_region

# image dir (contains images)
image_dir = r'D:\straipsniai\straipsnis\test\Data_with_gamma_correction\Image/'

train_image_dir = r'D:\straipsniai\straipsnis\train\Data_with_gamma_correction\Image/'

# label dir (contains images)
label_dir = r'D:\straipsniai\straipsnis\test\Data_with_gamma_correction\Label/'

train_label_dir = r'D:\straipsniai\straipsnis\train\Data_with_gamma_correction\Label/'

# prediction dir (contains folder in which are images)
prediction_dir = r'F:\sensors drilled holes\major_revision_v2\conventional_method_cpp_build\Laplace_th/'

# output
output_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop/'


def make_test():
    full_images = gather_image_from_dir(image_dir)
    full_images.extend(gather_image_from_dir(train_image_dir))

    full_labels = gather_image_from_dir(label_dir)
    full_labels.extend(gather_image_from_dir(train_label_dir))

    predictions = gather_image_from_dir(prediction_dir)

    crop_image_dir = output_dir + 'image/'
    crop_label_dir = output_dir + 'label/'
    make_directory(crop_image_dir)
    make_directory(crop_label_dir)

    for prediction in predictions:
        full_image_found = False
        full_label_found = False

        prediction_name = get_file_name(prediction)
        prediction_name = get_file_name(prediction_name)
        prediction_name_parts = prediction_name.split(sep='_')
        name_parts = prediction_name_parts[0:-2]
        formed_name = ''
        for i, name_part in enumerate(name_parts):
            formed_name += name_part
            if i < (len(name_parts) - 1):
                formed_name += '_'
        x = int(prediction_name_parts[-2])
        y = int(prediction_name_parts[-1])
        print(f'{formed_name}_{x}_{y}')

        cooresponding_full_image_path = ''
        for full_image_path in full_images:
            full_image_name = get_file_name(full_image_path)
            if full_image_name == formed_name:
                cooresponding_full_image_path = full_image_path
                full_image_found = True
                break

        cooresponding_full_label_path = ''
        for full_label_path in full_labels:
            full_label_name = get_file_name(full_label_path)
            if full_label_name == formed_name:
                cooresponding_full_label_path = full_label_path
                full_label_found = True
                break

        if not (full_image_found or full_label_found):
            print(f'{formed_name} not found')

        full_image = cv2.imread(cooresponding_full_image_path, cv2.IMREAD_GRAYSCALE)
        full_label = cv2.imread(cooresponding_full_label_path, cv2.IMREAD_GRAYSCALE)

        image_crop = crop_image_from_region(full_image, [x, y, x + 320, y + 320])
        label_crop = crop_image_from_region(full_label, [x, y, x + 320, y + 320])

        cv2.imwrite(crop_image_dir + prediction_name + '.png', image_crop)
        cv2.imwrite(crop_label_dir + prediction_name + '.png', label_crop)


if __name__ == "__main__":
    make_test()
