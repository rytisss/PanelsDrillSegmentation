import cv2

from utilities import gather_image_from_dir, make_directory, get_file_name
from processing import split_image_to_tiles, crop_image_from_region

images_dir = r'C:\Users\Rytis\Desktop\straipsnis\test_full\Image/'
labels_dir = r'C:\Users\Rytis\Desktop\straipsnis\test_full\Label/'

output_dir = 'output/'

def crop():
    images = gather_image_from_dir(images_dir)
    labels = gather_image_from_dir(labels_dir)
    for data_item in zip(images, labels):
        image = cv2.imread(data_item[0], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(data_item[1], cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[:2]
        rois = split_image_to_tiles(width, height, 320, 320, 160, 160)
        for roi in rois:
            image_crop = crop_image_from_region(image, roi)
            label_crop = crop_image_from_region(label, roi)
            name = get_file_name(data_item[0])
            output_name = name + '_' + str(roi[0]) + '_' + str(roi[1]) + '.png'
            label_output = output_dir + 'label/'
            image_output = output_dir + 'image/'
            make_directory(label_output)
            make_directory(image_output)
            cv2.imwrite(image_output + output_name, image_crop)
            cv2.imwrite(label_output + output_name, label_crop)
            cv2.imshow('image', image_crop)
            cv2.imshow('label', label_crop)
            cv2.waitKey(1)

if __name__ == '__main__':
    crop()
