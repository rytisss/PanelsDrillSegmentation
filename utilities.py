import glob
import os


def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def get_file_name_with_ext(path: str):
    file_name_with_ext = os.path.basename(path)
    return file_name_with_ext


def get_file_name(path: str):
    file_name_with_ext = os.path.basename(path)
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    return file_name


def make_directory(output_dir):
    if not os.path.exists(output_dir):
        print('Making output directory: ' + output_dir)
        os.makedirs(output_dir)
