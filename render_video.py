import predict
import tensorflow as tf
import glob
import os
import cv2
import numpy as np
from models.losses import Loss
import processing
from models.autoencoder import unet_autoencoder


def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def get_file_name(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    return file_name


def get_first_weight_in_dir(dir):
    weights = glob.glob(dir + '*.hdf5')
    return None if len(weights) == 0 else weights[0]

def predict_image(model, image):
    tensor = processing.image_to_tensor(image)
    prediction = model.predict(tensor)
    prediction_image = processing.tensor_to_image(prediction)
    return prediction_image

def render_videos():
    # Output path
    output_path = r'C:\Users\Rytis\Desktop\straipsnis\video_output//'

    # Tile/patch/region size
    input_size = (320, 320)

    networks = ['UNet', 'UNet_SE', 'UNet_CoordConv', 'UNet_CoordConv_SE', 'UNet_Res_ASPP', 'UNet_Res_ASPP_SE',
                'UNet_Res_ASPP_CoordConv', 'UNet_Res_ASPP_CoordConv_SE']

    for network in networks:
        model = None
        tf.keras.backend.clear_session()
        print('Deploying ' + network)
        if network == 'UNet':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4/')
            model = unet_autoencoder(pretrained_weights=weights_path)
        elif network == 'UNet_SE':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_se=True)
        elif network == 'UNet_CoordConv':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_coord/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_coord_conv=True)
        elif network == 'UNet_CoordConv_SE':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_coord_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_coord_conv=True, use_se=True)
        elif network == 'UNet_Res_ASPP':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_res_aspp/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True)
        elif network == 'UNet_Res_ASPP_SE':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_aspp_res_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_se=True)
        elif network == 'UNet_Res_ASPP_CoordConv':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_res_aspp_coord/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_coord_conv=True)
        elif network == 'UNet_Res_ASPP_CoordConv_SE':
            weights_path = get_first_weight_in_dir(r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_aspp_res_aspp_coord_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_coord_conv=True, use_se=True)

        overlay = 240
        tile_width = 320
        tile_height = 320

        images_path = r'C:\Users\Rytis\Desktop\straipsnis\test\Data_with_gamma_correction\Image/'
        images = gather_image_from_dir(images_path)

        labels_path = r'C:\Users\Rytis\Desktop\straipsnis\test\Data_with_gamma_correction\Label/'
        labels = gather_image_from_dir(labels_path)

        for data_item in zip(images, labels):
            image_path = data_item[0]
            label_path = data_item[1]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            height, width = image.shape[:2]
            regions = processing.split_image_to_tiles(width, height, tile_width, tile_height, overlay, overlay)

            for region in regions:
                image_crop = processing.crop_image_from_region(image, region)
                label_crop = processing.crop_image_from_region(label, region)

                prediction = predict_image(model, image_crop)

                cv2.imshow('image', image_crop)
                cv2.imshow('label', label_crop)
                cv2.imshow('prediction', prediction)

                cv2.waitKey(1)



if __name__ == '__main__':
    render_videos()
