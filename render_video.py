import tensorflow as tf

"""physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)"""

import predict

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
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4/')
            model = unet_autoencoder(pretrained_weights=weights_path)
        elif network == 'UNet_SE':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_se=True)
        elif network == 'UNet_CoordConv':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_coord/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_coord_conv=True)
        elif network == 'UNet_CoordConv_SE':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_coord_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_coord_conv=True, use_se=True)
        elif network == 'UNet_Res_ASPP':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_res_aspp/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True)
        elif network == 'UNet_Res_ASPP_SE':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_aspp_res_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_se=True)
        elif network == 'UNet_Res_ASPP_CoordConv':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_res_aspp_coord/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_coord_conv=True)
        elif network == 'UNet_Res_ASPP_CoordConv_SE':
            weights_path = get_first_weight_in_dir(
                r'C:\Users\Rytis\Desktop\straipsnis\dataForTraining_v3_only_epoch\best_weights\UNet4_aspp_res_aspp_coord_SE/')
            model = unet_autoencoder(pretrained_weights=weights_path, use_residual_connections=True,
                                     use_aspp=True, use_coord_conv=True, use_se=True)

        overlay = 160
        tile_width = 320
        tile_height = 320

        images_path = r'C:\Users\Rytis\Desktop\straipsnis\test\Data_with_gamma_correction\Image/'
        images = gather_image_from_dir(images_path)

        labels_path = r'C:\Users\Rytis\Desktop\straipsnis\test\Data_with_gamma_correction\Label/'
        labels = gather_image_from_dir(labels_path)

        # create a video
        # make video
        video_frame_width = 1920
        video_frame_height = 1080

        screen_width = 1000
        screen_height = 1000

        video = cv2.VideoWriter(output_path + network + '.avi',
                                cv2.VideoWriter_fourcc(*'DIVX'), 30, (video_frame_width, video_frame_height))

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

                """ cv2.imshow('image', image_crop)
                cv2.imshow('label', label_crop)
                cv2.imshow('prediction', prediction)"""

                resized_image, roi, ratio = processing.resize_and_insert(image, screen_width, screen_height)

                roi_x = region[0]
                roi_y = region[1]
                roi_w = region[2] - roi_x
                roi_h = region[3] - roi_y

                color = (235,186,52)

                # render region on image
                imageBGR = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
                roi_top_left = ((int)(roi_x * ratio + roi[0]), (int)(roi_y * ratio + roi[1]))
                roi_bottom_right = ((int)((roi_x + roi_w) * ratio + roi[0]), (int)((roi_y + roi_h) * ratio + roi[1]))
                cv2.rectangle(imageBGR, roi_top_left, roi_bottom_right, color, 2)
                """cv2.imshow('region', imageBGR)
                cv2.waitKey(1)"""

                # make video output frame with every image moded9
                frame_image = np.zeros((video_frame_height, video_frame_width, 3), np.uint8)
                # calculate top left corner of main image in the frame
                resized_image_width = resized_image.shape[1]
                resized_image_height = resized_image.shape[0]
                # take 1400 pixels width
                image_topLeft = (
                    (int)((1400 - resized_image_width) / 2), (int)((video_frame_height - resized_image_height) / 2))
                frame_image[image_topLeft[1]:image_topLeft[1] + resized_image_height,
                image_topLeft[0]:image_topLeft[0] + resized_image_width] = imageBGR
                # draw rectangle
                cv2.rectangle(frame_image, image_topLeft,
                              (image_topLeft[0] + resized_image_width, image_topLeft[1] + resized_image_height),
                              (10, 150, 10), 2)
                cv2.rectangle(frame_image, (0, 0), (1400, video_frame_height),
                              (150, 50, 10), 5)

                # crop images place
                top_offset = (int)((video_frame_height - screen_height) / 2)
                original_image_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2 - 100),
                                           top_offset + (int)((320 - input_size[1]) / 2))
                crop_imageBGR = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
                frame_image[original_image_top_left[1]:original_image_top_left[1] + input_size[1],
                original_image_top_left[0]:original_image_top_left[0] + input_size[0]] = crop_imageBGR
                cv2.rectangle(frame_image, original_image_top_left,
                              (original_image_top_left[0] + input_size[0], original_image_top_left[1] + input_size[1]),
                              color, 2)

                label_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2 - 100),
                                  340 + top_offset + (int)((320 - input_size[1]) / 2))
                crop_labelBGR = cv2.cvtColor(label_crop, cv2.COLOR_GRAY2RGB)
                frame_image[label_top_left[1]:label_top_left[1] + input_size[1],
                label_top_left[0]:label_top_left[0] + input_size[0]] = crop_labelBGR
                cv2.rectangle(frame_image, label_top_left, (
                    label_top_left[0] + input_size[0], label_top_left[1] + input_size[1]),
                              color, 2)

                prediction_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2 - 100),
                                       680 + top_offset + (int)((320 - input_size[1]) / 2))
                # make prediction rgb
                wo_coord_conv_pred_RGB = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
                frame_image[prediction_top_left[1]:prediction_top_left[1] + input_size[1],
                prediction_top_left[0]:prediction_top_left[0] + input_size[0]] = wo_coord_conv_pred_RGB
                cv2.rectangle(frame_image, prediction_top_left, (
                    prediction_top_left[0] + input_size[0], prediction_top_left[1] + input_size[1]),
                              color, 2)

                """prediction_coordConv_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2 - 100),
                                                 750 + top_offset + (int)((250 - input_size[1]) / 2))
                # make prediction rgb
                w_coord_conv_pred_RGB = cv2.cvtColor(w_coord_conv_pred, cv2.COLOR_GRAY2RGB)
                frame_image[prediction_coordConv_top_left[1]:prediction_coordConv_top_left[1] + input_size[1],
                prediction_coordConv_top_left[0]:prediction_coordConv_top_left[0] + input_size[
                    0]] = w_coord_conv_pred_RGB
                cv2.rectangle(frame_image, prediction_coordConv_top_left, (
                    prediction_coordConv_top_left[0] + input_size[0], prediction_coordConv_top_left[1] + input_size[1]),
                              (20, 220, 20), 2)"""


                # draw line connecting cropped regions
                cv2.line(frame_image, (roi_bottom_right[0] + image_topLeft[0], roi_top_left[1] + image_topLeft[1]),
                         original_image_top_left, color, 1, cv2.LINE_AA)
                cv2.line(frame_image, (roi_bottom_right[0] + image_topLeft[0], roi_bottom_right[1] + image_topLeft[1]),
                         (original_image_top_left[0], original_image_top_left[1] + input_size[1]), color, 1,
                         cv2.LINE_AA)

                text_x_offset = 10

                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 0.7
                font_thickness = 1

                text = 'Image'
                (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                pt = (original_image_top_left[0] + input_size[0] + text_x_offset,
                      (int)(original_image_top_left[1] + ((input_size[0] + label_height) / 2)))
                cv2.putText(frame_image, text, pt, font, font_scale, color,
                            font_thickness)

                text = 'Label'
                (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                pt = (label_top_left[0] + input_size[0] + text_x_offset,
                      (int)(label_top_left[1] + ((input_size[0] + label_height) / 2)))
                cv2.putText(frame_image, text, pt, font, font_scale, color,
                            font_thickness)

                text = 'Prediction'
                (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                pt = (prediction_top_left[0] + input_size[0] + text_x_offset,
                      (int)(prediction_top_left[1] + ((input_size[0] + label_height) / 2)))
                cv2.putText(frame_image, text, pt, font, font_scale, color,
                            font_thickness)

                #cv2.imshow('video_frame', frame_image)
                #cv2.waitKey(1)

                for i in range(0, 7):  # seven frame
                    video.write(frame_image)

            video.release()


if __name__ == '__main__':
    render_videos()
