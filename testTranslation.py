from RoadPavementSegmentation.models.utilities import *
import glob
from RoadPavementSegmentation.models.autoencoder import *
from RoadPavementSegmentation.preprocessing.crop_to_tiles import *
import cv2
import tensorflow as tf
import os

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

def predict_image(model, image):
    # Preprocess
    image_norm = image / 255
    image_norm = np.reshape(image_norm, image_norm.shape + (1,))
    image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
    # Predict
    prediction_norm = model.predict(image_norm)
    # Normalize to image
    prediction = prediction_norm[0, :, :, 0]
    prediction *= 255
    prediction = prediction.astype(np.uint8)
    return prediction

def find_holes_center_point(label):
    contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for i in range(0, len(contours)):
        boundRect = cv2.boundingRect(contours[i])
        bbox_x = boundRect[0]
        bbox_y = boundRect[1]
        bbox_w = boundRect[2]
        bbox_h = boundRect[3]
        # filter out rubbish that may occur in jpeg
        if bbox_w > 4 and bbox_h > 4:
            center_pt = (bbox_x + (int)(bbox_w / 2), bbox_y + (int)(bbox_h / 2))
            centers.append(center_pt)
    return centers

def predict_by_patches():
    # Output path
    output_path = r'D:\drilled holes data for training\UNet4_res_assp_5x5_16k_320x320_leaky//'

    # Tile/patch/region size
    input_size = (224, 224)

    # First model without coordconv
    # Weights *.hdf5 file path
    path_to_weight = r'C:\Users\Rytis\Desktop\hole detection\UNet4_res_assp_5x5_16k_320x320_leaky\UNet4_res_assp_5x5_16k_320x320_leaky/UNet4_res_assp_5x5_16k_320x320-010-0.0785.hdf5'
    # Choose your 'super-model'
    model = UNet4_res_aspp_First5x5(pretrained_weights=path_to_weight, number_of_kernels=16, input_size=(320, 320, 1),
                                  loss_function=Loss.CROSSENTROPY50DICE50, useLeakyReLU=True)

    # Second model with coordconv
    # Weights *.hdf5 file path
    path_to_weight_with_coordConv = r'C:\Users\Rytis\Desktop\hole detection\UNet4_res_assp_5x5_coord_16k_320x320_leaky_bug_fig\UNet4_res_assp_5x5_coord_16k_320x320_leaky_bug_fig/UNet4_res_assp_5x5_16k_320x320_coord-010-0.0782.hdf5'
    # Choose your 'super-model'
    model_with_coordConv = UNet4_res_aspp_First5x5_CoordConv(pretrained_weights=path_to_weight_with_coordConv, number_of_kernels=16, input_size=(320, 320, 1),
                                    loss_function=Loss.CROSSENTROPY50DICE50, useLeakyReLU=True)

    # Test images directory
    test_images = r'C:\Users\Rytis\Desktop\hole detection\drilled holes test/Image/'
    test_labels = r'C:\Users\Rytis\Desktop\hole detection\drilled holes test/Label/'

    image_paths = gather_image_from_dir(test_images)
    label_paths = gather_image_from_dir(test_labels)

    video_frame_width = 1920
    video_frame_height = 1080

    for i in range(0, len(image_paths)):
        image_path = image_paths[i]
        label_path = label_paths[i]
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # remove trash in jpeg
        _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

        holes_centers = find_holes_center_point(label)

        width = image.shape[1]
        height = image.shape[0]

        # define screen or desired max size
        screen_width = 1000
        screen_height = 1000
        # resize image to fit it to the screen
        width_ratio = (float)(width) / (float)(screen_width)
        height_ratio = (float)(height) / (float)(screen_height)
        ratio = 1.0
        if width_ratio > height_ratio:
            ratio = width_ratio
        else:
            ratio = height_ratio
        resized_image = cv2.resize(image, (0,0), dst=None, fx = 1.0/ratio, fy = 1.0/ratio)

        # processes each hole with translation
        for hole_center in holes_centers:
            x_step = 16
            y_step = 16
            y = 0
            while y <= input_size[1]:
                x = 0
                while x <= input_size[0]:
                    roi_x = hole_center[0] - x
                    roi_y = hole_center[1] - y
                    roi_w = input_size[0]
                    roi_h = input_size[1]
                    roi_image = cropImageFromRegion(image, (roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
                    roi_label = cropImageFromRegion(label, (roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
                    show_roi = True
                    if show_roi:
                        cv2.imshow('roi_image', roi_image)
                        cv2.imshow('roi_label', roi_label)
                        cv2.waitKey(1)
                    x += x_step
                    # render region on image
                    imageBGR = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
                    roi_top_left = ((int)(roi_x * 1.0/ratio), (int)(roi_y * 1.0/ratio))
                    roi_bottom_right = ((int)((roi_x+roi_w) * 1.0 / ratio), (int)((roi_y+roi_h) * 1.0 / ratio))
                    cv2.rectangle(imageBGR, roi_top_left, roi_bottom_right, (20,220,20), 2)
                    cv2.imshow('region', imageBGR)
                    cv2.waitKey(1)

                    # make video output frame with every image moded9
                    frame_image = np.zeros((video_frame_height, video_frame_width, 3), np.uint8)
                    # calculate top left corner of main image in the frame
                    resized_image_width = resized_image.shape[1]
                    resized_image_height = resized_image.shape[0]
                    # take 1400 pixels width
                    image_topLeft = ((int)((1400 - resized_image_width) / 2), (int)((video_frame_height - resized_image_height) / 2))
                    frame_image[image_topLeft[1]:image_topLeft[1]+resized_image_height, image_topLeft[0]:image_topLeft[0]+resized_image_width] = imageBGR

                    # crop images place
                    top_offset = (int)((video_frame_height - screen_height) / 2)
                    original_image_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2), top_offset + (int)((250 - input_size[1]) / 2))
                    crop_imageBGR = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
                    frame_image[original_image_top_left[1]:original_image_top_left[1] + input_size[1], original_image_top_left[0]:original_image_top_left[0] + input_size[0]] = crop_imageBGR
                    cv2.rectangle(frame_image, original_image_top_left, (original_image_top_left[0]+input_size[0], original_image_top_left[1]+input_size[1]), (20, 220, 20), 2)

                    label_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2), 250 + top_offset + (int)((250 - input_size[1]) / 2))
                    crop_labelBGR = cv2.cvtColor(roi_label, cv2.COLOR_GRAY2RGB)
                    frame_image[label_top_left[1]:label_top_left[1] + input_size[1], label_top_left[0]:label_top_left[0] + input_size[0]] = crop_labelBGR
                    cv2.rectangle(frame_image, label_top_left, (
                    label_top_left[0] + input_size[0], label_top_left[1] + input_size[1]),
                                  (20, 220, 20), 2)

                    prediction_top_left =  (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2), 500 + top_offset + (int)((250 - input_size[1]) / 2))
                    #make prediction rgb
                    frame_image[prediction_top_left[1]:prediction_top_left[1] + input_size[1],  prediction_top_left[0]:prediction_top_left[0] + input_size[0]] = crop_imageBGR
                    cv2.rectangle(frame_image, prediction_top_left, (
                        prediction_top_left[0] + input_size[0], prediction_top_left[1] + input_size[1]),
                                  (20, 220, 20), 2)

                    prediction_coordConv_top_left = (1400 + (int)((video_frame_width - 1400 - input_size[0]) / 2), 750 + top_offset + (int)((250 - input_size[1]) / 2))
                    # make prediction rgb
                    frame_image[prediction_coordConv_top_left[1]:prediction_coordConv_top_left[1] + input_size[1], prediction_coordConv_top_left[0]:prediction_coordConv_top_left[0] + input_size[0]] = crop_imageBGR
                    cv2.rectangle(frame_image, prediction_coordConv_top_left, (
                        prediction_coordConv_top_left[0] + input_size[0], prediction_coordConv_top_left[1] + input_size[1]),
                                  (20, 220, 20), 2)

                    #draw line connecting cropped regions
                    cv2.line(frame_image, (roi_bottom_right[0] + image_topLeft[0], roi_top_left[1] + image_topLeft[1]), original_image_top_left, (20, 220, 20), 1)
                    cv2.line(frame_image, (roi_bottom_right[0] + image_topLeft[0], roi_bottom_right[1] + image_topLeft[1]),
                             (label_top_left[0],label_top_left[1]+input_size[1]), (20, 220, 20), 1)

                    cv2.imshow('video_frame', frame_image)
                    cv2.waitKey(1)

                y += y_step
    tf.keras.backend.clear_session()

def main():
    predict_by_patches()

if __name__ == '__main__':
    main()