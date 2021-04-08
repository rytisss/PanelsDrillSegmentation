import cv2
import numpy as np


def resize_and_insert(image, destination_width, destination_height):
    """
    Resize to desired size and maintain aspect ratio (does offset [bottom/top or left/right] if needed)
    :param image: input image
    :param destination_width: desired width
    :param destination_height: desired height
    :return: resized image, ROI of the image [aspect ratio maintained], resize ratio
    """
    destination_image = np.zeros((destination_height, destination_width), np.uint8)
    # resize 'imageROI' to fit into destination image, also keep the aspect ratio
    roi_height, roi_width = image.shape[:2]
    x_aspect_ratio = destination_width / roi_width
    y_aspect_ratio = destination_height / roi_height
    resize_ratio = x_aspect_ratio if x_aspect_ratio < y_aspect_ratio else y_aspect_ratio
    resized_width = int(resize_ratio * float(roi_width))
    resized_height = int(resize_ratio * float(roi_height))
    # prevention from too big width
    if resized_width > destination_width:
        resized_width = destination_width
    if resized_height > destination_height:
        resized_height = destination_height

    resized_roi = cv2.resize(image, (resized_width, resized_height))
    # calculate offset for roi in image (roi is centered)
    resized_roi_height, resized_roi_width = resized_roi.shape[:2]
    x_offset = int((destination_width - resized_roi_width) / 2)
    y_offset = int((destination_height - resized_roi_height) / 2)
    destination_image[y_offset:y_offset + resized_roi_height, x_offset:x_offset + resized_roi_width] = resized_roi
    return destination_image, (
        x_offset, y_offset, x_offset + resized_roi_width, y_offset + resized_roi_height), resize_ratio


def split_image_to_tiles(width, height, tile_width, tile_height, tile_overlay_X, tile_overlay_Y):
    tile_regions = []
    if tile_width > width or tile_height > height:
        return tile_regions  # nothing to do in this case if tile is bigger than image

    in_width_range = True
    in_height_range = True
    current_x = 0
    current_y = 0
    step_x = tile_width - tile_overlay_X
    step_y = tile_height - tile_overlay_Y
    last_iteration = False
    while in_height_range:
        in_width_range = True
        current_x = 0
        while in_width_range:
            # form region
            if current_x >= width - tile_width - 1:
                current_x = width - tile_width - 1  # adapt region to fit into image
                in_width_range = False
            region = (current_x, current_y, current_x + tile_width, current_y + tile_height)
            tile_regions.append(region)
            current_x += step_x
        current_y += step_y
        if last_iteration:
            in_height_range = False
        if current_y >= height - tile_height - 1:
            current_y = height - tile_height
            last_iteration = True
        if tile_height == height:
            break
    return tile_regions


def crop_image_from_region(image, roi):
    image_width = image.shape[1]
    image_height = image.shape[0]
    # make empty tile and place region that fits into image into this tile
    x = roi[0]
    y = roi[1]
    width = roi[2] - roi[0]
    height = roi[3] - roi[1]
    # fitted roi
    x_ = 0
    y_ = 0
    w_ = 0
    h_ = 0

    # check if image is RGB
    if len(image.shape) == 3:
        crop_image = np.zeros((width, height, 3), dtype=np.uint8)
    else:
        crop_image = np.zeros((width, height), dtype=np.uint8)

    if x > image_width:
        return crop_image
    if y > image_height:
        return crop_image
    if x + width < 0:
        return crop_image
    if y + height < 0:
        return crop_image

    # roi starts outside but overlays image (partly or fully)
    if x < 0 and (x + width > 0):
        x_ = 0
        width_ = width + x
        # check if 'fittedBoundingBox.width' is not out of image range
        if width_ > image_width:
            width_ = image_width
    # roi starts outside but overlays image (partly or fully)
    if y < 0 and (y + height > 0):
        y_ = 0
        height_ = height + y
        # check if 'fittedBoundingBox.height' is not out of image range
        if height_ > image_height:
            height_ = image_height
    # roi start inside image
    if x >= 0 and (x < image_width):
        x_ = x
        if x + width <= image_width:
            width_ = width
        else:
            width_ = image_width - x_
    # roi start inside image
    if y >= 0 and (y < image_height):
        y_ = y
        if y + height <= image_height:
            height_ = height
        else:
            height_ = image_height - y_
    crop_img_ = image[y_:y_ + height_, x_:x_ + width_]

    # check how much of the region is out of image and put cropped image it to requested size image
    x_req = 0
    y_req = 0
    if x < 0:
        x_req = -x
    if y < 0:
        y_req = -y
    crop_image[y_req:y_req + height_, x_req:x_req + width_] = crop_img_
    return crop_image


def map_to_original_size(prediction_image, roi, original_width, original_height):
    # get roi from prediction
    prediction_roi = prediction_image[roi[1]:roi[3], roi[0]:roi[2]]
    # resize to original size
    original_size_prediction = cv2.resize(prediction_roi, (original_width, original_height))
    return original_size_prediction


def fit_image_to_screen(image, screen_width=1920, screen_height=1080, scale=0.75):
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


def image_to_tensor(image):
    # preprocess
    image_norm = image / 255
    image_norm = np.reshape(image_norm, image_norm.shape + (1,))
    image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
    return image_norm


def tensor_to_image(tensor):
    # normalize to image
    prediction_image_norm = tensor[0, :, :, 0]
    prediction_image = prediction_image_norm * 255
    prediction_image = prediction_image.astype(np.uint8)
    return prediction_image
