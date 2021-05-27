import cv2

from models.autoencoder import unet_autoencoder
from processing import tensor_to_image, image_to_tensor
from utilities import gather_image_from_dir

import tensorflow as tf
# Weights path
weight_path = None # provide path to weights '*.hdf5'
# Test images directory
test_images = 'data/image/'


def predict():
    # Define model
    model = unet_autoencoder(filters_in_input=16,
                             input_size=(320, 320, 1),
                             learning_rate=1e-3,
                             use_se=False,
                             use_aspp=True,
                             use_residual_connections=True,
                             leaky_relu_alpha=0.1,
                             pretrained_weights=weight_path)

    image_paths = gather_image_from_dir(test_images)

    # Load and predict on all images from directory
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # preprocess
        norm_image = image_to_tensor(image)
        # predict
        prediction = model.predict(norm_image)
        # make image uint8
        prediction_image = tensor_to_image(prediction)

        # Do you want to visualize image?
        show_image = True
        if show_image:
            cv2.imshow("image", image)
            cv2.imshow("prediction", prediction_image)
            cv2.waitKey(1000)


if __name__ == '__main__':
    predict()
