import os

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def adjust_data(img, mask, labels):
    if len(labels) == 2:
        img_ = img / 255.
        mask_ = mask / 255.0
        mask_[mask > 0.5] = 1.0
        mask_[mask <= 0.5] = 0.0
    else:
        img_ = img / 255.
        mask = mask[:, :, :, 0]
        mask_ = np.zeros(mask.shape + (len(labels),))
        for i in range(len(labels)):
            mask_[mask == labels[i], i] = 1
    return img_, mask_


def data_generator(batch_size,
                   image_folder,
                   label_folder,
                   image_color_mode="grayscale",
                   mask_color_mode="grayscale",
                   multiclass_labels=[0, 255],
                   target_size=(320, 320),
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_base_dir, image_folder_name = os.path.split(os.path.dirname(image_folder))
    label_base_dir, label_folder_name = os.path.split(os.path.dirname(label_folder))
    # add slash to the end of dir
    image_base_dir = os.path.join(image_base_dir, '')
    label_base_dir = os.path.join(label_base_dir, '')
    image_generator = image_datagen.flow_from_directory(
        image_base_dir,
        classes=[image_folder_name],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True)
    mask_generator = mask_datagen.flow_from_directory(
        label_base_dir,
        classes=[label_folder_name],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, multiclass_labels)
        yield (img, mask)
