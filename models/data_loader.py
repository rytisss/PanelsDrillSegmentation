import cv2
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
                   train_path,
                   image_folder,
                   mask_folder,
                   aug_dict,
                   image_color_mode="grayscale",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   multiclass_labels=[0, 255],
                   save_to_dir=None,
                   target_size=(320, 320),
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        shuffle=True)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        shuffle=True)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, multiclass_labels)
        yield (img, mask)
