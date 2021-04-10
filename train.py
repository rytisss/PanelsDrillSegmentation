import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.autoencoder import unet_autoencoder
from models.losses import Loss
from models.data_loader import data_generator
import os

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'C:\Users\Rytis\Desktop\straipsnis/test/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'
# batch size. How many samples you want to feed in one iteration?
batch_size = 4
# number_of_epoch. How many epoch you want to train?
number_of_epoch = 20


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # also save if validation error is smallest
        if 'val_dice_score' in logs.keys():
            val_score = logs['val_dice_score']
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                print('New best weights found!')
                self.model.save(weights_output_dir + '_best.hdf5')
        else:
            print('Key val_dice_score does not exist!')


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 5
    init_lr = 0.001
    lr = init_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def train():
    number_of_images = 14219
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = number_of_images // batch_size
    # Define model
    model = unet_autoencoder(filters_in_input=16,
                             input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50,
                             learning_rate=1e-3,
                             use_se=True,
                             use_aspp=True,
                             use_coord_conv=True,
                             leaky_relu_alpha=0.1)

    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'C:\Users\Rytis\Desktop\straipsnis\training data\dataForTraining_v3\dataForTraining_v3/'

    # Define data generator that will take images from directory
    train_data_generator = data_generator(batch_size,
                            image_folder=data_dir + 'Image_rois/',
                            label_folder=data_dir + 'Label_rois/',
                            target_size=(320, 320),
                            image_color_mode='grayscale')

    # create weights output directory
    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving for the best-performing weights
    saver = CustomSaver()
    # Learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss', verbose=1, save_best_only=False,
                                                          save_weights_only=False)
    model.fit(train_data_generator,
                        steps_per_epoch=number_of_iteration,
                        epochs=number_of_epoch,
                        callbacks=[model_checkpoint, learning_rate_scheduler],
                        shuffle=True)


def main():
    train()


if __name__ == "__main__":
    main()
