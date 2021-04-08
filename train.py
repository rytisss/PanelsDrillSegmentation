import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.autoencoder import *
from models.losses import Loss
from models.data_loader import data_generator
import os

###############################
# Super-basic training routine
###############################

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'D:\drilled holes data for training\UNet4_res_assp_5x5_16k_320x320_coordConv_v2/'
weights_output_name = 'UNet4_res_assp_5x5_16k_320x320'


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.overallIteration = 0

    def on_batch_end(self, iteration, logs={}):
        self.overallIteration += 1
        if self.overallIteration % 5000 == 0 and self.overallIteration != 0:  # or save after some epoch, each k-th epoch etc.
            print('Saving iteration ' + str(self.overallIteration))
            self.model.save(weights_output_dir + weights_output_name + "_{}.hdf5".format(self.overallIteration))


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 6
    init_lr = 0.001
    lr = init_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def train():
    number_of_images = 14219
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 3
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = number_of_images / batch_size

    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 20

    # Define model
    model = UNet4_res_aspp_First5x5(number_of_kernels=32,
                                    input_size=(512, 512, 3),
                                    loss_function=Loss.CROSSENTROPY50DICE50,
                                    learning_rate=1e-3,
                                    useLeakyReLU=True,
                                    LeakyReLU_alpha=0.1)

    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = './data/output/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size,
                               data_dir,
                               'Image_rois',
                               'Label_rois',
                               data_gen_args,
                               save_to_dir=None,
                               target_size=(512, 512),
                               image_color_mode='rgb')

    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving
    #saver = CustomSaver()
    # Learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False)
    model.fit_generator(generator, steps_per_epoch=number_of_iteration, epochs=number_of_epoch, callbacks=[model_checkpoint, learning_rate_scheduler], shuffle=True)


def main():
    train()


if __name__ == "__main__":
    main()