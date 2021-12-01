import math

import numpy as np
import tensorflow as tf

"""from tensorflow.keras.layers import LeakyReLU, Activation, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, \
    BatchNormalization, Dense, AveragePooling2D, concatenate, UpSampling2D, Add, MaxPooling2D, Input"""
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, LeakyReLU, \
    AveragePooling2D, UpSampling2D, concatenate, Add, MaxPooling2D, Conv2DTranspose, Input
# from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.custom_layers import AddCoords2D
from models.losses import Loss
from models.losses import dice_loss, dice_score, dice_eval, binary_crossentropy, Active_Contour_Loss, surface_loss, FocalLoss, \
    weighted_bce_loss, adjusted_weighted_bce_loss, cross_and_dice_loss, \
    weighted_cross_and_dice_loss, cross_and_dice_loss_multiclass


def CompileModel(model, lossFunction, num_class=2, learning_rate=1e-3):
    if num_class == 2:
        if lossFunction == Loss.DICE:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=dice_loss, metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=binary_crossentropy, metrics=[dice_eval])
        elif lossFunction == Loss.ACTIVECONTOURS:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Active_Contour_Loss, metrics=[dice_eval])
        elif lossFunction == Loss.SURFACEnDice:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=surface_loss, metrics=[dice_eval])
        elif lossFunction == Loss.FOCALLOSS:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=FocalLoss, metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_bce_loss, metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.5, 0.5), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.25, 0.75), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.75, 0.25), metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.5, 0.5),
                          metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.25, 0.75),
                          metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.75, 0.25),
                          metrics=[dice_eval])
    else:
        if lossFunction == Loss.CROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.5, 0.5),
                          metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.25, 0.75),
                          metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.75, 0.25),
                          metrics=[dice_eval])
    return model


def squeeze_and_excitation(input_tensor):
    # calculate channels
    channel_count = int(np.shape(input_tensor)[-1])
    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(channel_count, use_bias=False, kernel_initializer='he_normal')(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(channel_count, use_bias=False, kernel_initializer='he_normal')(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, channel_count])
    scale = input_tensor * excitation
    return scale


def aspp(input_tensor,
         kernels=8,
         kernel_size=3,
         leaky_relu_alpha=0.1):
    # dilate = 1
    dilate1 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=1, kernel_initializer='he_normal',
                     use_bias=False)(input_tensor)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = LeakyReLU(alpha=leaky_relu_alpha)(dilate1) if leaky_relu_alpha > 0.0 else Activation('relu')(dilate1)

    # dilate = 2
    dilate2 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=2, kernel_initializer='he_normal',
                     use_bias=False)(input_tensor)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = LeakyReLU(alpha=leaky_relu_alpha)(dilate2) if leaky_relu_alpha > 0.0 else Activation('relu')(dilate2)

    # dilate = 3
    dilate3 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=4, kernel_initializer='he_normal',
                     use_bias=False)(input_tensor)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = LeakyReLU(alpha=leaky_relu_alpha)(dilate3) if leaky_relu_alpha > 0.0 else Activation('relu')(dilate3)

    height, width, n_ch = input_tensor.shape.as_list()[1:]
    pool = AveragePooling2D(pool_size=(height, width))(input_tensor)
    pool = UpSampling2D((height, width), interpolation='bilinear')(pool)

    output = concatenate([dilate1, dilate2, dilate3, pool])

    # perform parameters reduction with 1x1
    output = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                    kernel_initializer='he_normal', use_bias=False)(output)
    output = BatchNormalization()(output)
    output = LeakyReLU(alpha=leaky_relu_alpha)(output) if leaky_relu_alpha > 0.0 else Activation('relu')(output)

    return output


def make_downscale_layer(input_tensor,
                         number_of_filters,
                         filter_size,
                         max_pool=True,
                         use_residual_connections=False,
                         use_aspp=False,
                         use_se=False,
                         use_coord_conv=False,
                         leaky_relu_alpha=0.1):
    if use_residual_connections:
        shortcut = Conv2D(number_of_filters, kernel_size=(1, 1), strides=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(input_tensor)
    # 1 conv
    conv = AddCoords2D()(input_tensor) if use_coord_conv else input_tensor
    conv = Conv2D(number_of_filters, kernel_size=(filter_size, filter_size), strides=1, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv) if leaky_relu_alpha > 0.0 else Activation('relu')(conv)
    # 2 conv
    conv = AddCoords2D()(conv) if use_coord_conv else conv
    conv = Conv2D(number_of_filters, kernel_size=(filter_size, filter_size), strides=1, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    if use_se:
        conv = squeeze_and_excitation(conv)
    # add shortcut (optional)
    if use_residual_connections:
        conv = Add()([conv, shortcut])
    # activate
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv) if leaky_relu_alpha > 0.0 else Activation('relu')(conv)
    # atrous spatial pyramid pool
    if use_aspp:
        conv = aspp(conv, number_of_filters, filter_size, leaky_relu_alpha)
    opposite_connection = conv
    output = MaxPooling2D(pool_size=(2, 2))(conv) if max_pool else conv
    return opposite_connection, output


def make_upscale_layer(input_tensor,
                       skipped_connection,
                       number_of_filters,
                       filter_size,
                       use_residual_connections=False,
                       use_se=False,
                       use_coord_conv=False,
                       leaky_relu_alpha=0.1):
    conv = AddCoords2D()(input_tensor) if use_coord_conv else input_tensor
    conv = Conv2DTranspose(number_of_filters, (filter_size, filter_size), strides=(2, 2), padding="same",
                           kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv) if leaky_relu_alpha > 0.0 else Activation('relu')(conv)
    conv = concatenate([conv, skipped_connection])
    # residual optional
    if use_residual_connections:
        shortcut = Conv2D(number_of_filters, kernel_size=(1, 1), strides=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(conv)
    # 1 conv
    conv = AddCoords2D()(conv) if use_coord_conv else conv
    conv = Conv2D(number_of_filters, kernel_size=(filter_size, filter_size), strides=1, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv) if leaky_relu_alpha > 0.0 else Activation('relu')(conv)
    # 2 conv
    conv = AddCoords2D()(conv) if use_coord_conv else conv
    conv = Conv2D(number_of_filters, kernel_size=(filter_size, filter_size), strides=1, padding='same',
                  kernel_initializer='he_normal', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    if use_se:
        conv = squeeze_and_excitation(conv)
    # add shortcut (optional)
    if use_residual_connections:
        conv = Add()([conv, shortcut])
    # activate
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv) if leaky_relu_alpha > 0.0 else Activation('relu')(conv)
    return conv


def unet_autoencoder(pretrained_weights=None,
                     num_class=2,
                     input_size=(320, 320, 1),
                     filters_in_input=16,
                     downscale_times=4,
                     use_residual_connections=False,
                     use_aspp=False,
                     use_se=False,
                     use_coord_conv=False,
                     leaky_relu_alpha=0.1,
                     filter_size_in_first_layer=5,
                     loss_function=Loss.CROSSENTROPY50DICE50,
                     learning_rate=0.001):
    # sanity check
    if num_class < 2:
        raise ValueError('Error! Should be at least 2 classes (background and object)!')

    down_layers = []
    # sanity check, must be at least 2 downscales
    if downscale_times < 2:
        raise ValueError('Should be at least 2 downscales!')
    inputs = Input(input_size)
    # first layer always simple, without residual connections
    opposite_connection_0, enc_0 = make_downscale_layer(inputs, filters_in_input, filter_size_in_first_layer,
                                                        use_residual_connections=False, use_aspp=False, use_se=use_se,
                                                        use_coord_conv=False, leaky_relu_alpha=leaky_relu_alpha)
    down_layers.append((opposite_connection_0, enc_0))
    # calculate how many layers until bottleneck (without input and bottleneck)
    layers_until_bottleneck = downscale_times - 2
    for i in range(layers_until_bottleneck):
        opposite_connection, enc = make_downscale_layer(down_layers[i][1], int(filters_in_input * math.pow(2, (i + 1))),
                                                        3,
                                                        use_residual_connections=use_residual_connections,
                                                        use_aspp=False, use_se=use_se,
                                                        use_coord_conv=use_coord_conv,
                                                        leaky_relu_alpha=leaky_relu_alpha)
        down_layers.append((opposite_connection, enc))
    # bottleneck
    _, enc_bridge = make_downscale_layer(down_layers[layers_until_bottleneck][1],
                                         int(filters_in_input * math.pow(2, downscale_times - 1)), 3,
                                         max_pool=False,
                                         use_residual_connections=use_residual_connections,
                                         use_aspp=use_aspp, use_se=use_se,
                                         use_coord_conv=use_coord_conv,
                                         leaky_relu_alpha=leaky_relu_alpha)
    # down_layers.append((opposite_connection_bridge, enc_bridge))
    # reverse for iterations
    down_layers.reverse()
    # how many layers until output (without bottleneck and last layer)
    layers_until_output = layers_until_bottleneck
    latest_dec_input_tensor = enc_bridge
    for i in range(layers_until_output):
        latest_dec_input_tensor = make_upscale_layer(latest_dec_input_tensor, down_layers[i][0],
                                                     int(filters_in_input * math.pow(2, downscale_times - i - 2)),
                                                     filter_size=3,
                                                     use_residual_connections=use_residual_connections,
                                                     use_se=use_se, use_coord_conv=use_coord_conv,
                                                     leaky_relu_alpha=leaky_relu_alpha)
    # make last layer
    last_up_layer = make_upscale_layer(latest_dec_input_tensor, down_layers[downscale_times - 2][0], filters_in_input,
                                       filter_size=3,
                                       use_residual_connections=False, use_se=use_se, use_coord_conv=use_coord_conv,
                                       leaky_relu_alpha=leaky_relu_alpha)

    output = Conv2D(num_class, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
                    use_bias=False)(
        last_up_layer)
    output = BatchNormalization()(output)
    output = LeakyReLU(alpha=leaky_relu_alpha)(output) if leaky_relu_alpha > 0.0 else Activation('relu')(output)
    if num_class == 2:
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal',
                         use_bias=False)(output)
    else:
        outputs = Conv2D(num_class, (1, 1), padding="same", activation="softmax", kernel_initializer='glorot_normal',
                         use_bias=False)(
            output)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, num_class, learning_rate)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        #plot_model(model, to_file='UNet4.png', show_shapes=True, show_layer_names=True)
    return model
