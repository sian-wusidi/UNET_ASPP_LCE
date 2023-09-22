#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose,\
    Lambda, Activation, MaxPooling2D, Dropout, AveragePooling2D, DepthwiseConv2D, Concatenate, Cropping2D
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
import tensorflow as tf


def create_models(n_channels=3, shape=256, num_classes=4):

    image_shape = (shape, shape, n_channels)
    num_filters = 16
    OS = 16
    atrous_rates = (3, 6, 9)

    def create_classifier_original():

        inputs = Input(shape=image_shape, name="class_input")

        # 256
        c1 = Conv2D(num_filters, (3, 3),
                    kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Activation('elu')(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(num_filters, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation('elu')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        # 128
        c2 = Conv2D(num_filters * 2, (3, 3),
                    kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('elu')(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(num_filters * 2, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        c2 = Activation('elu')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        # 64
        c3 = Conv2D(num_filters * 4, (3, 3),
                    kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Activation('elu')(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(num_filters * 4, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        c3 = Activation('elu')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        # 32
        c4 = Conv2D(num_filters * 8, (3, 3),
                    kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Activation('elu')(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(num_filters * 8, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        c4 = Activation('elu')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        # 16
        c5 = Conv2D(num_filters * 16, (3, 3),
                    kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Activation('elu')(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(num_filters * 16, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)
        c5 = Activation('elu')(c5)

        # ASPP
        # Image Feature branch
        #out_shape = int(np.ceil(input_shape[0] / OS))
        b4 = AveragePooling2D(pool_size=(
            int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(c5)
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
        b4 = BatchNormalization(epsilon=1e-5)(b4)
        b4 = Activation('elu')(b4)
        b4 = BilinearUpsampling(
            (int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5)
        b0 = BatchNormalization(epsilon=1e-5)(b0)
        b0 = Activation('elu')(b0)
        #rate = 3 (6)
        b1 = SepConv_BN(
            c5, 256, rate=atrous_rates[0], depth_activation=False, epsilon=1e-5)
        # rate = 6 (12)
        b2 = SepConv_BN(
            c5, 256, rate=atrous_rates[1], depth_activation=False, epsilon=1e-5)
        # rate = 9 (18)
        b3 = SepConv_BN(
            c5, 256, rate=atrous_rates[2], depth_activation=False, epsilon=1e-5)

        # concatenate ASPP branches & project
        c5 = Concatenate()([b4, b0, b1, b2, b3])

        # simple 1x1 again
        c5 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5)

        c5 = BatchNormalization(epsilon=1e-5)(c5)
        c5 = Activation('elu')(c5)
        c5 = Dropout(0.1)(c5)

        # 32
        u6 = Conv2DTranspose(num_filters * 8, (2, 2),
                             strides=(2, 2), padding='same')(c5)
        u6 = Concatenate()([u6, c4])
        c6 = Conv2D(num_filters * 8, (3, 3),
                    kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Activation('elu')(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(num_filters * 8, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)
        c6 = Activation('elu')(c6)

        # 64
        u7 = Conv2DTranspose(num_filters * 4, (2, 2),
                             strides=(2, 2), padding='same')(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(num_filters * 4, (3, 3),
                    kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Activation('elu')(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(num_filters * 4, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)
        c7 = Activation('elu')(c7)

        # 128
        u8 = Conv2DTranspose(num_filters * 2, (2, 2),
                             strides=(2, 2), padding='same')(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(num_filters * 2, (3, 3),
                    kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Activation('elu')(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(num_filters * 2, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)
        c8 = Activation('elu')(c8)

        # 256
        u9 = Conv2DTranspose(num_filters, (2, 2),
                             strides=(2, 2), padding='same')(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(num_filters, (3, 3),
                    kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Activation('elu')(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(num_filters, (3, 3),
                    kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)
        c9 = Activation('elu')(c9)

        # per-class predictions + per-class uncertainty
        finalConv = Conv2D(2*num_classes, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=inputs, outputs=finalConv, name="classifier")

        return model

    classifier = create_classifier_original()
    return classifier


def SepConv_BN(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
            the code is based on keras implementation of deeplabV3+ https://github.com/bonlime/keras-deeplab-v3-plus
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('elu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(
        rate, rate), padding=depth_padding, use_bias=False)(x)  # depthwise
    x = BatchNormalization(epsilon=epsilon)(x)

    if depth_activation:
        x = Activation('elu')(x)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)  # pointwise
    x = BatchNormalization(epsilon=epsilon)(x)

    if depth_activation:
        x = Activation('elu')(x)

    return x


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]), align_corners=True)
            # return K.tensorflow_backend.tf.image.resize(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]))
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0], self.output_size[1]), align_corners=True)
            # return K.tensorflow_backend.tfimage.resize(inputs, (self.output_size[0], self.output_size[1]))

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
