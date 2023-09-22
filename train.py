#!/usr/bin/env python3

import sys

from keras.optimizers import Adam

from models.models import create_models
from models.data import DataGenerator
from models.losses import DiceLoss, DiceLoss_Uncertain, RegularizationLoss
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

K.clear_session()

# for tensorflow 1*
tf.device("/gpu:0")
config = tf.ConfigProto()  # for tf 2 tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)  # for tf 2 tf.compat.v1.Session
init = tf.global_variables_initializer()
sess.run(init)


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    batch_size = 1  # 2
    classifier = create_models(n_channels=3, shape=256, num_classes=4)

    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = 0

    epoch_format = '.{epoch:03d}.h5'

    if initial_epoch != 0:
        suffix = epoch_format.format(epoch=initial_epoch)
        classifier.load_weights('classifier' + suffix)

    print("initial_epoch:", initial_epoch)
    epochs = 50
    opt = Adam(lr=0.001, decay=0.0001)  # 0.0005

    classifier.compile(loss=DiceLoss_Uncertain, optimizer=opt, metrics=[
                       'acc', DiceLoss, RegularizationLoss])

    classifier.summary()

    data_location = "./data/"

    with open("training_example.txt") as file:
        lines = file.readlines()
        training_samples = [line.rstrip() for line in lines]

    num_train_images = len(training_samples)
    training_generator = DataGenerator(
        data_location, training_samples, batch_size=batch_size, shuffle=True)

    checkpoint = ModelCheckpoint("./weights/" + datetime.now().strftime("%Y%m%d-%H%M%S") +
                                 "best_model.hdf", monitor="loss", verbose=1, save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint, tf.keras.callbacks.TensorBoard(
        log_dir="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))]
    classifier.fit_generator(training_generator,
                             steps_per_epoch=num_train_images // batch_size,
                             epochs=epochs,
                             callbacks=callbacks_list)


if __name__ == '__main__':
    main()
