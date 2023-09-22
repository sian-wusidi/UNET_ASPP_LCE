
import os.path
import glob
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import keras
import tensorflow.keras.backend as K

import pdb


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_location, list_IDs, batch_size, shuffle=True, img_size=256):
        'Initialization'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_location = data_location
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.img_size = img_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Generate data
        X = []
        Y = []
        for i, ID in enumerate(list_IDs_temp):
            file_names = ID.split(',')
            anno_ID = file_names[1]
            sheet_ID = file_names[0]
            anno_ID_img = np.load(os.path.join(
                self.data_location, anno_ID))['arr_0']
            sheet_ID_img = np.load(os.path.join(
                self.data_location, sheet_ID))['arr_0']
            X.append(sheet_ID_img)
            Y.append(anno_ID_img)
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X, Y
