#!/usr/bin/env python3

from models.models import create_models
from tqdm import tqdm
import gdal
import numpy as np
import re
import os
import tensorflow as tf


def predict_sheet(classifier, input_sheet_location, output_sheet_location, batch_size=1, resolution=1.25, padding=0, img_size=256, upsample=False, downsample=False, num_classes=4):
    if upsample:
        padding = int(padding / 2)
        img_size = int(img_size / 2)
    if downsample:
        temp_sheet_location = "E://DL//instances//test-instance//prediction//temp.tif"
        gdal.Translate(temp_sheet_location, input_sheet_location,
                       xRes=resolution*2, yRes=resolution*2)
        ds = gdal.Open(temp_sheet_location)
    else:
        ds = gdal.Open(input_sheet_location)

    transform_in = ds.GetGeoTransform()
    if upsample:
        transform_out = (transform_in[0], transform_in[1] / 2, transform_in[2],
                         transform_in[3], transform_in[4], transform_in[5] / 2)
    else:
        transform_out = (transform_in[0], transform_in[1], transform_in[2],
                         transform_in[3], transform_in[4], transform_in[5])

    shape = np.shape(ds.ReadAsArray())
    print('shape of ds', shape)

    sheet = ds.ReadAsArray()
    sheet = np.moveaxis(sheet, 0, -1)[:, :, 0:3]

    # make sure that the sheet matches the model extent parameters
    excess_x = sheet.shape[1] % img_size
    excess_y = sheet.shape[0] % img_size

    if not excess_x == 0:
        additional_padding_x = img_size - excess_x
    else:
        additional_padding_x = 0

    if not excess_y == 0:
        additional_padding_y = img_size - excess_y
    else:
        additional_padding_y = 0

    if upsample:
        sheet_template = np.zeros(
            (sheet.shape[0] * 2, sheet.shape[1] * 2, 4), np.float32)

    else:
        sheet_template = np.zeros(
            (sheet.shape[0] + additional_padding_y, sheet.shape[1] + additional_padding_x, 4), np.float32)

    sheet_extended = np.zeros((int(sheet.shape[0] + 2*padding + additional_padding_y), int(
        sheet.shape[1] + 2*padding + additional_padding_x), 3), np.float32)
    sheet_extended[padding:-padding-additional_padding_y,
                   padding:-padding-additional_padding_x, :] = sheet

    x_count = int((sheet_extended.shape[1] - padding * 2) / img_size)
    y_count = int((sheet_extended.shape[0] - padding * 2) / img_size)

    for y in range(y_count):
        print(str(y) + str("/") + str(y_count))
        for x in range(x_count):
            y_start = y * img_size
            y_end = y * img_size + img_size
            x_start = x * img_size
            x_end = x * img_size + img_size

            sub_img = sheet_extended[y_start:y_end + 2 *
                                     padding, x_start:x_end + 2 * padding] / 255

            if upsample:
                sub_img = cv2.resize(
                    sub_img, (sub_img.shape[1] * 2, sub_img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

            # print(np.amax(sub_img))
            sub_img_expanded = np.expand_dims(sub_img, axis=0)
            Y_pred = classifier.predict(sub_img_expanded, batch_size)[0]
            if upsample:
                y_start_upsample = y * img_size * 2
                y_end_upsample = y * img_size * 2 + img_size * 2
                x_start_upsample = x * img_size * 2
                x_end_upsample = x * img_size * 2 + img_size * 2
                sheet_template[y_start_upsample:y_end_upsample,
                               x_start_upsample:x_end_upsample, :] = Y_pred.copy()

            else:
                sheet_template[y_start:y_end, x_start:x_end, :] = Y_pred.copy()[
                    :, :, 0:num_classes]

    sheet_out = sheet_template[0:sheet.shape[0], 0:sheet.shape[1], :]
    print('the size of sheet out', np.shape(sheet_out))

    # write raster
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_sheet_location, sheet_out.shape[1], sheet_out.shape[0], 4,
                            gdal.GDT_Float32, ['COMPRESS=LZW'])
    outdata.SetGeoTransform(transform_out)

    outdata.SetProjection(ds.GetProjection())
    sheet_out[sheet_out <= 0.005] = 0

    for i in range(4):
        outdata.GetRasterBand(i+1).WriteArray(np.squeeze(sheet_out[:, :, i]))

    outdata.FlushCache()
    outdata = None
    ds = None


if __name__ == '__main__':

    tf.device("/gpu:0")
    config = tf.ConfigProto()  # for tf 2 tf.compat.v1.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)  # for tf 2 tf.compat.v1.Session
    init = tf.global_variables_initializer()
    sess.run(init)
    c = create_models(n_channels=3)

    c.load_weights("./weights/20230922-121620best_model.hdf")
    imagesize = 256

    target_dir = "./predictions/"  # the dir to write the predictions
    sheet_dir = "F:/trainingdataset/siegfried+oldnational/"  # the dir that stores sheets

    with open("testsheets_examples.txt", "r") as file:    # the txt files with sheet names
        test_sheets = file.readlines()

    for sheet in tqdm(test_sheets):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        input_sheet_location = os.path.join(sheet_dir, sheet)
        output_sheet_location = os.path.join(target_dir, sheet)

        print('input', input_sheet_location,
              'and output', output_sheet_location)
        if os.path.isfile(input_sheet_location) == False or os.path.isfile(output_sheet_location) == True:
            continue
        predict_sheet(c, input_sheet_location, output_sheet_location, batch_size=1,
                      resolution=1.25, padding=0, img_size=imagesize, upsample=False, downsample=False)
