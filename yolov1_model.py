
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

def CONV(x, number_of_out_layer, kernel_size, stride, padding,activation_function=None):
    input       = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    conv        = slim.conv2d(input, number_of_out_layer, kernel_size, stride, "SAME", activation_fn=activation_function)
    batch_norm  = slim.batch_norm(conv)
    return tf.nn.leaky_relu(features=batch_norm, alpha=0.1)

def MAXPOOL(x, kernel_size, stride):
    return slim.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding="VALID")

def DROPOUT(x, keep_prob=0.5):
    return slim.dropout(x, keep_prob=keep_prob)

def build_yolov1(input):

    x = input
    batch_size = input.shape[0]
    with tf.variable_scope("CNN"):

        # LAYER 1
        conv1       =   CONV    (x,         number_of_out_layer=64,     kernel_size=7,      stride=2,       padding=0)
        maxpool1    =   MAXPOOL (conv1,     kernel_size=2,      stride=2)

        # LAYER 2
        conv2       =   CONV    (maxpool1,  number_of_out_layer=192,    kernel_size=3,      stride=1,       padding=0)
        maxpool2    =   MAXPOOL (conv2,     kernel_size=2,      stride=2)

        # LAYER 3
        conv3       =   CONV    (maxpool2,  number_of_out_layer=128,    kernel_size=1,      stride=1,       padding=0)
        conv4       =   CONV    (conv3,     number_of_out_layer=256,    kernel_size=3,      stride=1,       padding=0)
        conv5       =   CONV    (conv4,     number_of_out_layer=256,    kernel_size=1,      stride=1,       padding=0)
        conv6       =   CONV    (conv5,     number_of_out_layer=512,    kernel_size=3,      stride=1,       padding=0)
        maxpool3    =   MAXPOOL (conv6,     kernel_size=2,      stride=2)

        # LAYER 4
        conv7       =   CONV    (maxpool3,  number_of_out_layer=256,    kernel_size=1,      stride=1,       padding=0)
        conv8       =   CONV    (conv7,     number_of_out_layer=512,    kernel_size=3,      stride=1,       padding=0)
        conv9       =   CONV    (conv8,     number_of_out_layer=256,    kernel_size=1,      stride=1,       padding=0)
        conv10      =   CONV    (conv9,     number_of_out_layer=512,    kernel_size=3,      stride=1,       padding=0)
        conv11      =   CONV    (conv10,    number_of_out_layer=256,    kernel_size=1,      stride=1,       padding=0)
        conv12      =   CONV    (conv11,    number_of_out_layer=512,    kernel_size=3,      stride=1,       padding=0)
        conv13      =   CONV    (conv12,    number_of_out_layer=256,    kernel_size=1,      stride=1,       padding=0)
        conv14      =   CONV    (conv13,    number_of_out_layer=512,    kernel_size=3,      stride=1,       padding=0)
        conv15      =   CONV    (conv14,    number_of_out_layer=512,    kernel_size=1,      stride=1,       padding=0)
        conv16      =   CONV    (conv15,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)
        maxpool4    =   MAXPOOL (conv16,    kernel_size=2, stride=2)

        # LAYER 5
        conv17      =   CONV    (maxpool4,  number_of_out_layer=512,    kernel_size=1,      stride=1,       padding=0)
        conv18      =   CONV    (conv17,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)
        conv19      =   CONV    (conv18,    number_of_out_layer=512,    kernel_size=1,      stride=1,       padding=0)
        conv20      =   CONV    (conv19,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)
        conv21      =   CONV    (conv20,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)
        conv22      =   CONV    (conv21,    number_of_out_layer=1024,   kernel_size=3,      stride=2,       padding=0)

        # LAYER 6
        conv23      =   CONV    (conv22,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)
        conv24      =   CONV    (conv23,    number_of_out_layer=1024,   kernel_size=3,      stride=1,       padding=0)

        dropout     =   DROPOUT(tf.reshape(conv24, [batch_size,-1]), 0.5)
        connected         =   slim.fully_connected(dropout, 1470, activation_fn=None)

        final       =   tf.reshape(connected, [batch_size,7,7,30])

        # TODO. MODELS SUMMARY
        print("INTPUT\t\t: {}".format(input))
        print("CONV\t\t: {}".format(conv1))
        print("MAX\t\t: {}".format(maxpool1))
        print("CONV\t\t: {}".format(conv2))
        print("MAX\t\t: {}".format(maxpool2))
        print("CONV\t\t: {}".format(conv3))
        print("CONV\t\t: {}".format(conv4))
        print("CONV\t\t: {}".format(conv5))
        print("CONV\t\t: {}".format(conv6))
        print("MAX\t\t: {}".format(maxpool3))
        print("CONV\t\t: {}".format(conv7))
        print("CONV\t\t: {}".format(conv8))
        print("CONV\t\t: {}".format(conv9))
        print("CONV\t\t: {}".format(conv10))
        print("CONV\t\t: {}".format(conv11))
        print("CONV\t\t: {}".format(conv12))
        print("CONV\t\t: {}".format(conv13))
        print("CONV\t\t: {}".format(conv14))
        print("CONV\t\t: {}".format(conv15))
        print("CONV\t\t: {}".format(conv16))
        print("MAX\t\t: {}".format(maxpool4))
        print("CONV\t\t: {}".format(conv17))
        print("CONV\t\t: {}".format(conv18))
        print("CONV\t\t: {}".format(conv19))
        print("CONV\t\t: {}".format(conv20))
        print("CONV\t\t: {}".format(conv21))
        print("CONV\t\t: {}".format(conv22))
        print("CONV\t\t: {}".format(conv23))
        print("CONV\t\t: {}".format(conv24))
        print("DROPOUT\t\t: {}".format(dropout))
        print("CONNECTED\t: {}".format(connected))

        # TODO. PARAMETERS SUMMARY

        # TODO. PARAMETERS
        total_parameters = 0
        count = 1
        print("================== PARAMETERS SUMMARY ========================")
        print("======================= LAYER {} =============================".format(str(count)))
        for variable in tf.trainable_variables():
            print("LAYER {}".format(str(count)))
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print("Filter Shape\t\t:\t{},\nlength of shape\t\t:\t{}".format(shape, len(shape)))
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print("variable parameters\t:\t{}".format(variable_parameters))
            total_parameters += variable_parameters
            count += 1
            print("======================= LAYER {} =============================".format(str(count)))
        print("TOTAL PARAMETERS\t:\t{}".format(total_parameters))

        return final


input = tf.placeholder(tf.float32, [5, 448, 448, 3])

result = build_yolov1(input)

print(result)

def loss(final, y_hat):
    cls, x, y, w, h = y_hat

    cell_index = [x//64, y//64]
    delta_cell = [float(x%64), float(y%64)]
    width = w
    height = h

    '''
    0  : object of probability
    1  : x point
    2  : y point
    3  : w
    4  : h
    5  : object of probability
    6  : x point
    7  : y point
    8  : w
    9  : h
    10 : cls[0]
    .  :
    .  :
    .  :
    20 : cls[19] 
    
    '''

    for i in range(0,7):
        for j in range(0,7):
            if i is cell_index[0] and j is cell_index[1]:
                object_prob_1   = (1 - final[i][j][0])**2
                point_loss_1    = 5.0 * ((delta_cell[0] - final[i][j][1])**2 + (delta_cell[1] - final[i][j][2])**2)
                size_loss_1     = 5.0 * ((tf.sqrt(width) - tf.sqrt(final[i][j][3]))**2 + (tf.sqrt(height) - tf.sqrt(final[i][j][3]))**2)

                object_prob_2 = (1 - final[i][j][4])**2
                point_loss_2 = 5.0 * ((delta_cell[0] - final[i][j][5]) ** 2 + (delta_cell[1] - final[i][j][6]) ** 2)
                size_loss_2 = 5.0 * ((tf.sqrt(width) - tf.sqrt(final[i][j][7])) ** 2 + (tf.sqrt(height) - tf.sqrt(final[i][j][8])) ** 2)

                




























