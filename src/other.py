import cv2, caffe
import numpy as np
from matplotlib import cm

import keras
import tensorflow as tf

import os
from os.path import join
import json
import random
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.preprocessing import image

def prepare_img(im, mean):
    """
        transform img into caffe's input img.
    """
    im_data=np.transpose(im-mean, (2, 0, 1))
    return im_data


def draw_boxes(im, bboxes, is_display=True, color=None, caption="Image", wait=True):
    """
        boxes: bounding boxes
    """
    #boxes = filter(lambda b: can_be_plate(b, im), bboxes)
    for box in bboxes:
        if color==None:
            if len(box)==5 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        stretched = stretch(box, im)
        cv2.rectangle(im, tuple(stretched[:2]), tuple(stretched[2:4]), c)
    if is_display:
        cv2.imshow(caption, im)
        if wait:
            cv2.waitKey(0)
    return im


def cut_boxes(im, bboxes, color=None):
    """
        boxes: bounding boxes
    """
    #boxes = filter(lambda b: can_be_plate(b, im), bboxes)
    # if len(boxes) > 1:
    #     import ipdb; ipdb.set_trace()
    # else:
    #     return
    crop_img = []
    for box in bboxes:
        if color==None:
            if len(box)==5 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        stretched = stretch(box, im)
        crop_img.append(im[stretched[1]:stretched[3], stretched[0]:stretched[2]])
    return crop_img


def can_be_plate(box, im):
    min_white_pixels = 0.1
    min_aspect_ratio, max_aspect_ratio = 3, 6
    min_area, max_area = 1000, 20000
    x1, y1, x2, y2 = map(int, box[:4])
    width, height = x2 - x1, y2 - y1
    aspect_ratio = float(width)/height
    area = width * height
    if not (min_aspect_ratio < aspect_ratio < max_aspect_ratio and min_area < area < max_area):
        return False

    _, gray = cv2.threshold(cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.count_nonzero(gray == 255)
    return white_pixels > min_white_pixels * area


def stretch(box, im, scale=1.1):
    im_height, im_width = im.shape[:2]
    x1, y1, x2, y2 = box[:4]
    xc, yc = (x1+x2) / 2,(y1+y2) / 2
    new_width, new_height = scale * (x2-x1), scale * (y2-y1)
    new_x1 = max(0, int(xc - new_width / 2))
    new_x2 = min(im_width, int(xc + new_width / 2))
    new_y1 = max(0, int(yc - new_height / 2))
    new_y2 = min(im_height, int(yc + new_height / 2))
    # cv2.imshow('', im[new_y1:new_y2, new_x1:new_x2])
    # cv2.waitKey(0)
    return new_x1, new_y1, new_x2, new_y2


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


def normalize(data):
    if data.shape[0]==0:
        return data
    max_=data.max()
    min_=data.min()
    return (data-min_)/(max_-min_) if max_-min_!=0 else data-min_


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def create_osr_model(img_w, load=False):
    img_h = 64
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(23, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[8], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model


class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in xrange(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class CaffeModel:
    def __init__(self, net_def_file, model_file):
        self.net_def_file=net_def_file
        self.net=caffe.Net(net_def_file, model_file, caffe.TEST)

    def blob(self, key):
        return self.net.blobs[key].data.copy()

    def forward(self, input_data):
        return self.forward2({"data": input_data[np.newaxis, :]})

    def forward2(self, input_data):
        for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            self.net.blobs[k].data[...]=v
        return self.net.forward()

    def net_def_file(self):
        return self.net_def_file
