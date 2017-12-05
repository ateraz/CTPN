#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
import cv2, os, sys
import os.path as osp
import numpy as np
import itertools
from other import create_osr_model
from operator import mul

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

DEMO_IMAGE_DIR="uploads/"
NET_DEF_FILE="models/deploy.prototxt"
DETECTOR_MODEL_FILE="../supervisely-tutorials/anpr/models/model_192x256_vgg.h5"
#OSR_MODEL_FILE="../supervisely-tutorials/anpr_ocr/models/model5000_ua_25e_rnn256_rotation5.h5"
#OSR_MODEL_FILE="../supervisely-tutorials/anpr_ocr/models/model12000_ua_200e_cnn16x16x_fc32_rnn512_rotation3.h5"
OSR_MODEL_FILE="../supervisely-tutorials/anpr_ocr/models/weights-improvement-185-1.1535.hdf5"
LETTERS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'I', 'K', 'M', 'O', 'P', 'T', 'X')

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(LETTERS):
                outstr += LETTERS[c]
        ret.append(outstr)
    return ret

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
K.set_session(sess)

model_detector = load_model(DETECTOR_MODEL_FILE)

model_ocr = create_osr_model(256)
model_ocr.load_weights(OSR_MODEL_FILE)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)

img_w, img_h = 256, 48
for im_name in demo_imnames:
    if 'jpg' not in im_name:
        continue

new_h, new_w = 192.0, 256.0
im_file=osp.join(DEMO_IMAGE_DIR, im_name)
img_orig=cv2.imread(im_file)
img=img_orig[:,:,0] / 255.
h, w = img.shape
sf_h, sf_w = h / new_h, w / new_w
img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
predictions = model_detector.predict(img.reshape(-1, int(new_h), int(new_w), 1))
ps = (predictions+1) * (new_w/2, new_h/2, new_w/2, new_h/2)
ps[0][0] *= sf_w 
ps[0][1] *= sf_h
ps[0][2] *= sf_w
ps[0][3] *= sf_h
x1 = int(ps[0][0])
y1 = int(ps[0][1])
x2 = int(ps[0][2])
y2 = int(ps[0][3])
print y1, y2-y1, x1, x2-x1
print img_orig.shape
img_cropped = img_orig[y1:y2, x1:x2]
print img_cropped.shape
cv2.rectangle(img_orig,(x1,y1),(x2,y2),(0,255,0),3)
cv2.imwrite('./web/processed/' + im_name.replace('.', '_with_boxes.'), img_orig)
# Actual recognition result should be printed here
r,g,b = cv2.split(img_cropped)
img = cv2.merge([b,g,r])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_w, img_h))
img = img / 255.0
img = img.T
img = img.reshape(1, img_w, img_h, 1)
net_inp = model_ocr.get_layer(name='the_input').input
net_out = model_ocr.get_layer(name='softmax').output
net_out_value = sess.run(net_out, feed_dict={net_inp:img})
mult_prob = round(reduce(mul, sorted(net_out_value[0].max(axis=0))[-8:], 1), 2)
avg_prob = round(np.mean(sorted(net_out_value[0].max(axis=0))[-8:]), 2)
print decode_batch(net_out_value)[0], mult_prob, avg_prob

