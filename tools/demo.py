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
from other import draw_boxes, cut_boxes, resize_im, CaffeModel, create_osr_model
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
import numpy as np
import itertools
from utils.timer import Timer
from matplotlib import cm
from operator import mul

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

DEMO_IMAGE_DIR="uploads/"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"
OSR_MODEL_FILE="../supervisely-tutorials/anpr_ocr/models/model5000_ua_25e_rnn256_rotation5.h5"
OSR_MODEL_FILE="..//supervisely-tutorials/anpr_ocr/models/model12000_ua_200e_cnn16x16x_fc32_rnn512_rotation3.h5"
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

model = create_osr_model(128)
model.load_weights(OSR_MODEL_FILE)

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)

img_w, img_h = 128, 64
for im_name in demo_imnames:
    if 'jpg' not in im_name:
        continue

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)
    height = im.shape[0]
    #im=im[int(height*0.10):int(height*0.9), :]

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    #print "Number of the detected text lines: %s"%len(text_lines)

    color = tuple(cm.jet([0.9])[0, 2::-1]*255)
    im_with_text_lines=draw_boxes(im, text_lines, is_display=False, caption=im_name, wait=True, color=color)
    cv2.imwrite('./web/processed/' + im_name.replace('.', '_with_boxes.'), im_with_text_lines)
    # Actual recognition result should be printed here
    img_list = cut_boxes(im, text_lines)
    for img in img_list: 
        r,g,b = cv2.split(img)
        img = cv2.merge([b,g,r])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w, img_h))
        img = img / 255.0
        img = img.T
        img = img.reshape(1, img_w, img_h, 1)
        net_inp = model.get_layer(name='the_input').input
        net_out = model.get_layer(name='softmax').output
        net_out_value = sess.run(net_out, feed_dict={net_inp:img})
        print decode_batch(net_out_value)[0], reduce(mul, sorted(net_out_value[0].max(axis=0))[-8:], 1)
