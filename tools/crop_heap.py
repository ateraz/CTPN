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
from other import draw_boxes, cut_boxes, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

DEMO_IMAGE_DIR="/mnt/license_plates/autos/"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)
timer=Timer()

#import ipdb; ipdb.set_trace()
img_w, img_h = 128, 64
for im_name in set(os.listdir("/mnt/license_plates/autos/")):
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "\n%s"%im_name,
    if 'jpg' not in im_name:
        continue

    im_file=osp.join("/mnt/license_plates/autos/", im_name)
    im=cv2.imread(im_file)
    height = im.shape[0]
    #im=im[int(height*0.10):int(height*0.9), :]

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    #print "Number of the detected text lines: %s"%len(text_lines)
    #print "Time: %f"%timer.toc()

    im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=im_name, wait=True)
#print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#print "Thank you for trying our demo. Press any key to exit..."
#cv2.waitKey(0)

