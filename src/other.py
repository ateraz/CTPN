import cv2, caffe
import numpy as np
from matplotlib import cm


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
    boxes = filter(lambda b: can_be_plate(b, im), bboxes)
    for box in boxes:
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


def can_be_plate(box, im):
    min_white_pixels = 0.4
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
