from mmdet.apis import init_detector, inference_detector
import cv2
import os
import numpy as np
from tqdm import trange
import six
import argparse
import random
import matplotlib.pyplot as plt


def all_NMS(all_objects):
    iou_th = 0.4
    tmp_objects = []
    re_idx = []
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['bbox']
        score1 = obj['score']

        for idx_c in range(idx + 1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            bbox2 = obj_c['bbox']
            score2 = obj_c['score']
            iou, inter, area1, area2 = box_iou(bbox1, bbox2)
            if iou > iou_th:
                id_m = idx if score1 < score2 else idx_c
                re_idx.append(id_m)
            elif inter == area1 or inter == area2:
                id_m = idx if score1 < score2 else idx_c
                re_idx.append(id_m)
        if idx not in re_idx:
            tmp_objects.append(obj)

    return tmp_objects


def box_iou(bbox1, bbox2):
    '''

    :param bbox1: GT(或det_obj)的矩形框（存放xmin,ymin,xmax,ymax的列表）
    :param bbox2: det_obj(或GT)的矩形框（存放xmin,ymin,xmax,ymax的列表）
    :return:iou, inter,area1,area2
    '''
    area1 = box_area(bbox1)
    area2 = box_area(bbox2)

    lt_x, lt_y = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    rb_x, rb_y = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    if rb_x - lt_x > 0 and rb_y - lt_y > 0:
        inter = (lt_x - rb_x) * (lt_y - rb_y)
    else:
        inter = 0
    iou = inter / (area1 + area2 - inter)
    return iou, inter, area1, area2


def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


config_file = '/mapai/haowenguo/code/SPL/mmdetection/plane_cfg/roi_test/expand_roi_size/faster_r50_P234_expand/config.py'
checkpoint_file = '/root/jizhi_logs/map_pic_auto2B9847BB731D4169B7CD/exp_0/epoch_120.pth'
img_path = '/mapai/haowenguo/code/SPL/mmdetection/tools/tsne_visualization/926.png'

num_class = 10
cls_map = {"Boeing737": 1,
           "Boeing747": 2,
           "Boeing777": 3,
           "Boeing787": 4,
           "A220": 5,
           "A321": 6,
           "A330": 7,
           "A350": 8,
           "ARJ21": 9,
           "other": 10}
sorces_th = 0

model = init_detector(config_file, checkpoint_file)
print(model)
cls_list = model.CLASSES

img = cv2.imread(img_path)
det_result = inference_detector(model, img)
# print(det_result)
all_objects = []
all_labels = []
for idx_cls in range(num_class):
    obbox_cls = det_result[idx_cls]
    # print(obbox_cls)
    for ids, each_obj in enumerate(obbox_cls):
        # print(each_obj)
        object_struct = {}
        # print(each_obj)
        # each_obj = [i for k in each_obj for i in k ]
        # print(each_obj)
        score = each_obj[4]  #####原为 each_obj[4]
        bbox = each_obj[:4]  #####原为 each_obj[:4]
        if score > sorces_th:
            object_struct['bbox'] = bbox
            object_struct['label'] = idx_cls
            object_struct['score'] = score
            all_objects.append(object_struct)
            all_labels.append(object_struct['label'])
        else:
            pass

final_objects = all_NMS(all_objects)
# print(final_objects)
# print(len(final_objects))

