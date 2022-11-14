import cv2
import numpy as np

import aitool


def draw_bbox(img, bbox, color=(0, 0, 255), line_width=2):
    """show rectangle (bbox)

    Args:
        img (np.array): input image
        bbox (list): [xmin, ymin, xmax, ymax]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).
        line_width (int, optional): line width

    Returns:
        np.array: output image
    """
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, line_width)

    return img

def draw_bbox_w_text(img, bbox, pred_cat_name, color=(0, 0, 255), line_width=2, font_scale=2, thickness=1):
    """show rectangle (bbox)

    Args:
        img (np.array): input image
        bbox (list): [xmin, ymin, xmax, ymax]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).
        line_width (int, optional): line width

    Returns:
        np.array: output image
    """
    print(bbox)
    rect = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, line_width)
    cv2.putText(rect, str(pred_cat_name), (int(bbox[0]), int(bbox[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=color, thickness=thickness)

    return img

def draw_pointobb_w_text(img, pointobb, pred_cat_name, color=(0, 0, 255), line_width=2, font_scale=2, thickness=1):
    """show rectangle (bbox)

    Args:
        img (np.array): input image
        bbox (list): [xmin, ymin, xmax, ymax]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).
        line_width (int, optional): line width

    Returns:
        np.array: output image
    """
    # print(pointobb)
    # pointobb_tmp = pointobb
    # pointobb_tmp.append(pointobb[0])
    # pointobb_tmp.append(pointobb[1])
    pointobb_tmp = np.array(pointobb).reshape(-1, 1, 2)
    # print(pointobb_tmp)
    poly = cv2.polylines(img, [pointobb_tmp], True, color, thickness=line_width)
    cv2.putText(poly, str(pred_cat_name), (int(pointobb[0]), int(pointobb[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=color, thickness=thickness)

    return img

def draw_confusion_matrix_pointobb(img, gt_bboxes, pred_bboxes, gt_cat_list=None, pred_cat_list=None, with_gt_TP=False, line_width=2, font_scale=3, thickness=1):
    colors = {'gt_TP':   (255, 255, 0),     # yellow --> 检测到，但是类别分错了
              'pred_TP': (0, 255, 0),       # green --> 检测到，且分类正确
              'FP':      (0, 255, 255),     # blue --> 检测到，但是实际上gt里没有
              'FN':      (255, 0, 0)}       # red --> 没有检测到

    objects = aitool.get_confusion_matrix_indexes_pointobb(gt_bboxes, pred_bboxes)
    print(objects)
    # assert False

    if len(objects) == 0:
        return []
    
    ###
    non_TP_idx_list = []
    for idx, pred_polygon in enumerate(objects['pred_pointobbs']):
        if idx in objects['pred_TP_indexes']:
            gt_class_name = gt_cat_list[objects['gt_TP_indexes'][objects['pred_TP_indexes'].index(idx)]]
            pred_class_name = pred_cat_list[idx]
            if gt_class_name == pred_class_name:
                color = colors['pred_TP'][::-1]
                if pred_cat_list == None:
                    img = aitool.draw_bbox(img, pred_polygon, color=color, line_width=line_width)
                else:
                    img = aitool.draw_pointobb_w_text(img, pred_polygon, pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
            else:
                non_TP_idx_list.append(objects['gt_TP_indexes'][objects['pred_TP_indexes'].index(idx)]) # 把gt_TP_indexes中存放的序号加进来
        else:
            color = colors['FP'][::-1]
            if pred_cat_list == None:
                img = aitool.draw_bbox(img, pred_polygon, color=color, line_width=line_width)
            else:
                img = aitool.draw_pointobb_w_text(img, pred_polygon, pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    print(non_TP_idx_list)

    for idx, gt_polygon in enumerate(objects['gt_pointobbs']):
        if idx not in objects['gt_TP_indexes']:
            color = colors['FN'][::-1]
            if gt_cat_list == None:
                img = aitool.draw_bbox(img, gt_polygon, color=color, line_width=line_width)
            else:
                img = aitool.draw_pointobb_w_text(img, gt_polygon, gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
        if idx in non_TP_idx_list:
            color = colors['gt_TP'][::-1]
            if gt_cat_list == None:
                img = aitool.draw_bbox(img, gt_polygon, color=color, line_width=line_width)
            else:
                img = aitool.draw_pointobb_w_text(img, gt_polygon, gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    ###
    
    # for idx, gt_polygon in enumerate(objects['gt_polygons']):
    #     if idx in objects['gt_TP_indexes']:
    #         if with_gt_TP:
    #             color = colors['gt_TP'][::-1]
    #             img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #     else:
    #         color = colors['FN'][::-1]
    #         # img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #         if gt_cat_list == None:
    #             img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #         else:
    #             img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(gt_polygon), gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    
    ###
    # for idx, pred_polygon in enumerate(objects['pred_polygons']):
    #     if idx in objects['pred_TP_indexes']:
    #         color = colors['pred_TP'][::-1]
    #     else:
    #         color = colors['FP'][::-1]
        
    #     # img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)

    #     ###
    #     if pred_cat_list == None:
    #         img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)
    #     else:
    #         img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(pred_polygon), pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    #     ###
    return img  


def draw_confusion_matrix(img, gt_bboxes, pred_bboxes, gt_cat_list=None, pred_cat_list=None, with_gt_TP=False, line_width=2, font_scale=3, thickness=1):
    colors = {'gt_TP':   (255, 255, 0),     # yellow --> 检测到，但是类别分错了
              'pred_TP': (0, 255, 0),       # green --> 检测到，且分类正确
              'FP':      (0, 255, 255),     # blue --> 检测到，但是实际上gt里没有
              'FN':      (255, 0, 0)}       # red --> 没有检测到

    objects = aitool.get_confusion_matrix_indexes(gt_bboxes, pred_bboxes)
    # print(objects)
    # assert False
    

    if len(objects) == 0:
        return []
    
    ###
    non_TP_idx_list = []
    for idx, pred_polygon in enumerate(objects['pred_polygons']):
        if idx in objects['pred_TP_indexes']:
            gt_class_name = gt_cat_list[objects['gt_TP_indexes'][objects['pred_TP_indexes'].index(idx)]]
            pred_class_name = pred_cat_list[idx]
            if gt_class_name == pred_class_name:
                color = colors['pred_TP'][::-1]
                if pred_cat_list == None:
                    img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)
                else:
                    img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(pred_polygon), pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
            else:
                non_TP_idx_list.append(objects['gt_TP_indexes'][objects['pred_TP_indexes'].index(idx)]) # 把gt_TP_indexes中存放的序号加进来
        else:
            color = colors['FP'][::-1]
            if pred_cat_list == None:
                img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)
            else:
                img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(pred_polygon), pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    print(non_TP_idx_list)

    for idx, gt_polygon in enumerate(objects['gt_polygons']):
        if idx not in objects['gt_TP_indexes']:
            color = colors['FN'][::-1]
            if gt_cat_list == None:
                img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
            else:
                img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(gt_polygon), gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
        if idx in non_TP_idx_list:
            color = colors['gt_TP'][::-1]
            if gt_cat_list == None:
                img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
            else:
                img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(gt_polygon), gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    ###
    
    # for idx, gt_polygon in enumerate(objects['gt_polygons']):
    #     if idx in objects['gt_TP_indexes']:
    #         if with_gt_TP:
    #             color = colors['gt_TP'][::-1]
    #             img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #     else:
    #         color = colors['FN'][::-1]
    #         # img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #         if gt_cat_list == None:
    #             img = aitool.draw_bbox(img, aitool.polygon2bbox(gt_polygon), color=color, line_width=line_width)
    #         else:
    #             img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(gt_polygon), gt_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    
    ###
    # for idx, pred_polygon in enumerate(objects['pred_polygons']):
    #     if idx in objects['pred_TP_indexes']:
    #         color = colors['pred_TP'][::-1]
    #     else:
    #         color = colors['FP'][::-1]
        
    #     # img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)

    #     ###
    #     if pred_cat_list == None:
    #         img = aitool.draw_bbox(img, aitool.polygon2bbox(pred_polygon), color=color, line_width=line_width)
    #     else:
    #         img = aitool.draw_bbox_w_text(img, aitool.polygon2bbox(pred_polygon), pred_cat_list[idx], color=color, line_width=line_width, thickness=thickness)
    #     ###
    return img  