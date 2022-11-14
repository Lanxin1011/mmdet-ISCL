import numpy as np
import geopandas

import aitool

### add
from shapely.geometry import Polygon


def get_confusion_matrix_indexes(gt_bboxes, pred_bboxes):
    """calculate confusion matrix

    Args:
        gt_bboxes (list): list of ground truth bboxes ([xmin, ymin, xmax, ymax])
        pred_bboxes (list): list of prediction bboxes ([xmin, ymin, xmax, ymax])

    Returns:
        dict: information about TP, FP, FN
    """
    objects = dict()
    
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        print(f"Skip this combination, because length gt_bboxes or length pred_bboxes is zero")
        return []

    gt_polygons = [aitool.bbox2polygon(_) for _ in gt_bboxes]
    pred_polygons = [aitool.bbox2polygon(_) for _ in pred_bboxes]

    gt_polygons_origin = gt_polygons[:]
    pred_polygons_origin = pred_polygons[:]

    gt_polygons = geopandas.GeoSeries(gt_polygons)
    pred_polygons = geopandas.GeoSeries(pred_polygons)

    gt_df = geopandas.GeoDataFrame({'geometry': gt_polygons, 'gt_df':range(len(gt_polygons))})
    pred_df = geopandas.GeoDataFrame({'geometry': pred_polygons, 'pred_df':range(len(pred_polygons))})

    gt_df = gt_df.loc[~gt_df.geometry.is_empty]
    pred_df = pred_df.loc[~pred_df.geometry.is_empty]

    res_intersection = geopandas.overlay(gt_df, pred_df, how='intersection')

    iou = np.zeros((len(pred_polygons), len(gt_polygons)))
    for idx, row in res_intersection.iterrows():
        gt_idx = row.gt_df
        pred_idx = row.pred_df

        inter = row.geometry.area
        union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

        iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

    iou_indexes = np.argwhere(iou >= 0.5)

    gt_TP_indexes = list(iou_indexes[:, 1])
    pred_TP_indexes = list(iou_indexes[:, 0])

    gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
    pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

    objects['gt_iou'] = np.max(iou, axis=0)

    objects['gt_TP_indexes'] = gt_TP_indexes
    objects['pred_TP_indexes'] = pred_TP_indexes
    objects['gt_FN_indexes'] = gt_FN_indexes
    objects['pred_FP_indexes'] = pred_FP_indexes

    objects['gt_polygons'] = gt_polygons
    objects['pred_polygons'] = pred_polygons

    objects['gt_polygons_matched'] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
    objects['pred_polygons_matched'] = np.array(pred_polygons_origin)[pred_TP_indexes].tolist()

    return objects

def get_confusion_matrix_indexes_pointobb(gt_bboxes, pred_bboxes):
    """calculate confusion matrix

    Args:
        gt_bboxes (list): list of ground truth bboxes ([xmin, ymin, xmax, ymax])
        pred_bboxes (list): list of prediction bboxes ([xmin, ymin, xmax, ymax])

    Returns:
        dict: information about TP, FP, FN
    """
    objects = dict()
    
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        print(f"Skip this combination, because length gt_bboxes or length pred_bboxes is zero")
        return []

    # gt_polygons = [aitool.bbox2polygon(_) for _ in gt_bboxes]
    # pred_polygons = [aitool.bbox2polygon(_) for _ in pred_bboxes]

    gt_bboxes_tmp = [aitool.pointobb2bbox(_) for _ in gt_bboxes]
    gt_polygons = [aitool.bbox2polygon(_) for _ in gt_bboxes_tmp]
    pred_bboxes_tmp = [aitool.pointobb2bbox(_) for _ in pred_bboxes]
    pred_polygons = [aitool.bbox2polygon(_) for _ in pred_bboxes_tmp]

    # print(gt_polygons)
    # print(pred_polygons)

    gt_polygons_origin = gt_polygons[:]
    pred_polygons_origin = pred_polygons[:]

    gt_polygons = geopandas.GeoSeries(gt_polygons)
    pred_polygons = geopandas.GeoSeries(pred_polygons)

    gt_df = geopandas.GeoDataFrame({'geometry': gt_polygons, 'gt_df':range(len(gt_polygons))})
    pred_df = geopandas.GeoDataFrame({'geometry': pred_polygons, 'pred_df':range(len(pred_polygons))})

    gt_df = gt_df.loc[~gt_df.geometry.is_empty]
    pred_df = pred_df.loc[~pred_df.geometry.is_empty]

    res_intersection = geopandas.overlay(gt_df, pred_df, how='intersection')

    iou = np.zeros((len(pred_polygons), len(gt_polygons)))
    for idx, row in res_intersection.iterrows():
        gt_idx = row.gt_df
        pred_idx = row.pred_df

        inter = row.geometry.area
        union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

        iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

    iou_indexes = np.argwhere(iou >= 0.5)

    gt_TP_indexes = list(iou_indexes[:, 1])
    pred_TP_indexes = list(iou_indexes[:, 0])

    gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
    pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

    objects['gt_iou'] = np.max(iou, axis=0)

    objects['gt_TP_indexes'] = gt_TP_indexes
    objects['pred_TP_indexes'] = pred_TP_indexes
    objects['gt_FN_indexes'] = gt_FN_indexes
    objects['pred_FP_indexes'] = pred_FP_indexes

    objects['gt_polygons'] = gt_polygons
    objects['pred_polygons'] = pred_polygons

    objects['gt_pointobbs'] = gt_bboxes
    objects['pred_pointobbs'] = pred_bboxes

    objects['gt_polygons_matched'] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
    objects['pred_polygons_matched'] = np.array(pred_polygons_origin)[pred_TP_indexes].tolist()

    return objects