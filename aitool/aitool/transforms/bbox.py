import numpy as np
from shapely.geometry import Polygon

import aitool


def xyxy2cxcywh(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [cx, cy, w, h]
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    w = xmax - xmin
    h = ymax - ymin
    
    return [cx, cy, w, h]

def cxcywh2xyxy(bbox):
    """bbox format convert

    Args:
        bbox (list): [cx, cy, w, h]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    cx, cy, w, h = bbox
    xmin = int(cx - w / 2.0)
    ymin = int(cy - h / 2.0)
    xmax = int(cx + w / 2.0)
    ymax = int(cy + h / 2.0)
    
    return [xmin, ymin, xmax, ymax]

def xywh2xyxy(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, w, h]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, w, h = bbox
    xmax = xmin + w
    ymax = ymin + h
    
    return [xmin, ymin, xmax, ymax]

def xyxy2xywh(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [xmin, ymin, w, h]
    """
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    
    return [xmin, ymin, w, h]

def bbox2polygon(bbox):
    """convert single bbox to polygon

    Arguments:
        bbox {list} -- contains coordinates of bounding box ([xmin, ymin, xmax, ymax])

    Returns:
        Polygon: converted Polygon
    """
    pointobb = aitool.bbox2pointobb(bbox)
    bbox_x = pointobb[0::2]
    bbox_y = pointobb[1::2]
    bbox_coord = [(x, y) for x, y in zip(bbox_x, bbox_y)]

    polygon = Polygon(bbox_coord)

    return polygon

def polygon2bbox(polygon):
    """convet polygon to single bbox

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([xmin, ymin, xmax, ymax])
    """
    pointobb = np.array(polygon.exterior.coords, dtype=int)[:-1].ravel().tolist()
    bbox = aitool.pointobb2bbox(pointobb)
    
    return bbox