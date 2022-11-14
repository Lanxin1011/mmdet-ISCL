import numpy as np
from shapely.geometry import Polygon


def polygon2mask(polygon):
    """convet polygon to mask

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    mask = np.array(polygon.exterior.coords, dtype=int)[:-1].ravel().tolist()
    return mask

def mask2polygon(mask):
    """convert mask to polygon

    Arguments:
        mask {list} -- contains coordinates of mask boundary ([x1, y1, x2, y2, ...])
    """
    mask_x = mask[0::2]
    mask_y = mask[1::2]
    mask_coord = [(x, y) for x, y in zip(mask_x, mask_y)]

    polygon = Polygon(mask_coord)

    return polygon

def mask2bbox(mask):
    """convert mask to bbox

    Arguments:
        mask {list} -- contains coordinates of mask boundary ([x1, y1, x2, y2, ...])
    """
    mask_x = mask[0::2]
    mask_y = mask[1::2]
    xmin, ymin, xmax, ymax = min(mask_x), min(mask_y), max(mask_x), max(mask_y)
    bbox = [xmin, ymin, xmax, ymax]

    return bbox