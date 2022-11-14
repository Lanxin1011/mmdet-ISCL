import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import cv2
import aitool


def generate_polygon(mask_image, 
                     min_area=20):
    """convert the mask to polygon

    Args:
        mask_image (np.array): input mask image
        min_area (int, optional): threshold of area, when area < min_area, filter this object. Defaults to 20.

    Returns:
        list: list of polygons
    """
    contours = measure.find_contours(mask_image, 0.5, positive_orientation='low')

    polygons = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        if contour.shape[0] < 3:
            continue
        
        poly = Polygon(contour)
        if poly.area < min_area:
            continue
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.geom_type == 'MultiPolygon':
            for poly_ in poly:
                if poly_.area < min_area:
                    continue
                valid_flag = aitool.single_valid_polygon(poly_)
                if not valid_flag:
                    continue
                polygons.append(poly_)
        elif poly.geom_type == 'Polygon':
            valid_flag = aitool.single_valid_polygon(poly)
            if not valid_flag:
                continue
            polygons.append(poly)
        else:
            continue

    return polygons

def generate_polygon_opencv(mask_image, 
                            min_area=20):
    """convert the mask to polygon with OpenCV API

    Args:
        mask_image (np.array): input mask image
        min_area (int, optional): threshold of area, when area < min_area, filter this object. Defaults to 20.

    Returns:
        list: list of polygons
    """
    contours = cv2.findContours(mask_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    if len(contours) == 0:
        return []

    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        
        contour = np.array(contour).reshape(1, -1).tolist()[0]
        if len(contour) < 8:
            continue

        poly = aitool.mask2polygon(contour)
        if poly.area < min_area:
            continue
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.geom_type == 'MultiPolygon':
            for poly_ in poly:
                if poly_.area < min_area:
                    continue
                valid_flag = aitool.single_valid_polygon(poly_)
                if not valid_flag:
                    continue
                polygons.append(poly_)
        elif poly.geom_type == 'Polygon':
            valid_flag = aitool.single_valid_polygon(poly)
            if not valid_flag:
                continue
            polygons.append(poly)
        else:
            continue

    return polygons