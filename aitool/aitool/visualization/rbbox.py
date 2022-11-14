import numpy as np
import cv2


def draw_rectangle_by_points(img, point, color=(0, 0, 255)):
    """draw rectangle by points

    Args:
        img (np.array): input image
        point (list): [x1, y1, x2, y2, ...]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: image with points
    """
    for idx in range(-1, 3, 1):
        cv2.line(img, (int(point[idx*2]), int(point[idx*2+1])), (int(point[(idx+1)*2]), int(point[(idx+1)*2+1])), color, 3)

    return img

def show_bbox(img, bbox, color=(0, 0, 255)):
    """show rectangle (bbox)

    Args:
        img (np.array): input image
        bbox (list): [xmin, ymin, xmax, ymax]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: output image
    """
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 3)

    return img

def show_pointobb(img, pointobb, color=(0, 0, 255)):
    """show pointobb on img

    Args:
        img (np.array): input image
        pointobb (list): [x1, y1, x2, y2, x3, y3, x4, y4]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: image with points
    """
    img = draw_rectangle_by_points(img, pointobb, color=color)
    
    return img

def show_thetaobb(img, thetaobb, color=(0, 0, 255)):
    """show single theteobb

    Args:
        im (np.array): input image
        thetaobb (list): [cx, cy, w, h, theta]
        color (tuple, optional): draw color. Defaults to (0, 0, 255).

    Returns:
        np.array: image with thetaobb
    """
    cx, cy, w, h, theta = thetaobb

    rect = ((cx, cy), (w, h), theta / np.pi * 180.0)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    cv2.drawContours(img, [rect], -1, color, 3)

    return img