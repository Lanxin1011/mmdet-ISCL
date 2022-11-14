import numpy as np
import cv2

import aitool


def split_image(img, 
                subsize=(1024, 1024), 
                gap=200,
                mode='keep_all', 
                expand_boundary=True):
    """split the original image to sub-images

    Args:
        img (np.array): input image
        subsize (tuple, optional): the size of sub-image. Defaults to [1024, 1024].
        gap (int, optional): the gap of sliding windows. Defaults to 200.
        mode (str, optional): keep all or drop boundary. Defaults to 'keep_all'.
        expand_boundary (bool, optional): if original image size < subsize, boundary will be expanded. Defaults to True.

    Returns:
        dict: splitted sub-images
    """
    if isinstance(img, str):
        img = cv2.imread(img)

    if img is None:
        print("This image is empty")
        return None

    if isinstance(subsize, tuple) or isinstance(subsize, list):
        sub_width, sub_height = subsize
    elif isinstance(subsize, float) or isinstance(subsize, int):
        sub_width, sub_height = subsize, subsize
    else:
        raise(TypeError(f"Error input subsize value type: {type(subsize)}"))
    
    img_height, img_width = img.shape[0], img.shape[1]

    start_xs = np.arange(0, img_width, sub_width - gap)
    if mode == 'keep_all':
        start_xs[-1] = img_width - sub_width if img_width - start_xs[-1] <= sub_width else start_xs[-1]
    elif mode == 'drop_boundary':
        if img_width - start_xs[-1] < sub_width - gap:
            start_xs = np.delete(start_xs, -1)
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, sub_height - gap)
    if mode == 'keep_all':
        start_ys[-1] = img_height - sub_height if img_height - start_ys[-1] <= sub_height else start_ys[-1]
    elif mode == 'drop_boundary':
        if img_height - start_ys[-1] < sub_height - gap:
            start_ys = np.delete(start_ys, -1)
    start_ys[-1] = np.maximum(start_ys[-1], 0)

    subimages = dict()
    
    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + sub_width, img_width)
            end_y = np.minimum(start_y + sub_height, img_height)
            if expand_boundary:
                subimage = aitool.generate_image(sub_height, sub_width, color=(0, 0, 0))
                subimage[0:end_y-start_y, 0:end_x-start_x, ...] = img[start_y:end_y, start_x:end_x, ...]
            else:
                subimage = img[start_y:end_y, start_x:end_x, ...]
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage

    return subimages