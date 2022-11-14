import numpy as np
import cv2


def generate_image(height=512, 
                   width=512, 
                   color=(255, 255, 255)):
    """generate specific size image with specific color

    Args:
        height (int, optional): image height. Defaults to 512.
        width (int, optional): image width. Defaults to 512.
        color (tuple, optional): image init color. Defaults to (255, 255, 255).

    Returns:
        np.array: generated image
    """
    if type(color) == tuple:
        b = np.full((height, width, 1), color[0], dtype=np.uint8)
        g = np.full((height, width, 1), color[1], dtype=np.uint8)
        r = np.full((height, width, 1), color[2], dtype=np.uint8)
        img = np.concatenate((b, g, r), axis=2)
    else:
        gray = np.full((height, width), color, dtype=np.uint8)
        img = gray

    return img

def generate_subclass_mask(mask_image,
                           subclasses=(1, 3)):
    """extract the mask from mask image with specified classes

    Args:
        mask_image (np.array): input mask image
        subclasses (tuple, optional): specified class. Defaults to (1, 3).

    Returns:
        np.array: extracted the mask with class
    """
    mask_shape = mask_image.shape
    sub_mask = generate_image(mask_shape[0], mask_shape[1], color=0)
    if mask_image.ndim == 2:
        gray_mask_image = mask_image[:, :]
    else:
        gray_mask_image = mask_image[:, :, 0]
    
    if isinstance(subclasses, (list, tuple)):
        if len(subclasses) == 2:
            keep_bool = np.logical_or(gray_mask_image == subclasses[0], gray_mask_image == subclasses[1])
        else:
            keep = []
            for subclass in subclasses:
                keep.append(gray_mask_image == subclass)

            keep_bool = np.logical_or.reduce(tuple(keep))
    else:
        keep_bool = (gray_mask_image == subclasses)

    sub_mask[keep_bool] = 1

    return sub_mask