import cv2
import numpy as np

import aitool


def show_image(img, 
               output_file=None,
               win_name='',
               win_size=800,
               wait_time=0):
    """show image

    Args:
        img (np.array): input image
        win_name (str, optional): windows name. Defaults to ''.
        win_size (int, optional): windows size. Defaults to 800.
        wait_time (int, optional): wait time . Defaults to 0.
        output_file ([type], optional): save the image. Defaults to None.

    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_size, win_size)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)
    if output_file != None:
        dir_name = aitool.get_dir_name(output_file)
        aitool.mkdir_or_exist(dir_name)

        cv2.imwrite(output_file, img)