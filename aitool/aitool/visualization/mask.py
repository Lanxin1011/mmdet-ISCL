import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_coco_mask(coco, image_file, anns, output_file=None):
    if isinstance(image_file, str):
        img = cv2.imread(image_file)
    elif isinstance(image_file, np.ndarray):
        img = image_file.copy()
    else:
        raise("Wrong input image type!", type(image_file))

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    coco.showAnns(anns)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=600, pad_inches=0.0)
        plt.clf()
    else:
        plt.show()

def draw_mask_boundary(img, mask, color=(0, 0, 255), thickness=2):
    """draw boundary of masks

    Args:
        img (np.array): input image
        masks (list): list of masks
        color (tuple, optional): color of boundary. Defaults to (0, 0, 255).
        thickness (int, optional): thickness of line. Defaults to 3.

    Returns:
        np.array: image with mask boundary
    """
    mask = np.array(mask).reshape((-1, 1, 2))
    img = cv2.polylines(img, [mask], True, color, thickness=thickness, lineType=cv2.LINE_AA)

    return img