import os

import aitool


if __name__ == '__main__':
    image_dir = '/data/plane/v1/train/images'
    ann_file = '/data/plane/v1/coco/annotations/plane_train_K1.json'

    coco_parser = aitool.COCOParser(ann_file)

    for img_fn in coco_parser.img_fns:
        anns = coco_parser(img_fn)

        image_file = os.path.join(image_dir, img_fn + '.tif')
        aitool.show_coco_mask(coco_parser.coco, image_file, anns)