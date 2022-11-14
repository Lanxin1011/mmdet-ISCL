import os
import numpy as np
import cv2
import tqdm
from pycocotools.coco import COCO

import aitool


class GenerateSegBase():
    def __init__(self, 
                 ann_file,
                 output_dir,
                 output_format='.png',
                 seg_key='segmentation',
                 binary=True,
                 sort_mode='keep_small'):
        self.seg_key = seg_key
        self.sort_mode = sort_mode
        aitool.mkdir_or_exist(output_dir)
        coco_parser = aitool.COCOParser(ann_file, data_keys=['category_id'] + [self.seg_key])

        print("begin to convert segmentation to png file")
        for img_fn in tqdm.tqdm(coco_parser.img_fns):
            png_file = os.path.join(output_dir, img_fn + output_format)
            objects = coco_parser(img_fn)

            self._segmentation2png(objects, png_file, binary=binary)

    def _segmentation2png(self, objects, png_file, binary=True):
        if binary:
            foreground_color = 0
        else:
            foreground_color = (0, 0, 0)
        if len(objects) != 0:
            img_height, img_width = objects[0]['img_height'], objects[0]['img_width']
            foreground = aitool.generate_image(img_height, img_width, foreground_color)
        else:
            foreground = aitool.generate_image(1024, 1024, foreground_color)

        objects = self._sort_objects(objects, mode=self.sort_mode)
        for idx, data in enumerate(objects):
            segmentation = data[self.seg_key]
            label = data['category_id']
            color = self._label2color(label, index=idx, binary=binary)
            segmentation = np.array(segmentation, dtype=np.int32).reshape(1, -1, 2)
            cv2.fillPoly(foreground, segmentation, color)

        cv2.imwrite(png_file, foreground)

    def _sort_objects(self, objects, mode='random'):
        if mode == 'random':
            return objects
        elif mode == 'keep_small':
            areas = [aitool.mask2polygon(data[self.seg_key]).area for data in objects]
            index = np.argsort(areas)[::-1]
            return np.array(objects)[index].tolist()
        elif mode == 'keep_large':
            areas = [aitool.mask2polygon(data[self.seg_key]).area for data in objects]
            index = np.argsort(areas)
            return np.array(objects)[index].tolist()
        else:
            raise RuntimeError(f'do not support the sort mode: {mode}')

    def _label2color(self, label, index, binary=True):
        if binary:
            color = label
        else:
            color_list = list(aitool.COLORS.keys())
            color = (aitool.COLORS[color_list[index % 20]][2], aitool.COLORS[color_list[index % 20]][1], aitool.COLORS[color_list[index % 20]][0])

        return color
