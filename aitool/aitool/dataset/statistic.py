import numpy as np
from collections import defaultdict

import aitool


class COCOStatisticBase():
    def __init__(self,
                 ann_file,
                 output_format='.png',
                 data_keys=['bbox', 'category_id', 'segmentation']):
        self.coco_parser = aitool.COCOParser(ann_file, data_keys=data_keys)
        categories = self.coco_parser.categories
        self.coco_class = dict()
        for category in categories:
            self.coco_class[category['id']] = category['name']
        self.img_fns = self.coco_parser.img_fns

        np.set_printoptions(precision=2, suppress=True)

    def class_distribution(self):
        class_nums = defaultdict(lambda: 0)
        class_sizes = defaultdict(list)
        class_wh_ratios = defaultdict(list)
        for img_fn in self.img_fns:
            objects = self.coco_parser(img_fn)
            for obj in objects:
                class_nums[self.coco_class[obj['category_id']]] += 1
                class_sizes[self.coco_class[obj['category_id']]].append(np.sqrt(obj['bbox'][2] * obj['bbox'][3]))
                h, w = (obj['bbox'][2], obj['bbox'][3]) if obj['bbox'][2] > obj['bbox'][3] else (obj['bbox'][3], obj['bbox'][2])
                class_wh_ratios[self.coco_class[obj['category_id']]].append(h / w)

        class_nums = {class_name: class_nums[class_name] for class_name in sorted(class_nums)}
        class_sizes = {class_name: int(np.array(class_sizes[class_name]).mean()) for class_name in sorted(class_sizes)}
        class_wh_ratios = {class_name: np.array(class_wh_ratios[class_name]).mean() for class_name in sorted(class_wh_ratios)}
        
        print(f"Class num distribution: {class_nums}")
        print(f"Class average size distribution: {class_sizes}")
        print(f"Class average wh ratio distribution: {class_wh_ratios}")

    def instance_distribution(self):
        pass

    def image_distribution(self):
        pass


class COCOStatistic_Plane(COCOStatisticBase):
    def class_distribution(self):
        class_nums = defaultdict(lambda: 0)
        class_sizes = defaultdict(list)
        class_wh_ratios = defaultdict(list)
        for img_fn in self.img_fns:
            objects = self.coco_parser(img_fn)
            for obj in objects:
                class_nums[self.coco_class[obj['category_id']]] += 1
                class_sizes[self.coco_class[obj['category_id']]].append(np.sqrt(obj['thetaobb'][2] * obj['thetaobb'][3]))
                h, w = (obj['thetaobb'][2], obj['thetaobb'][3]) if obj['thetaobb'][2] > obj['thetaobb'][3] else (obj['thetaobb'][3], obj['thetaobb'][2])
                class_wh_ratios[self.coco_class[obj['category_id']]].append(h / w)

        class_nums = {class_name: class_nums[class_name] for class_name in sorted(class_nums)}
        class_sizes = {class_name: int(np.array(class_sizes[class_name]).mean()) for class_name in sorted(class_sizes)}
        class_wh_ratios = {class_name: np.array(class_wh_ratios[class_name]).mean() for class_name in sorted(class_wh_ratios)}
        
        print(f"Class num distribution: {class_nums}")
        print(f"Class average size distribution: {class_sizes}")
        print(f"Class average wh ratio distribution: {class_wh_ratios}")