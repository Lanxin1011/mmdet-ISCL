import os
import numpy as np
import cv2
import json
import random
import tqdm

import aitool


class Convert2COCO():
    def __init__(self,
                image_dir,
                label_dir,
                output_file,
                image_format='.png',
                label_format='.txt',
                imageset_file=None,
                expand_image_list=None,
                dataset_info=None,
                dataset_licenses=None,
                dataset_type='instance',
                categories=None,
                img_size=None,
                with_groundtruth=True,
                min_area=10,
                k_fold=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.imageset_file = imageset_file
        self.dataset_info = dataset_info
        self.dataset_licenses = dataset_licenses
        self.dataset_type = dataset_type
        self.categories = categories

        self.image_format = image_format
        self.label_format = label_format
        self.img_size = img_size
        self.with_groundtruth = with_groundtruth
        self.min_area = min_area
        self.small_object_counter = 0
        
        self.max_object_num_per_image = 0
        self.min_object_length = 2048 * 2048
        self.max_object_length = 0

        self.expand_image_list = self._parse_expand_image_list(expand_image_list)

        image_list = self._get_image_list(k_fold=k_fold)

        if k_fold == 0:
            self._dump_coco_json(image_list, output_file)
        else:
            for fold_idx in range(k_fold):
                sub_image_list = image_list[fold_idx]

                val_output_file = output_file[0:-5] + f'_K{fold_idx + 1}' + output_file[-5:]
                self._dump_coco_json(sub_image_list, val_output_file)

    def _parse_expand_image_list(self, expand_image_list):
        if isinstance(expand_image_list, list):
            return expand_image_list
        elif isinstance(expand_image_list, str):
            return_list = []
            with open(expand_image_list, 'r') as f:
                lines = f.readlines()
            for line in lines:
                return_list.append(aitool.get_basename(line.strip()))
            return return_list
        elif expand_image_list is None:
            return expand_image_list
        else:
            raise NotImplementedError(f"don't support the type of expand_image_list: {type(expand_image_list)}")

    def _dump_coco_json(self, image_list, output_file):
        images, annotations = self._get_image_annotation_pairs(image_list)
        if self.categories is None:
            raise RuntimeError("please input the coco categories")
        json_data = {"info": self.dataset_info,
                    "images": images,
                    "licenses": self.dataset_licenses,
                    "type": self.dataset_type,
                    "annotations": annotations,
                    "categories" : self.categories}
        
        with open(output_file, "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)        

    def _split_image_list(self, image_list, k_fold):
        length = len(image_list)
        step = length // k_fold
        result = []
        for idx in range(0, length, step):
            splitted_list = image_list[idx: idx + step]
            result.append(splitted_list)

        return result

    def _get_image_list(self, k_fold=0):
        if self.imageset_file is not None:
            print(f"loading image list from imageset file: {self.imageset_file}")
            raise NotImplementedError
        else:
            print(f"loading image list from image dir: {self.image_dir}")
            random.seed(0)
            image_file_list = aitool.get_file_list(self.image_dir, self.image_format)
            image_list = [aitool.get_basename(image_file) for image_file in image_file_list]
            image_list.sort()
            random.shuffle(image_list)
        
        if k_fold == 0:
            if self.expand_image_list is None:
                return image_list
            else:
                return image_list + self.expand_image_list
        else:
            splitted_image_list = self._split_image_list(image_list, k_fold)
            return splitted_image_list

    def _get_image_annotation_pairs(self, image_list):
        images, annotations = [], []
        ann_idx = 0
        img_idx = 0
        for image_basename in tqdm.tqdm(image_list):
            img_idx += 1
            image_file = os.path.join(self.image_dir, image_basename + self.image_format)
            label_file = os.path.join(self.label_dir, image_basename + self.label_format)

            if self.img_size is not None:
                img_height, img_width = self.img_size
            else:
                img = cv2.imread(image_file)
                if img is not None:
                    img_height, img_width = img.shape[0], img.shape[1]
                else:
                    print(f"This image {image_file} is empty")
                    img_idx -= 1
                    continue

            images.append({'date_captured': 2020,
                           'file_name': image_basename + self.image_format,
                           'id': img_idx,
                           "url": "http://jwwangchn.cn",
                           "height": img_height,
                           "width": img_width})

            if not self.with_groundtruth:
                continue

            objects = self._dataset_parser(image_file, label_file)
            
            if len(objects) != 0:
                data_keys = objects[0].keys()
                if 'bbox' not in data_keys or 'category_id' not in data_keys:
                    raise RuntimeError(f"objects need to contain item of 'bbox' and 'category_id'")
            else:
                # delete the image with zero object
                images.pop()
                img_idx -= 1
                continue
            
            for data in objects:
                ann_idx += 1
                data['id'] = ann_idx
                data['image_id'] = img_idx
                if 'segmentation' in data:
                    if len(data['segmentation']) > 1:
                        data['segmentation'] = [data['segmentation']]
                if 'iscrowd' not in data:
                    data["iscrowd"] = 0
                if data['area'] < self.min_area:
                    self.small_object_counter += 1
                
                if np.sqrt(data['area']) < self.min_object_length:
                    self.min_object_length = int(np.sqrt(data['area']))
                if np.sqrt(data['area']) > self.max_object_length:
                    self.max_object_length = int(np.sqrt(data['area']))

                annotations.append(data)

            if len(objects) > self.max_object_num_per_image:
                self.max_object_num_per_image = len(objects)

            if img_idx % (len(image_list) // (len(image_list) if len(image_list) > 20 else len(image_list))) == 0 or img_idx == len(image_list) - 1:
                print(f"Image ID: {img_idx}, Instance ID: {ann_idx}, Small Object Counter: {self.small_object_counter}, Max Object Number: {self.max_object_num_per_image}, Min Object Area: {self.min_object_length}")
            
        print("Summary: ")
        print(f"Image ID: {img_idx}, Instance ID: {ann_idx}, Small Object Counter: {self.small_object_counter}, Max Object Per Image: {self.max_object_num_per_image}, Min Object Length: {self.min_object_length}, Max Object Length: {self.max_object_length}")

        return images, annotations

    def _dataset_parser(self, image_file, label_file):
        raise NotImplementedError
