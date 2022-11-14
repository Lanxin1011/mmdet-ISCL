import os
import numpy as np
import cv2
import json
import lxml.etree as ET

import aitool


class XMLDumperBase():
    """save objects to xml format (voc)
    """
    def __init__(self,
                output_dir,
                classes=None):
        self.output_dir = output_dir
        aitool.mkdir_or_exist(self.output_dir)
        
        self.classes = classes

    def __call__(self, objects, image_fn):
        basename = aitool.get_basename(image_fn)
        root = ET.Element("annotations")
        ET.SubElement(root, "folder").text = 'VOC'
        ET.SubElement(root, "filename").text = image_fn
        
        for data in objects:
            obj = ET.SubElement(root, "object")
            if self.classes is not None:
                ET.SubElement(obj, "name").text = self.classes[data['category_id'] - 1]
            else:
                ET.SubElement(obj, "name").text = str(data['category_id'])
            ET.SubElement(obj, "type").text = "bndbox"

            bndbox = ET.SubElement(obj, "bndbox")
            bbox = data['bbox']
            xmin, ymin, xmax, ymax = aitool.xywh2xyxy(bbox)
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)
            
        tree = ET.ElementTree(root)
        tree.write(f"{self.output_dir}/{basename}.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')


class XMLDumperPlane(XMLDumperBase):
    """save objects to specific file format (plane competition, http://en.sw.chreos.org/)
    """
    def __call__(self, objects, image_fn, team='CAPTAIN-VIPG-Plane'):
        basename = aitool.get_basename(image_fn)
        root = ET.Element("annotation")
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "filename").text = basename + '.tif'
        ET.SubElement(source, "origin").text = 'GF2/GF3'

        research = ET.SubElement(root, "research")
        ET.SubElement(research, "version").text = "4.0"
        ET.SubElement(research, "provider").text = "Wuhan University"
        ET.SubElement(research, "author").text = team
        ET.SubElement(research, "pluginname").text = "Airplane Detection and Recognition in Optical Images"
        ET.SubElement(research, "pluginclass").text = "Detection"
        ET.SubElement(research, "time").text = "2020-07-2020-11"

        objects_handle = ET.SubElement(root, "objects")
        
        for data in objects:
            obj = ET.SubElement(objects_handle, "object")
            ET.SubElement(obj, "coordinate").text = "pixel"
            ET.SubElement(obj, "type").text = "rectangle"
            ET.SubElement(obj, "description").text = "None"

            possibleresult = ET.SubElement(obj, "possibleresult")
            if self.classes is not None:
                ET.SubElement(possibleresult, "name").text = self.classes[data['category_id'] - 1]
            else:
                ET.SubElement(possibleresult, "name").text = str(data['category_id'])
            ET.SubElement(possibleresult, "probability").text = str(data['score'])

            points = ET.SubElement(obj, "points")
            pointobb = data['pointobb']

            for idx in range(5):
                if idx == 4:
                    idx = 0
                ET.SubElement(points, "point").text = f"{pointobb[2 * idx]},{pointobb[2 * idx + 1]}"
            
        tree = ET.ElementTree(root)
        tree.write(f"{self.output_dir}/{basename}.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')


class XMLDumperRoVOC(XMLDumperBase):
    """save objects to rotated voc format
    """
    def __call__(self, objects, image_fn):
        basename = aitool.get_basename(image_fn)
        root = ET.Element("annotations")
        ET.SubElement(root, "folder").text = 'ROVOC'
        ET.SubElement(root, "filename").text = image_fn
        
        for data in objects:
            obj = ET.SubElement(root, "object")
            if self.classes is not None:
                ET.SubElement(obj, "name").text = self.classes[data['category_id'] - 1]
            else:
                ET.SubElement(obj, "name").text = str(data['category_id'])
            ET.SubElement(obj, "type").text = "robndbox"

            bndbox = ET.SubElement(obj, "robndbox")
            thetaobb = data['thetaobb']
            cx, cy, w, h, theta = thetaobb
            ET.SubElement(bndbox, "cx").text = str(cx)
            ET.SubElement(bndbox, "cy").text = str(cy)
            ET.SubElement(bndbox, "w").text = str(w)
            ET.SubElement(bndbox, "h").text = str(h)
            ET.SubElement(bndbox, "angle").text = str(theta)
            
        tree = ET.ElementTree(root)
        tree.write(f"{self.output_dir}/{basename}.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')
        

class ObjectDumperBase():
    def __init__(self,
                 output_dir,
                 output_format='.png',
                 min_area=5):
        self.output_dir = output_dir
        self.output_format = output_format
        self.min_area = min_area

    def __call__(self, img, objects, ori_image_fn):
        image_basename = aitool.get_basename(ori_image_fn)
        for data in objects:
            bbox = [int(_) for _ in data['bbox']]
            bbox = self._fix_bbox_bound(img, bbox)
            x, y, w, h = bbox

            if self._filter_invalid_bbox(bbox):
                continue

            img_save = img[y:y + h, x:x + w, :]
            output_fn = [image_basename]
            output_fn += [str(int(coord)) for coord in bbox]
            
            output_file = os.path.join(self.output_dir, "_".join(output_fn) + self.output_format)
            cv2.imwrite(output_file, img_save)

    def _filter_invalid_bbox(self, bbox):
        x, y, w, h = bbox

        if w * h < self.min_area or w < 2.0 or h < 2.0:
            return True

        return False

    def _fix_bbox_bound(self, img, bbox):
        height, width = img.shape[0], img.shape[1]
        x, y, w, h = bbox
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = width - x - 1 if x + w > width - 1 else w
        h = height - h - 1 if y + h > height - 1 else h

        return [x, y, w, h]

class ObjectDumperPlane(ObjectDumperBase):
    def __call__(self, img, objects, ori_image_fn):
        image_basename = aitool.get_basename(ori_image_fn)
        for data in objects:
            bbox = [int(_) for _ in data['bbox']]
            bbox = self._fix_bbox_bound(img, bbox)
            pointobb = data['pointobb']
            x, y, w, h = bbox

            if self._filter_invalid_bbox(bbox):
                continue

            img_save = img[y:y + h, x:x + w, :]
            output_fn = [image_basename]
            output_fn += [str(int(coord)) for coord in pointobb]
            output_file = os.path.join(self.output_dir, "_".join(output_fn) + self.output_format)
            cv2.imwrite(output_file, img_save)

class TXTDumperBase():
    """save objects to txt format (xmin, ymin, xmax, ymax, class)
    """
    def __init__(self,
                output_dir,
                classes=None):
        self.output_dir = output_dir
        aitool.mkdir_or_exist(self.output_dir)
        
        self.classes = classes

    def __call__(self, objects, image_fn):
        basename = aitool.get_basename(image_fn)
        with open(f"{self.output_dir}/{basename}.txt", 'w') as f:
            for data in objects:
                if self.classes is not None:
                    object_info = [self.classes[data['category_id'] - 1]]
                else:
                    object_info = [str(data['category_id'])]

                bbox = data['bbox']
                xmin, ymin, xmax, ymax = aitool.xywh2xyxy(bbox)
                object_info.extend([str(xmin), str(ymin), str(xmax), str(ymax)])
                
                f.write(" ".join(object_info))


class TXTDumperBase_HJJ_Ship(TXTDumperBase):
    """save objects to txt format (class, x1, y1, x2, y2, x3, y3, x4, y4)
    """
    def __call__(self, objects, image_fn):
        basename = aitool.get_basename(image_fn)
        with open(f"{self.output_dir}/{basename}.txt", 'w') as f:
            for data in objects:
                if self.classes is not None:
                    object_info = [self.classes[data['category_id'] - 1]]
                else:
                    object_info = [str(data['category_id'])]

                pointobb = data['pointobb']
                object_info.extend([str(_) for _ in pointobb] + ['\n'])
                
                f.write(" ".join(object_info))


class JSONDumperBase():
    """save data to a json file
    """
    def __init__(self):
        self.image_info = {"No image info!"}

    def _convert_data(self, label_info, image_info):
        if image_info is not None:
            self.image_info = image_info

        anns = []
        for label in label_info:
            objects = {}
            objects['bbox'] = label['bbox']
            objects['class'] = label['class']
            objects['segmentation'] = label['segmentation']

            anns.append(objects)

        json_data = {"image": self.image_info,
                     "annotations": anns}

        return json_data

    def dumper(self, label_info, image_info, json_file):
        json_data = self._convert_data(label_info, image_info)

        with open(json_file, "w") as jsonfile:
            json.dump(json_data, jsonfile, indent=4)


class JSONDumperBONAI(JSONDumperBase):
    def _convert_data(self, label_info, image_info):
        if image_info is not None:
            self.image_info = image_info

        bboxes = label_info['bbox']
        segmentations = label_info['segmentation']

        anns = []
        for bbox, segmentation in zip(bboxes, segmentations):
            objects = {}
            objects['bbox'] = bbox
            objects['class'] = 1
            objects['segmentation'] = segmentation

            anns.append(objects)

        json_data = {"image": self.image_info,
                     "annotations": anns}

        return json_data