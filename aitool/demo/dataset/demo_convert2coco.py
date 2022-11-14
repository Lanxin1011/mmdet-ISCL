import xml.etree.ElementTree as ET

import aitool


class Convert2COCO_Plane(aitool.Convert2COCO):
    def _dataset_parser(self, image_file, label_file):
        objects = []
        tree = ET.parse(label_file)
        root = tree.getroot()
        objects_handle = root.find('objects')
        for single_object in objects_handle.findall('object'):
            points = single_object.find('points')
            object_struct = {}

            pointobb = []
            for point in points[:-1]:
                coords = [float(coord) for coord in point.text.split(',')]
                pointobb += coords

            bbox = aitool.pointobb2bbox(pointobb)
            bbox = aitool.xyxy2xywh(bbox)

            object_struct['area'] = bbox[2] * bbox[3]
            object_struct['segmentation'] = [pointobb]
            object_struct['pointobb'] = pointobb
            object_struct['bbox'] = bbox
            object_struct['category_id'] = classes[single_object.find('possibleresult').find('name').text]
            
            objects.append(object_struct)

        return objects


if __name__ == "__main__":
    image_dir = './data/plane/v1/train/images'
    label_dir = './data/plane/v1/train/labels'
    output_file = './data/plane/v1/coco/annotations/plane_train.json'

    k_fold = 5

    image_format, label_format = '.tif', '.xml'

    classes = {'Boeing737': 1, 'Boeing747': 2, 'Boeing777': 3, 'Boeing787': 4, 'A220': 5, 'A321': 6, 'A330': 7, 'A350': 8, 'ARJ21': 9, 'other': 10}

    coco_classes = [{'supercategory': 'plane', 'id': 1,  'name': 'Boeing737',       },
                    {'supercategory': 'plane', 'id': 2,  'name': 'Boeing747',       },
                    {'supercategory': 'plane', 'id': 3,  'name': 'Boeing777',       },
                    {'supercategory': 'plane', 'id': 4,  'name': 'Boeing787',       },
                    {'supercategory': 'plane', 'id': 5,  'name': 'A220',            },
                    {'supercategory': 'plane', 'id': 6,  'name': 'A321',            },
                    {'supercategory': 'plane', 'id': 7,  'name': 'A330',            },
                    {'supercategory': 'plane', 'id': 8,  'name': 'A350',            },
                    {'supercategory': 'plane', 'id': 9,  'name': 'ARJ21',           },
                    {'supercategory': 'plane', 'id': 10, 'name': 'other',           }]

    Convert2COCO_Plane(image_dir=image_dir, 
                       label_dir=label_dir, 
                       output_file=output_file,
                       image_format=image_format, 
                       label_format=label_format,
                       img_size=(1024, 1024),
                       categories=coco_classes,
                       k_fold=k_fold)


    