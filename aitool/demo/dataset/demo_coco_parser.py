import aitool


if __name__ == "__main__":
    ann_file = './data/plane/v1/coco/annotations/plane_train.json'

    coco_parser = aitool.COCOParser(ann_file, 
                                  data_keys=['pointobb', 'category_id'])

    for img_fn in coco_parser.img_fns:
        objects = coco_parser(img_fn)