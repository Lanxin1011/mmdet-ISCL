import aitool


if __name__ == '__main__':
    ann_file = './data/plane/v1/coco/annotations/plane_train.json'

    coco_statistic = aitool.COCOStatistic_Plane(ann_file, data_keys=['thetaobb', 'category_id', 'bbox'])

    coco_statistic.class_distribution()