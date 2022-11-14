import aitool


if __name__ == "__main__":
    pkl_file = '../mmdetv2-bc/results/plane/pl_v001_mask_rcnn_x101_32x4d_fpn_2x/pl_v001_mask_rcnn_x101_32x4d_fpn_2x_coco_results.pkl'
    ann_file = './data/plane/v1/coco/annotations/plane_val.json'
    img_dir = './data/plane/v1/val/images'

    pkl_parser = aitool.PklParserMask(pkl_file, ann_file)

    image_list = aitool.get_file_list(img_dir, '.tif')

    for image_file in image_list:
        basename = aitool.get_basename(image_file)
        objects = pkl_parser(basename)

        print("number of objects: ", len(objects))