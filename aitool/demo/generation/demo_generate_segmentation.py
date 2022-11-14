import aitool


if __name__ == "__main__":
    ann_file = './data/plane/v1/coco/annotations/plane_train.json'
    output_dir = './data/plane/v1/train/segmentation'

    aitool.GenerateSegBase(ann_file,
                            output_dir=output_dir,
                            output_format='.tif',
                            seg_key='pointobb',
                            binary=True,
                            sort_mode='keep_small')

