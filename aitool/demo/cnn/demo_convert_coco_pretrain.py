import aitool


if __name__ == '__main__':
    input_model_file = '/home/jwwangchn/Downloads/detectors_htc_r50_1x_coco-329b1453.pth'
    output_model_file = '/home/jwwangchn/Downloads/detectors_htc_r50_1x_coco-329b1453_class_10.pth'

    aitool.convert_mmdet_pretrain(input_model_file,
                                  output_model_file,
                                  class_num=10,
                                  model_name='DetectoRS')