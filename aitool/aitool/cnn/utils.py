import torch


def convert_mmdet_pretrain(input_model_file,
                           output_mode_file,
                           class_num=21,
                           bbox_conv_num=4,
                           model_name='DetectoRS'):
    weights = torch.load(input_model_file)

    if model_name == 'Faster-R-CNN':
        weights['state_dict']['bbox_head.fc_cls.weight'].resize_(class_num, 1024)
        weights['state_dict']['bbox_head.fc_cls.bias'].resize_(class_num)
        weights['state_dict']['bbox_head.fc_reg.weight'].resize_(class_num * bbox_conv_num, 1024)
        weights['state_dict']['bbox_head.fc_reg.bias'].resize_(class_num * bbox_conv_num)
    elif model_name == 'DetectoRS':
        weights['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].resize_(class_num + 1, 1024)
        weights['state_dict']['roi_head.bbox_head.0.fc_cls.bias'].resize_(class_num + 1)

        weights['state_dict']['roi_head.bbox_head.1.fc_cls.weight'].resize_(class_num + 1, 1024)
        weights['state_dict']['roi_head.bbox_head.1.fc_cls.bias'].resize_(class_num + 1)

        weights['state_dict']['roi_head.bbox_head.2.fc_cls.weight'].resize_(class_num + 1, 1024)
        weights['state_dict']['roi_head.bbox_head.2.fc_cls.bias'].resize_(class_num + 1)

        weights['state_dict']['roi_head.mask_head.0.conv_logits.weight'].resize_(class_num, 256, 1, 1)
        weights['state_dict']['roi_head.mask_head.0.conv_logits.bias'].resize_(class_num)

        weights['state_dict']['roi_head.mask_head.1.conv_logits.weight'].resize_(class_num, 256, 1, 1)
        weights['state_dict']['roi_head.mask_head.1.conv_logits.bias'].resize_(class_num)

        weights['state_dict']['roi_head.mask_head.2.conv_logits.weight'].resize_(class_num, 256, 1, 1)
        weights['state_dict']['roi_head.mask_head.2.conv_logits.bias'].resize_(class_num)

        weights['state_dict']['roi_head.semantic_head.conv_logits.weight'].resize_(class_num, 256, 1, 1)
        weights['state_dict']['roi_head.semantic_head.conv_logits.bias'].resize_(class_num)
    
    torch.save(weights, output_mode_file)