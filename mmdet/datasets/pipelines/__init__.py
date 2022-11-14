# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
                           ContrastTransform, EqualizeTransform, Rotate, Shear,
                           Translate)
from .compose import Compose
from .formating import (Collect, DefaultFormatBundle, ImageToTensor,
                        ToDataContainer, ToTensor, Transpose, to_tensor)
from .instaboost import InstaBoost
from .loading import (LoadAnnotations, LoadImageFromFile, LoadImageFromWebcam,
                      LoadMultiChannelImageFromFiles, LoadProposals)
from .test_time_aug import MultiScaleFlipAug
from .transforms import (Albu, CutOut, Expand, MinIoURandomCrop, MixUp, Mosaic,
                         Normalize, Pad, PhotoMetricDistortion, RandomAffine,
                         RandomCenterCropPad, RandomCrop, RandomFlip,
                         RandomShift, Resize, SegRescale)
from .instance_switch2 import InstanceSwitch2  # randomly select instance without KNN fitting
from .instance_switch_knn import InstanceSwitchKNN  # randomly select instance with KNN fitting
from .instance_switch2_fair1m import InstanceSwitch2FAIR  # used in the random switch condition on fair1m dataset
from .instance_switch_by_class_serving import InstanceSwitchByClassServing  # used in the serving target version of ISCL for GF dataset
from .instance_switch_by_class_serving_fair1m import InstanceSwitchByClassServingFAIR  # used in the serving target version of ISCL for fair1m dataset

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'MultiScaleFlipAug',
    'Resize', 'RandomFlip', 'Pad', 'RandomCrop', 'Normalize', 'SegRescale',
    'MinIoURandomCrop', 'Expand', 'PhotoMetricDistortion', 'Albu',
    'InstaBoost', 'RandomCenterCropPad', 'AutoAugment', 'CutOut', 'Shear',
    'Rotate', 'ColorTransform', 'EqualizeTransform', 'BrightnessTransform',
    'ContrastTransform', 'Translate', 'RandomShift', 'Mosaic', 'MixUp',
    'RandomAffine', 'InstanceSwitch2', 'InstanceSwitchByClassServing',
    'InstanceSwitch2FAIR', 'InstanceSwitchByClassServingFAIR', 'InstanceSwitchKNN'
]
