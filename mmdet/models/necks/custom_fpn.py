import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class CustomFPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_indx = 0, ### 指示输出的叠加层，P2-P5
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(CustomFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.out_indx= out_indx ### add

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()  ### 定义modulelist来存储module
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = None
            fpn_conv = None
            if i >= min(out_indx):
                l_conv = ConvModule(                                   ### 即1×1的lateral部分
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            if i in out_indx:
                fpn_conv = ConvModule(                                 ### 即3×3的lateral部分
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)

            self.lateral_convs.append(l_conv)   ### 将module存入对应list
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        # print("=======================FPN======================")
        # print(self)
        # exit()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        ### 此处为 卷积核 和 步长 均为1的横向卷积
        ### 得到C2-C5的1×1横向卷积后的结果


        laterals = [
            lateral_conv(inputs[i + self.start_level]) if lateral_conv is not None else None
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]


        # build top-down path
        used_backbone_levels = len(laterals) ### 存放使用到的backbone卷积层的数量
        for i in range(used_backbone_levels - 1, min(self.out_indx), -1): ### 倒序 遍历
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg) # 解析输入参数中字典中的值，然后以关键字参数放入函数中
            else:
                prev_shape = laterals[i - 1].shape[2:]
                ### C5横向卷积后直接得到M5，
                ### 首先将C5上采样×2后加到C4横向卷积后的结果上得到M4，后面以此类推
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        ### 针对叠加后的每一层结果M2-M5 又进行了一次3×3的卷积，
        ### 此时是为了消除上采样带来的混叠影响
        if len(self.out_indx) < self.num_outs:
            outs = [self.fpn_convs[i](laterals[i]) if self.fpn_convs[i] is not None else None for i in self.out_indx]
            # print(f'the selected out_channel is P{[i for i in self.out_indx]}')
        else:
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
        # part 2: add extra levels
            if self.num_outs > len(outs):  ### 此时输出的层数要大于使用到的backbone卷积层数量，即used_backbone_levels
                # use max pool to get more levels on top of outputs
                # (e.g., Faster R-CNN, Mask R-CNN)
                if not self.add_extra_convs:
                    for i in range(self.num_outs - used_backbone_levels):
                        outs.append(F.max_pool2d(outs[-1], 1, stride=2)) ### 对最上层（e.g.M5 进行步长为二的最大池化，
                                                                         ### 相当于下采样）
                # add conv layers on top of original feature maps (RetinaNet)
                else:
                    if self.add_extra_convs == 'on_input':
                        extra_source = inputs[self.backbone_end_level - 1]
                    elif self.add_extra_convs == 'on_lateral':
                        extra_source = laterals[-1]
                    elif self.add_extra_convs == 'on_output':
                        extra_source = outs[-1]
                    else:
                        raise NotImplementedError
                    outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                    for i in range(used_backbone_levels + 1, self.num_outs):
                        if self.relu_before_extra_convs:
                            outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                        else:
                            outs.append(self.fpn_convs[i](outs[-1]))

        picked_outs = []
        for out in outs:
            if out is not None:
                picked_outs.append(out)

        return tuple(picked_outs)
