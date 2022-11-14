# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import force_fp32

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention, build_positional_encoding
from mmcv.cnn import ConvModule

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor

import torch
import torch.nn as nn

import numpy as np
import os

import matplotlib.pyplot as plt


@ROI_EXTRACTORS.register_module()
class MultiAttentionRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from given feature maps using multi-attention method.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        num_levels: Specify the number of input feature maps.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 # aggregation='sum',
                 attention=True,
                 attention_dropout=0.,
                 # attention_type='Channel',
                 num_head=1,
                 pre_cfg=None,
                 post_cfg=None,
                 enable_pos_embed=False,
                 post_fusion=False,
                 num_levels=3,
                 **kwargs):
        super(MultiAttentionRoIExtractor, self).__init__(**kwargs)

        self.local_rank = str(os.environ['LOCAL_RANK'])

        self.device = torch.device('cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else 'cpu')

        # assert aggregation in ['sum', 'concat']

        # self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]
        self.num_levels = num_levels
        self.attention = attention
        # self.attention_type = attention_type
        if self.attention:
            self.num_head = num_head  # a hyperparam to tune
            print(f'The number of multi-attention head is: {self.num_head}')
            self.multihead_attn = []
            self.multihead_attn_norm = []

            for _ in range(self.num_levels):
                self.multihead_attn.append(MultiheadAttention(256, self.num_head, dropout=attention_dropout).to(self.device))
                self.multihead_attn_norm.append(build_norm_layer(dict(type='LN'), 256)[1].to(self.device))
            self.enable_pos_embed = enable_pos_embed
            if self.enable_pos_embed:
                self.positional_encoding = build_positional_encoding(dict(type='SinePositionalEncoding',
                                                                          num_feats=128,
                                                                          normalize=True))
                mask = torch.zeros((1, 14, 14), dtype=torch.bool).to(self.device)

                self.pos_embed = self.positional_encoding(mask)  # 1,256,14,14
                self.pos_embed = self.pos_embed.flatten(2).permute(2, 0, 1).to(self.device)
        self.post_fusion = post_fusion
        if self.post_fusion:
            self.post_conv = ConvModule(256 * self.num_levels,
                                        256,
                                        1,  # kernel size
                                        padding=0,
                                        conv_cfg=None,
                                        norm_cfg=dict(type='BN'),
                                        act_cfg=dict(type='ReLU')).to(self.device)

    @force_fp32(apply_to=('feats', ), out_fp16=True)    ###### convert 'feats' to fp32 mode
    def forward(self, feats, rois, fusion_batchsize=256, roi_scale_factor=None):
        """
        Forward function.
        roi_labels: the roi labels of each roi, [0~10] where 10 means the roi is background
        """
        roi_size = self.roi_layers[0].output_size
        roi_feats = [None for _ in range(self.num_levels)]

        for n in range(self.num_levels):
            f = feats[n]  ###### torch.Size([4, 256, 256, 256]), 因为此时的featmap_strides=[4, 8, 16],所以可以将feats分成三种尺寸的特征图，即P2 P3 P4

            # some times rois is an empty tensor
            if rois.size(0) == 0:
                return feats[0].new_zeros(rois.size(0), self.out_channels, *roi_size)

            if roi_scale_factor is not None:
                rois = self.roi_rescale(rois, roi_scale_factor)

            roi_feats[n] = self.roi_layers[n](f, rois)  # .view(2048,256,14,14).contiguous()#.to(self.device)  ###### torch.Size([2048, 256, 14, 14]) --> torch.Size([2048, 256, 196])
            # feat_size = roi_feats_q_i.size(2)  ### 每个roi_feat的大小，此时为14

            roi_feats[n] = roi_feats[n].flatten(2)  ###### view的第一个维度不能直接是2048，因为有时候没有采样到2048个框
            # roi_feats[n] = roi_feats[n][pos_indx].permute(2, 0, 1)  ###### torch.Size([2048, 256, 196]) --> torch.Size([pos_roi_num, 256, 196]) --> torch.Size([196, pos_roi_num, 256])
            roi_feats[n] = roi_feats[n].permute(2, 0, 1)

        ### adopt multi-head attention operation
        attn_collect = []
        for n in range(self.num_levels):
            attn_output = []
            for i in range(0, roi_feats[0].shape[1], fusion_batchsize):
                if self.attention:
                    if self.enable_pos_embed:
                        actual_batchsize = roi_feats[0][:, i:i+fusion_batchsize, ...].shape[1]
                        pos_embed = torch.repeat_interleave(self.pos_embed, actual_batchsize, 1)  # 196,1,256 -> 196,batch,256
                    else:
                        pos_embed = None
                    temp_output = self.multihead_attn[n](roi_feats[0][:, i:i+fusion_batchsize, ...],
                                                         roi_feats[n][:, i:i+fusion_batchsize, ...],
                                                         roi_feats[n][:, i:i+fusion_batchsize, ...],
                                                         query_pos=pos_embed,
                                                         key_pos=pos_embed)
                    temp_output = self.multihead_attn_norm[n](temp_output)
                else:
                    temp_output = roi_feats[n][:, i:i+fusion_batchsize, ...]
                temp_output = temp_output.permute(1, 2, 0)
                temp_output = temp_output.reshape(temp_output.shape[0], temp_output.shape[1], *roi_size).contiguous()
                attn_output.append(temp_output)
            attn_collect.append(torch.cat(attn_output, dim=0))
        if self.post_fusion:
            final_result = torch.cat(attn_collect, dim=1)
            final_result = self.post_conv(final_result)
        else:
            final_result = 0
            for out in attn_collect:
                final_result += out / self.num_levels
        # print(attn_output.size(), attn_output_weights.size())
        # attn_output = attn_output.transpose(0,1).contiguous().transpose(1,2).contiguous()
        # attn_output = attn_output.view(attn_output.size(0),attn_output.size(1),feat_size,feat_size)
        # # print(attn_output.size())
        #
        # ori_roi_feats_P2 = self.roi_layers[0](fq, rois)
        # idx = 0
        # for i in pos_indx:
        #     ori_roi_feats_P2[i] = attn_output[idx]
        #     idx += 1


        # mark the starting channels for concat mode
        # start_channels = 0
        # for i in range(num_levels):
        #     roi_feats_t = self.roi_layers[i](feats[i], rois)  ###### [2048,256,14,14]
        #     # print(roi_feats_t.size())
        #     # assert False
        #     end_channels = start_channels + roi_feats_t.size(1)
        #     if self.with_pre:
        #         # apply pre-processing to a RoI extracted from each layer
        #         roi_feats_t = self.pre_module(roi_feats_t)
        #     if self.aggregation == 'sum':
        #         # and sum them all
        #         roi_feats += roi_feats_t
        #     else:
        #         # and concat them along channel dimension
        #         roi_feats[:, start_channels:end_channels] = roi_feats_t
        #     # update channels starting position
        #     start_channels = end_channels
        # # check if concat channels match at the end
        # if self.aggregation == 'concat':
        #     assert start_channels == self.out_channels
        #
        # if self.with_post:
        #     # apply post-processing before return the result
        #     roi_feats = self.post_module(roi_feats)
        # print(roi_feats.size()) ###### torch.Size([2048, 256, 14, 14])
        return final_result   ###### roi_feats
