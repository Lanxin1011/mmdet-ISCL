import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import numpy as np
import random
import os
from mmdet.models.losses import TripletLoss


@HEADS.register_module()
class StandardRoIHead2DeTri(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 stage_loss_weights=[1, 0.5],
                 num_stages=2,
                 enable_con_loss=False,
                 con_loss_type=None,
                 con_weight=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.local_rank = str(os.environ['LOCAL_RANK'])
        self.device = torch.device('cuda:{}'.format(self.local_rank) if torch.cuda.is_available() else 'cpu')

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.enable_con_loss = enable_con_loss
        if self.enable_con_loss:
            assert con_loss_type is not None
            self.con_loss_type = con_loss_type
            self.con_w = con_weight
        print(f'enable_con_loss:{self.enable_con_loss}')
        # print(f'con_loss_type:{self.con_loss_type}')
        super(StandardRoIHead2DeTri, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        # self.bbox_assigner = []
        # self.bbox_sampler = []
        # print(self.train_cfg)
        # bbox_assigner = self.train_cfg['assigner']
        # bbox_sampler = self.train_cfg['sampler']
        # if not isinstance(bbox_assigner, list):
        #     bbox_assigner = [bbox_assigner for _ in range(self.num_stages)]
        # if not isinstance(bbox_sampler, list):
        #     bbox_sampler = [bbox_sampler for _ in range(self.num_stages)]
        # if self.train_cfg is not None:
        #     for idx in range(len(bbox_assigner)):
        #         # print(rcnn_train_cfg)
        #         self.bbox_assigner.append(
        #             build_assigner(bbox_assigner[idx]))
        #         self.current_stage = idx
        #         self.bbox_sampler.append(
        #             build_sampler(bbox_sampler[idx], context=self))

        self.bbox_assigner = []
        self.bbox_sampler = []

        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                # print(rcnn_train_cfg)
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)

        # print(bbox_feats.size()) ### torch.Size([2048, 256, 14, 14])
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # print(f'roi size: {rois.size()}')

        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        bbox_results = self._bbox_forward(stage, x, rois)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            '''
            Each element in the 'sampling_results' is a dictionary, and each dictionary includes the following key-value pair:
                    'neg_bboxes' 'neg_inds' 'num_gts' 'pos_assigned_gt_inds' 'pos_bboxes' 'pos_inds' 'pos_is_gt'

            '''
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # print(pos_is_gts)
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                # print(roi_labels)
                # print(roi_labels.shape)

                # roi_gt_bboxes = bbox_results['bbox_targets'][4]  # when batchsize=4, roi_gt_bboxes.shape=torch.Size([2048, 4])
                # print(roi_gt_bboxes[0])  # 'rois' has an extra column to keep abatch_num(0~batchsize) compared to 'roi_gt_bboxes'

                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    # if cls_score.numel() == 0:
                    #     break

                    # roi_labels = torch.where(        # if meet the conditions, get element from cls_score[:, :-1].argmax(1), else, from 'roi_labels'
                    #     roi_labels == self.bbox_head[i].num_classes,
                    #     cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

            if self.enable_con_loss:
                if i == self.num_stages - 1:
                    roi_feats = bbox_results['bbox_feats']  # [(2048),256,14,14]
                    roi_labels = bbox_results['bbox_targets'][0]
                    assert roi_feats.shape[0] == roi_labels.shape[
                        0], 'The batchsize of roi features and roi lables should be the same!'
                    # get positive roi index
                    # with torch.no_grad():
                    roi_labels_cpu = np.array(roi_labels.cpu())
                    pos_indx = np.argwhere(roi_labels_cpu < 10)  # background number: 10, foregroung number: 0~9
                    pos_indx = [i for k in pos_indx for i in k]
                    # put all positive RoI features into a list，and call calc_con_loss() to calculate loss
                    pos_roi_labels = []
                    pos_roi_feats = []
                    ###### randomly select no larger than 200 positive roi features to avoid the computational burden caused by too much positive RoIs in the anaphase
                    max_pos_num = 200
                    select_idx = random.sample(pos_indx, min(max_pos_num, len(pos_indx)))
                    for m in select_idx:
                        pos_roi_feats.append(roi_feats[m])
                        pos_roi_labels.append(roi_labels[m])
                    assert len(select_idx) == len(pos_roi_feats) == len(pos_roi_labels)

                    # for k in pos_indx:
                    #     pos_roi_feats.append(roi_feats[k])
                    #     pos_roi_labels.append(roi_labels[k])

                    con_loss = self.calc_con_loss(pos_roi_feats, pos_roi_labels)
                    # if con_loss is not 0 and con_loss is not torch.tensor(0.):
                    #     # bbox_results['loss_bbox']['loss_cls'] += self.con_w * con_loss[0][0]
                    #     losses[f's{i}.loss_con'] = self.con_w * con_loss
                    # # else:
                    # #     losses[f's{i}.loss_con'] = torch.float(0)
                    if con_loss is not 0 and con_loss is not torch.tensor(0.):  # and con_loss is not
                        losses[f's{i}.loss_con'] = self.con_w * con_loss
                    else:
                        losses[f's{i}.loss_con'] = torch.tensor(0.).cuda()   #.to(self.device)
                        # print(con_loss)

        # print(losses)
        return losses

    def _calc_conloss_single(self, q_indx, roi_feats, roi_labels, max_num=30, tao=0.1):
        '''

        :param q_indx: index of current query sample
        :param roi_feats: all positive roi features of this batch with shape of (256,1)
        :param roi_labels: labels of positive roi features
        :param max_num: the maximum number of pos/neg samples
        :param tao:
        :return:
        '''
        q_cls = roi_labels[q_indx]
        q_feat = roi_feats[q_indx].permute(1, 0)  # torch.size([256, 1])
        pos_feats = []
        neg_feats = []
        for i in range(len(roi_feats)):
            if i == q_indx:
                continue
            else:
                cur_cls = roi_labels[i]
                if cur_cls == q_cls:
                    pos_feats.append(roi_feats[i].permute(1, 0))
                else:
                    neg_feats.append(roi_feats[i].permute(1, 0))
        # print(len(roi_feats), len(pos_feats), len(neg_feats))
        if len(pos_feats) == 0 or len(neg_feats) == 0:
            return 0

        else:
            if self.con_loss_type == 'infoNCE':
                con_loss_single = self.info_NCE_single(q_feat, pos_feats, neg_feats, max_num, tao)
            elif self.con_loss_type == 'triplet':
                con_loss_single = self.triplet_single(q_feat, pos_feats, neg_feats)

            return con_loss_single

    def calc_con_loss(self, pos_roi_feats, pos_roi_labels):
        avg_pool = nn.AdaptiveAvgPool2d(1)
        con_loss = 0

        for i in range(len(pos_roi_feats)):  ## select_idx
            pos_roi_feats[i] = avg_pool(pos_roi_feats[i]).flatten(1)  # (256, 1)
            pos_roi_feats[i] = torch.true_divide(pos_roi_feats[i],
                                                 torch.norm(pos_roi_feats[i], dim=0).reshape(-1, 1))  # normalization


        ### for triplet loss
        pos_roi_feats = torch.cat(pos_roi_feats, dim=1).permute(1,0)
        # print(pos_roi_feats.size())
        pos_roi_labels = torch.tensor(pos_roi_labels)
        # print(pos_roi_labels.size())

        triplet_loss = TripletLoss()
        con_loss = triplet_loss(pos_roi_feats, pos_roi_labels)
        print(f'The triplet loss of this batch is: {con_loss}')
        ###

        ### for infoNCE loss
        # valid_cnt = 0
        # all_cnt = 0
        # for j in range(len(pos_roi_feats)):
        #     con_loss_single = self._calc_conloss_single(j, pos_roi_feats, pos_roi_labels)
        #     all_cnt += 1
        #     if con_loss_single is not 0:
        #         valid_cnt += 1
        #         con_loss += con_loss_single
        #
        # # normalization of con_loss among all the valid roi features
        # con_loss = con_loss if valid_cnt == 0 else con_loss / valid_cnt
        # # print(f'the average con_loss of this batch is:{con_loss}')
        ###

        return con_loss

    def info_NCE_single(self, q_feat, pos_feats, neg_feats, max_num, tao, select='rule'):
        pos_logits = 0
        neg_logits = 0
        flag = 1
        if select == 'random':
            k = random.randint(0, len(pos_feats) - 1)
            pos_logits = torch.exp(torch.true_divide(torch.mm(q_feat.view(1, -1), pos_feats[k].view(-1, 1)), tao))

            for j in range(len(neg_feats)):
                if j > max_num:
                    break
                else:
                    cur_logit = torch.exp(
                        torch.true_divide(torch.mm(q_feat.view(1, -1), neg_feats[j].view(-1, 1)), tao))
                    neg_logits += cur_logit

            # print(len(pos_feats),len(neg_feats))
            # print(f'the pos logit is:{pos_logits}')
            # print(f'the neg logit is:{neg_logits}')
            # print(f'the con_loss_single is:{con_loss_single}')
        elif select == 'rule':
            if len(pos_feats) < 1 or len(neg_feats) < 3:
                flag = 0
            else:
                aspect = len(neg_feats) // 3 < len(pos_feats)
                if aspect == True:  # means num_neg/3 < num_pos --> select according to neg samples
                    n_neg = 3 * (len(neg_feats) // 3)
                    n_pos = len(neg_feats) // 3
                    if n_neg > max_num:
                        n_neg = max_num
                        n_pos = max_num // 3
                else:
                    n_neg = 3 * len(pos_feats)
                    n_pos = len(pos_feats)
                    if n_pos > max_num // 3:
                        n_pos = max_num // 3
                        n_neg = max_num
                # get randomly select pos and neg samples with specific number(1:3)
                i_neg = random.sample(range(0, len(neg_feats)), n_neg)
                i_pos = random.sample(range(0, len(pos_feats)), n_pos)
                for i in i_neg:
                    cur_logit = torch.exp(
                        torch.true_divide(torch.mm(q_feat.view(1, -1), neg_feats[i].view(-1, 1)), tao))
                    neg_logits += cur_logit
                for j in i_pos:
                    cur_logit = torch.exp(
                        torch.true_divide(torch.mm(q_feat.view(1, -1), pos_feats[j].view(-1, 1)), tao))
                    pos_logits += cur_logit
        if flag == 1:
            denominator = pos_logits + neg_logits
            con_loss_single = -torch.log(torch.true_divide(pos_logits, denominator))[0][0]
        else:
            con_loss_single = 0
        return con_loss_single

    def triplet_single(self, q_feat, pos_feats, neg_feats):
        # to use nn.TripletMarginLoss() in Pytorch, expand 'q_feat', 'pos_feats', 'neg_feats' as the same shape of (N, D)
        max_len = max(len(pos_feats), len(neg_feats))
        q_feat = torch.repeat_interleave(q_feat, max_len, 0)
        logit = int(len(pos_feats) - len(neg_feats))
        # print(logit)
        if logit == 0:
            pass
        elif logit > 0:
            tmp_list = [neg_feats[0] for i in range(abs(logit))]
            neg_feats = neg_feats + tmp_list
        else:
            tmp_list = [pos_feats[0] for i in range(abs(logit))]
            pos_feats = pos_feats + tmp_list
        # print(len(pos_feats))
        assert len(pos_feats) == len(neg_feats)

        pos_feats = torch.cat(pos_feats, 0)
        neg_feats = torch.cat(neg_feats, 0)
        # print(q_feat.shape)
        assert q_feat.shape == pos_feats.shape == neg_feats.shape
        triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)  # MOVE TO INIT
        output = triplet_loss(q_feat, pos_feats, neg_feats)
        # print(f'the output of the triplet loss is:{output}')

        return output

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            # print(cls_score)
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            if i == self.num_stages - 1:
                cls_score = cls_score.split(num_proposals_per_img, 0)
            else:
                pass
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        # bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        bbox_label = None
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        # print(f'The original ms_score is:{ms_scores}')      ######
        ms_scores.remove(None)
        # print(f'The filtered ms_score is:{ms_scores}')
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in mask_pred
                    ])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = np.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=dummy_scale_factor,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

    def onnx_export(self, x, proposals, img_metas):

        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals.shape[0] == 1, 'Only support one input image ' \
                                        'while in exporting to ONNX'
        # remove the scores
        rois = proposals[..., :-1]
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        # Eliminate the batch dimension
        rois = rois.view(-1, 4)

        # add dummy batch index
        rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

        max_shape = img_metas[0]['img_shape_for_onnx']
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            # Recover the batch dimension
            rois = rois.reshape(batch_size, num_proposals_per_img,
                                rois.size(-1))
            cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                          cls_score.size(-1))
            bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois = self.bbox_head[i].bbox_coder.decode(
                    rois[..., 1:], bbox_pred, max_shape=max_shape)
                rois = new_rois.reshape(-1, new_rois.shape[-1])
                # add dummy batch index
                rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois],
                                 dim=-1)

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        det_bboxes, det_labels = self.bbox_head[-1].onnx_export(
            rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            batch_index = torch.arange(
                det_bboxes.size(0),
                device=det_bboxes.device).float().view(-1, 1, 1).expand(
                det_bboxes.size(0), det_bboxes.size(1), 1)
            rois = det_bboxes[..., :4]
            mask_rois = torch.cat([batch_index, rois], dim=-1)
            mask_rois = mask_rois.view(-1, 5)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                aug_masks.append(mask_pred)
            max_shape = img_metas[0]['img_shape_for_onnx']
            # calculate the mean of masks from several stage
            mask_pred = sum(aug_masks) / len(aug_masks)
            segm_results = self.mask_head[-1].onnx_export(
                mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
                self.test_cfg, max_shape)
            segm_results = segm_results.reshape(batch_size,
                                                det_bboxes.shape[1],
                                                max_shape[0], max_shape[1])
            return det_bboxes, det_labels, segm_results