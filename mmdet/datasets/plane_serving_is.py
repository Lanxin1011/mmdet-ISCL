import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

###### add
import os
from tqdm import trange
from operator import itemgetter

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset


import pycocotools


@DATASETS.register_module()
class PlaneDataset(CustomDataset):

    CLASSES = ('Boeing737','Boeing747','Boeing777','Boeing787',
               'A220','A321','A330','A350','ARJ21','other')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file. #

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES) 
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()

        data_infos = []
        ###### fname_id_dict stores the filename-id pairs
        self.fname_id_dict = {}
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # print(f'info: {info}')
            info['filename'] = info['file_name']
            data_infos.append(info)
            self.fname_id_dict[info['id']] = info['file_name']
        # print(f'self.fname_id_dict: {self.fname_id_dict}')
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        # print(f'length of self : {len(self)}')
        for idx in range(len(self)):  # len(self)=197 the number of testing images
            img_id = self.img_ids[idx]
            result = results[idx]     # len(result)=10 the number of categories
            # print(f'length of result: {len(result)}')
            for label in range(len(result)):
                bboxes = result[label]
                # print(label)
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])

                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _det2json_xyxy(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        json_results_by_img = []
        # print(f'length of self : {len(self)}')
        for idx in range(len(self)):  # len(self)=803 the number of testing images
            json_r = []
            img_id = self.img_ids[idx]
            result = results[idx]     # len(result)=10 the number of categories
            # print(f'length of result: {len(result)}')
            for label in range(len(result)):
                json_rr = []    
                bboxes = result[label]
                # print(label)
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = bboxes[i][:4]
                    data['score'] = float(bboxes[i][4])

                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
                    json_rr.append(data)
                json_r.append(json_rr)
            json_results_by_img.append(json_r)
        print(f'length of json_results_by_img: {len(json_results_by_img)}')
        # print(json_results)
        print(f'length of json_results:{len(json_results)}')
        return json_results_by_img
 
    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir


    def all_NMS(self, all_objects):
        iou_th = 0.4
        tmp_objects = []
        re_idx = []
        num = len(all_objects)
        for idx, obj in enumerate(all_objects):
            if idx in re_idx:
                continue
            bbox1 = obj['bbox']
            score1 = obj['score']

            for idx_c in range(idx + 1, num):
                if idx_c in re_idx:
                    continue
                obj_c = all_objects[idx_c]
                bbox2 = obj_c['bbox']
                score2 = obj_c['score']
                iou, inter, area1, area2 = self.box_iou(bbox1, bbox2)
                if iou > iou_th:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
                elif inter == area1 or inter == area2:
                    id_m = idx if score1 < score2 else idx_c
                    re_idx.append(id_m)
            if idx not in re_idx:
                tmp_objects.append(obj)

        return tmp_objects


    def box_iou(self, bbox1, bbox2):
        '''

        :param bbox1: bounding box of GT(or det_obj) (a list of xmin,ymin,xmax,ymax)
        :param bbox2: bounding box of det_obj(or GT) (a list of xmin,ymin,xmax,ymax)
        :return:iou, inter,area1,area2
        '''
        area1 = self.box_area(bbox1)
        area2 = self.box_area(bbox2)

        lt_x, lt_y = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        rb_x, rb_y = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        if rb_x - lt_x > 0 and rb_y - lt_y > 0:
            inter = (lt_x - rb_x) * (lt_y - rb_y)
        else:
            inter = 0
        iou = inter / (area1 + area2 - inter)
        return iou, inter, area1, area2


    def box_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


    def fp_analysis(self, gts_path, json_results, scores_th=0):
        gts_files = os.listdir(gts_path)
        num_gts = len(gts_files)
        num_imgs = len(json_results)
        assert num_gts == num_imgs, ('The length of images and anns should be equal!')
        print('Successfully call the function!')

        # cls_map = {"Boeing737": 1,
        #         "Boeing747": 2,
        #         "Boeing777": 3,
        #         "Boeing787": 4,
        #         "A220": 5,
        #         "A321": 6,
        #         "A330": 7,
        #         "A350": 8,
        #         "ARJ21": 9,
        #         "other": 10}
        # num_class = len(cls_map)
        num_class = len(json_results[0])
        cls_list = self.CLASSES
        # print(f'cls_list:{cls_list}')

        obj_count = 1
        all_labels = []
        cm = np.zeros((num_class, num_class))
        print('Start calculating...')
        for i in trange(num_imgs):
            
            all_objects, final_objects, bboxes_curr, pbboxes, count = [], [], [], [], 0
            all_scores = [] 
            # [img  [label  {data}  ]  ][]

            img_id = 0

            det_result = self.json_results_by_img[i]
            for idx_cls in range(num_class):
                bbox_cls = det_result[idx_cls]
                for ids, each_obj in enumerate(bbox_cls):
                    object_struct = {}
                    score = each_obj['score']
                    bbox = each_obj['bbox']
                    img_id = each_obj['image_id']
                    if score > scores_th:
                        object_struct['bbox'] = bbox
                        object_struct['label'] = idx_cls
                        object_struct['score'] = score
                        all_objects.append(object_struct)
                        all_labels.append(object_struct['label'])
                    else:
                        pass
            if img_id != 0:
                fname = self.fname_id_dict.get(img_id)
                # print(fname)
            else:
                continue

            final_objects = self.all_NMS(all_objects)
            label_file = os.path.join(gts_path, fname.split(".")[0] + '.txt')
            # print(label_file)
            with open(label_file, mode='r') as f:
                gt_raw = f.readlines()
            gt_raw = [x.split(" ") for x in gt_raw]
            gt = []
            for g in gt_raw:
                gt.append([round(float(g[0])), round(float(g[1])), round(float(g[2])), round(float(g[3])), round(float(g[4].split("\n")[0]))])
            
            for obj in final_objects:
                det_obj = obj['bbox']
                for g in gt:
                    if self.box_iou(det_obj, g[:4])[0] > 0.5:
                        # print("gt:", cls_list[g[-1]], "->", "pred:", cls_list[obj['label']], "score:", str(obj["score"]))
                        cm[int(g[-1]), int(obj['label'])] += 1
                        break
        from collections import Counter

        print(f'the counting of all the bboxes: \n {Counter(all_labels)}')

        return cm


    def calc_cm(self, gts_path, results):
        self.json_results_by_img = self._det2json_xyxy(results)
        confusion_matrix = self.fp_analysis(gts_path, self.json_results_by_img)

        return confusion_matrix
        

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        
        ###### calculate confusion matrix
        # gts_path = '/mapai/haowenguo/code/SPL/mmdetection/plane_cm/trainval_labels_gf2/'
        # cm_save_path = '/mapai/haowenguo/code/SPL/mmdetection/cm_tmp.txt'
        # cm_save_path = '/jizhi/jizhi2/worker/trainer/cm_tmp.txt'

        gts_path = './plane_cm/trainval_labels_gf2/'
        cm_save_path = './cm_tmp.txt'

        map_thr = 0.0
        num_val = len(os.listdir(gts_path))
        if num_val == len(results):
            print('Calculating confusion matrix...')
            cm = self.calc_cm(gts_path, results)
            print('The confusion matrix is: \n')
            print(cm)
            flag = 1
        else:
            flag = 0

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
                # print(type(cocoDt))  # <class 'pycocotools.coco.COCO'>
                # print(cocoDt)  # <pycocotools.coco.COCO object at 0x7f640ac04828>
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    # print(cocoEval.eval.keys())
                    # assert False
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    results_per_category_50 = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]

                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                        ### add for ap_50 of each class
                        precision_50 = precisions[0, :, idx, 0, -1]
                        precision_50 = precision_50[precision_50 > -1]
                        if precision_50.size:
                            ap_50 = np.mean(precision_50)
                        else:
                            ap_50 = float('nan')
                        results_per_category_50.append(
                            (f'{nm["name"]}', f'{float(ap_50):0.3f}'))
                        # print(results_per_category_50)
                        ###
                    # print(results_per_category)  ### get AP of each class, e.g. [('Boeing737', '0.016'), ('Boeing747', '0.018')]

                    ### add for ap_50 of each class
                    num_columns = min(9, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    results_flatten_50 = list(itertools.chain(*results_per_category_50))

                    new_results_flatten = []
                    for idx, catId in enumerate(self.cat_ids):
                        new_results_flatten.append(results_flatten[idx * 2])
                        new_results_flatten.append(results_flatten[idx * 2 + 1])
                        new_results_flatten.append(results_flatten_50[idx * 2 + 1])

                    headers = ['category', 'AP', 'AP_50'] * (num_columns // 3)
                    results_2d = itertools.zip_longest(*[
                        new_results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)
                    ###


                    # num_columns = min(6, len(results_per_category) * 2)
                    # results_flatten = list(
                    #     itertools.chain(*results_per_category))
                    # headers = ['category', 'AP'] * (num_columns // 2)
                    # results_2d = itertools.zip_longest(*[
                    #     results_flatten[i::num_columns]
                    #     for i in range(num_columns)
                    # ])
                    # table_data = [headers]
                    # table_data += [result for result in results_2d]
                    # table = AsciiTable(table_data)
                    # print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
                
                ###### decide whether modify switch_cls.txt or not
                # print('Need to modify the txt file!!!')
                cm_len = 9 # whether to consider the last class 'others', if yes, cm_len = 10, else, cm_len = 9
                if flag and (ap[1] >= map_thr):
                    os.remove(cm_save_path)
                    with open(cm_save_path, 'w') as f:
                        for i in range(cm_len):
                            cm_line = ' '.join(str(cm[i,j]) for j in range(cm_len)) + '\n'
                            f.write(cm_line)
                        # f.write("1")
                    # print(type(cm))  # <class 'numpy.ndarray'>

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
