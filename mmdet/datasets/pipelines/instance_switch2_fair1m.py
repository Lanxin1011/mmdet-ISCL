from mmdet.datasets import PIPELINES
import json
import os
import cv2
import tqdm
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors

'''
Randomly select instance to replace with a certain probability.
'''

@PIPELINES.register_module()
class InstanceSwitch2FAIR:

    def get_bg_hsv_avg(self, obj_crop):
        obj_hsv = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2HSV)[:, [0, -1], :]
        obj_hsv = obj_hsv.reshape((-1, 3))
        bg_hsv = np.average(obj_hsv, axis=0)
        return bg_hsv

    def __init__(self, data_root="data/GF/airplane/", switch_prob=0.5,ann_path = "data/GF/airplane/ann_new/instances_trainval2021.json"):
        self.switch_prob = switch_prob
        crop_instance_path = os.path.join(data_root,"crop")
        crop_list = os.listdir(crop_instance_path)

        ann_file = open(ann_path)
        ann_json = json.load(ann_file)
        annotations = ann_json["annotations"]
        num_bridges = len(annotations)
        images = ann_json["images"]

        img_to_instance = {}
        img_to_instance_wo_name = []

        # self.switch_vis_path = '/mapai/haowenguo/code/SPL/mmdetection/switch_vis/'

        # try:
        #     os.mkdir(self.switch_vis_path)
        # except:
        #     pass


        print("Registering cropped instances...")
        self.crop_bridge = []
        crop_args = []
        for i in tqdm.trange(len(crop_list)):
            crop_filename = crop_list[i]
            o = int(crop_filename[0])
            if o == 0:
                cat_id = int(crop_filename[1])
            else:
                cat_id = int(crop_filename[:2])
            
            # cat_id = int(crop_filename[0]) 
            # if cat_id not in self.switch_class:
            #     continue
            obj_crop = cv2.imread(os.path.join(crop_instance_path, crop_filename))
            if obj_crop.shape[0] * obj_crop.shape[1] < 600:  # <1500 
                continue
            # if obj_crop.shape[0] < obj_crop.shape[1]:
            #     obj_crop = np.rot90(obj_crop)
            bg_hsv = self.get_bg_hsv_avg(obj_crop) / 30
            # km = KMeans(n_clusters=2, random_state=9)
            # _ = km.fit_predict(obj_hsv)
            # cluster_center = km.cluster_centers_
            self.crop_bridge.append([cat_id,obj_crop])            ##### to store the cat_id of cropped instances
            crop_args.append([*bg_hsv, np.log10(obj_crop.shape[0]*obj_crop.shape[1])*2])
        # KNN fitting
        print("Building a tree...")
        self.crop_bridge = np.asarray(self.crop_bridge,dtype=object)
        crop_args = np.asarray(crop_args)
        self.crop_nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(crop_args)
        print('Complete registration!')

    def get_distence(self, x, y):
        return (abs(x[0] - y[0]) ** 2 + abs(x[1] - y[1]) ** 2) ** 0.5

    def __call__(self, results):
        # if random.random() > self.switch_prob:
        #     return results
        img_cv = results["img"]
        filename = results["ori_filename"]
        ann_dict = results["ann_info"]
        bridges = ann_dict["masks"]
        ori_cat_ids = ann_dict["labels"]
        # print(ori_cat_ids)
        # print(len(bridges))
        # print(img_cv.shape)
        # assert False
        # print(results)
        # assert False
        # print(ann_dict)
        # assert False
        

        num_switch = 0

        for ii,bridge_ in enumerate(bridges):
            ori_cat_id = ori_cat_ids[ii]
            if ori_cat_id > 9:
                continue
            kk = random.random()
            if kk > self.switch_prob:
                # print(kk)
                continue
            # print(len(bridges), filename)
            # if ori_cat_id in self.switch_class:
            #     continue
            # if self.switch_num_of_more[self.more_class.index(ori_cat_id)] == 0:
            #     continue

            bridge = bridge_[0]
            canvas = np.zeros_like(img_cv[:, :, 0])
            obb = np.array(bridge, np.int32).reshape((4, 1, 2))
            rect = cv2.minAreaRect(obb)
            ((cx, cy), (w, h), theta) = rect
            scale_factor = (max(w, h) / min(w, h)) ** 0.5
            if w > h:
                h *= scale_factor
            else:
                w *= scale_factor
            rect = ((cx, cy), (w, h), theta)
            box = cv2.boxPoints(rect)
            box = np.asarray(box).astype(np.int32)
            canvas = cv2.fillPoly(canvas, [box], 255)

            dst_pts = box.astype("float32")

            # Crop and warp the orig bridge into shape 20x20. Get the bg hsv.
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array([[0 + 1, 20 - 1 - 1],
                                    [0 + 1, 0 + 1],
                                    [20 - 1 - 1, 0 + 1],
                                    [20 - 1 - 1, 20 - 1 - 1]], dtype="float32")
            else:
                src_pts = np.array([[0 + 1, 0 + 1],
                                    [20 - 1 - 1, 0 + 1],
                                    [20 - 1 - 1, 20 - 1 - 1],
                                    [0 + 1, 20 - 1 - 1]], dtype="float32")
            M0 = cv2.getPerspectiveTransform(dst_pts, src_pts)
            origin_bridge_0 = cv2.warpPerspective(img_cv, M0, (20, 20))
            orig_bg_hsv = self.get_bg_hsv_avg(origin_bridge_0) / 30

            # Use KNN to get 50 similar bridges and get a random choice.
            # find = 0
            _, indices = self.crop_nbrs.kneighbors(np.array([[*orig_bg_hsv, np.log10(w * h)*2]])) ##### * 3 original
            for i in range(len(indices[0])):
                cur_id = self.crop_bridge[indices[0][i]][0]
                if cur_id == ori_cat_id:
                    continue
                else:
                    cat_id = self.crop_bridge[indices[0][i]][0]
                    switch_target = self.crop_bridge[indices[0][i]][1]  ##### record cat_id and croped plane data of the nearest available plane
                    break
                if i == len(indices[0])-1:
                    print(f'fail!!!!!!!!!!!!!!!!!!!!!')

            target_id = cat_id
            # for i in range(len(indices[0])):
            #     cat_id = self.crop_bridge[indices[0][i]][0]
            #     if self.switch_num_of_less[self.switch_class.index(cat_id)] == 0:
            #         continue
            #     else:
            #         switch_target = self.crop_bridge[indices[0][i]][1]  ##### record cat_id and croped plane data of the nearest available plane
            #         target_id = cat_id
            #         self.switch_num_of_more[self.more_class.index(ori_cat_id)] -= 1
            #         # print(self.switch_num_of_more)
            #         self.switch_num_of_less[self.switch_class.index(cat_id)] -= 1
            #         # print(self.switch_num_of_less)
            #         find = 1
            #         break
            #
            # if find == 0:
            #     continue
            # print(list(self.crop_bridge)[0][1].shape)
            # print(self.crop_bridge.shape)
            # print(self.crop_bridge[0][0])
            # print(list(self.crop_bridge)[0][0])
            # print(self.crop_bridge[0][0] == list(self.crop_bridge)[0][0])

            ### randomly select a instance to switch rather than select according to KNN 
            # num_bridges = len(self.crop_bridge)
            # k = random.randint(0,num_bridges-1)
            # target_id = self.crop_bridge[k][0]
            # switch_target = self.crop_bridge[k][1]


            switch_h = switch_target.shape[0]
            switch_w = switch_target.shape[1]

            # To align the long side of instance crop.
            if self.get_distence(dst_pts[0], dst_pts[1]) > self.get_distence(dst_pts[1], dst_pts[2]):
                # Contract 1 pixel to avoid interpolation error.
                src_pts = np.array([[0 + 1, switch_h - 1 - 1],
                                    [0 + 1, 0 + 1],
                                    [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1]], dtype="float32")
            else:
                src_pts = np.array([[0 + 1, 0 + 1],
                                    [switch_w - 1 - 1, 0 + 1],
                                    [switch_w - 1 - 1, switch_h - 1 - 1],
                                    [0 + 1, switch_h - 1 - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            M_inv = np.linalg.inv(M)

            # Get cropped original bridge.
            origin_bridge = cv2.warpPerspective(img_cv, M_inv, (switch_w, switch_h))
            # print(origin_bridge.shape)
            # print(origin_bridge[:,:,0])
            # Generate gaussian kernel.
            gaussian_kernel = cv2.getGaussianKernel(switch_w, switch_w / 7).reshape((switch_w,))
            gaussian_kernel /= gaussian_kernel.max()
            mask_weight = gaussian_kernel * np.ones((switch_h, switch_w))
            mask_weight_h = mask_weight.T
            mask_weight_h = mask_weight_h[:, :, None]
            mask_weight_h = np.concatenate([mask_weight_h, mask_weight_h, mask_weight_h], axis=-1)
            mask_weight = mask_weight[:, :, None]
            mask_weight = np.concatenate([mask_weight, mask_weight, mask_weight], axis=-1)

            mask_weight_h = cv2.resize(mask_weight_h, (switch_w, switch_h))
            mask_weight = mask_weight + mask_weight_h
            mask_weight[mask_weight > 0.8] = 1

            mask_weight_inv = 1 - mask_weight
            # Fuse instance.
            switch_target = switch_target * mask_weight + origin_bridge * mask_weight_inv
            switch_target.astype(np.uint8)
            # *****************************************************8*
            # if num_switch < 10:
            #     num_switch += 1
            #     cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_tar.png', switch_target)

            # Paste instance onto image.
            switch_warped = cv2.warpPerspective(switch_target, M, (img_cv.shape[1], img_cv.shape[0]))
            # img_norm = cv2.normalize(img_cv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #
            # img_norm.astype(np.uint8)
            # img_cv = img_norm
            # if num_switch < 10:
            #     cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_ori.png', img_cv)
            img_cv[canvas == 255] = switch_warped[canvas == 255]
            # num_switch += 1
            # if num_switch <= 10:
            #     cv2.imwrite(self.switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_switched.png', img_cv)
            # print(f'{ori_cat_id} --> {target_id}')
            ori_cat_ids[ii] = target_id
            # print(ori_cat_ids[ii])



        return results