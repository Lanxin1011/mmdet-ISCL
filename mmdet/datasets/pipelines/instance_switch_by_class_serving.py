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
Randomly select instance to replace according to selection and serving probabilities.
'''

@PIPELINES.register_module()
class InstanceSwitchByClassServing:

    def get_bg_hsv_avg(self, obj_crop):
        obj_hsv = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2HSV)[:, [0, -1], :]
        obj_hsv = obj_hsv.reshape((-1, 3))
        bg_hsv = np.average(obj_hsv, axis=0)
        return bg_hsv

    def __init__(self, 
                data_root="data/GF/airplane/", 
                switch_prob=0.5,
                ann_path = "data/GF/airplane/ann_new/instances_trainval2021.json"):

        ### save data to draw class distribution after switching
        # self.log_file_path = '/mapai/haowenguo/code/SPL/mmdetection/switched_dis.txt'

        ###### initilization of cm properties.
        # self.cm_save_path = '/mapai/haowenguo/code/SPL/mmdetection/cm_tmp.txt'
        # self.cm_save_path = '/jizhi/jizhi2/worker/trainer/cm_tmp.txt'
        self.cm_save_path = './cm_tmp.txt'

        self.cm_len = 9 # if don't consider class 'others', also need to uncomment codes in line 196~197
        self.cm = [[1]*self.cm_len]*self.cm_len
        # Insert an all-1 confusion matrix and an equally probabilistic step normalized recall rate and step service probability
        with open(self.cm_save_path, 'w') as f:
            for i in range(self.cm_len):
                cm_line = ' '.join(str(self.cm[i][j]) for j in range(self.cm_len)) + '\n'
                f.write(cm_line)
        print(f'The initial cm is:\n {self.cm}') 
        self.stair_prob_matrix = []
        for i in range(self.cm_len):
            tmp_prob = []
            cur = 0
            for j in range(self.cm_len):
                if j == i:
                    tmp_prob.append(cur)
                else:
                    cur += 1/(self.cm_len - 1)
                    tmp_prob.append(cur)
            self.stair_prob_matrix.append(tmp_prob)
        print(f'The initial stair probability matrix is:\n{self.stair_prob_matrix}')

        self.stair_selection_prob = [(1/self.cm_len) * (i+1) for i in range(self.cm_len)]
        self.stair_serving_prob = [(1/self.cm_len) * (i+1) for i in range(self.cm_len)]


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

        print("Registering cropped instances...")
        self.crop_bridge = {}
        for i in range(10):
            self.crop_bridge[str(i)] = []
        # crop_args = []
        for i in tqdm.trange(len(crop_list)):
            crop_filename = crop_list[i]
            cat_id = int(crop_filename[0])
            # if cat_id not in self.switch_class:
            #     continue
            obj_crop = cv2.imread(os.path.join(crop_instance_path, crop_filename))
            if obj_crop.shape[0] * obj_crop.shape[1] < 1500:
                continue
           
            self.crop_bridge[str(cat_id)].append([cat_id,obj_crop])            ##### to store the cat_id of cropped instances
            # crop_args.append([*bg_hsv, np.log10(obj_crop.shape[0]*obj_crop.shape[1])*3])
        # KNN fitting
        # print("Building a tree...")
        # self.crop_bridge = np.asarray(self.crop_bridge,dtype=object)
        # crop_args = np.asarray(crop_args)
        # self.crop_nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(crop_args)
        self.initial_cat_distribution = [len(self.crop_bridge[str(i)]) for i in range(self.cm_len)]
        self.cur_cat_distribution = self.initial_cat_distribution
        print(f'The initial distribution of categories is:\n{self.initial_cat_distribution}')
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

        ###### loading confusion matrix --> cm:[[],[],...,[]]
        with open(self.cm_save_path, mode='r') as f:
            cm_tmp = f.readlines()
        cm_raw = [line.strip("\n").split(" ") for line in cm_tmp]
        cm = []
        for i in range(self.cm_len):
            cm_line = [float(v) for v in cm_raw[i]]
            cm.append(cm_line)
        if self.cm != cm:

            # print('re-computing is prob_matrix...')
            ###### compute is probability matrix
            # print(f'The original confusion matrix is:\n{self.cm}')
            self.prob_matrix = []
            for i in range(self.cm_len):
                tp = cm[i][i]
                fp = sum(cm[i]) - tp
                if fp == 0: 
                    prob_line = [0] * self.cm_len
                    if tp != 0:
                        prob_line[i] = tp / (fp + tp)
                else:
                    prob_line = [v/fp for v in cm[i]]  # fp by class
                    prob_line[i] = tp / (fp + tp)  # tp
                self.prob_matrix.append(prob_line)
            print(f'The probability matrix is:\n{self.prob_matrix}')

            ###### Rules for selection: lower recall --> lower is_prob 
            recall_list = [self.prob_matrix[i][i] for i in range(self.cm_len)]
            # print(f'The recall list is:\n{recall_list}')
            if recall_list != [0]*self.cm_len:
                norm_recall_list = [recall_list[i]/sum(recall_list) for i in range(self.cm_len)]
                # print(f'The norm_recall_list is:\n{norm_recall_list}')
                self.stair_selection_prob = [sum(norm_recall_list[:i+1]) for i in range(self.cm_len)]
                for i in range(self.cm_len):
                    self.stair_selection_prob[i] = self.stair_selection_prob[i] 
                # print(f'The stair selection probability is:\n {self.stair_selection_prob}')

                ###### compute the stair probability matrix
                self.stair_prob_matrix = []
                for i in range(self.cm_len):
                    prob_vector = self.prob_matrix[i]
                    prob_vector[i] = 0
                    ### some classes might have a prob_vector of [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ### then we just simply make it [1, 1,..., 0, 1,..., 1]
                    if prob_vector == [0]*self.cm_len:
                        prob_vector = [1/(self.cm_len-1)]*self.cm_len
                        prob_vector[i] = 0
                    # else:
                    #     prob_vector[i] = 0
                    self.stair_prob_matrix.append([sum(prob_vector[:j+1]) for j in range(self.cm_len)])  
                print(f'The stair probability matrix is:\n{self.stair_prob_matrix}')

                ###### Rules for serving: lower recall --> higher serving_prob
                # +0.00001 to prevent devide 0
                serving_prob_tmp = [1/(norm_recall_list[i]+0.00001) for i in range(self.cm_len)]  
                serving_prob = [serving_prob_tmp[i]/sum(serving_prob_tmp) for i in range(self.cm_len)]
                self.stair_serving_prob = [sum(serving_prob[:i+1]) for i in range(self.cm_len)]
                # print(stair_serving_prob[self.cm_len - 1])
                # print(f'The serving probability is:\n {self.stair_serving_prob}')
                self.cm = cm
                # print(f'The new confusion matrix is:\n {self.cm}')
            else:  ###### Restore the initial setting when the recall of all categories is 0
                self.cm = cm
                self.stair_prob_matrix = []
                for i in range(self.cm_len):
                    tmp_prob = []
                    cur = 0
                    for j in range(self.cm_len):
                        if j == i:
                            tmp_prob.append(cur)
                        else:
                            cur += 1/(self.cm_len - 1)
                            tmp_prob.append(cur)
                    self.stair_prob_matrix.append(tmp_prob)
                self.stair_selection_prob = [(1/self.cm_len) * (i+1) for i in range(self.cm_len)]
                self.stair_serving_prob = [(1/self.cm_len) * (i+1) for i in range(self.cm_len)]

        num_switch = 0
        for ii, bridge_ in enumerate(bridges):
            ###### every instance has the probability of 'P0*self.stair_selection_prob' to be switched
            ori_cat_id = ori_cat_ids[ii]
            ###### don't consider class 'others'
            if int(ori_cat_id) == 9:
                continue
            kk = random.random()
            cur_selection_prob = self.stair_selection_prob[ori_cat_id]
            if kk > cur_selection_prob:
                continue
            kk_p0 = random.random()
            if kk_p0 > self.switch_prob:  # means this instance has not been selected to participate in the switching process
                continue

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

            ###### based on the category of the current sample, choose the serving class according to the 'serving prob'
            kk_serve = random.random()
            for i in range(self.cm_len):
                if self.stair_serving_prob[i] >= kk_serve:
                    cur_serving_cls = i 
                    break
            ###### based on the seving class, choose the category to be switched to according to the 'prob_matrix'
            kk_switch = random.random()
            for i in range(self.cm_len):
                if self.stair_prob_matrix[cur_serving_cls][i] >= kk_switch:
                    cur_switch_cls = i
                    break

            # ### save data for class distribution map
            # with open(self.log_file_path, 'a') as f:
            #     f.write( str(filename).ljust(30) + str(ori_cat_id).ljust(3) + '-->' + str(cur_switch_cls).ljust(3) + '\n' ) 
           
            ###### ramdomly select a sample of the specific class from the sample base
            num_bridges = len(self.crop_bridge[str(cur_switch_cls)])
            # print(cur_switch_cls)
            # print(num_bridges)
            # print(f'The number of cat_{cur_switch_cls} is {num_bridges}')
            k = random.randint(0,num_bridges-1)
            target_id = self.crop_bridge[str(cur_switch_cls)][k][0]
            switch_target = self.crop_bridge[str(cur_switch_cls)][k][1]

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
            # if num_switch <= 10:
            #     cv2.imwrite(switch_vis_path + str(num_switch) + '_' + filename.split('.')[0] + '_switched.png', img_cv)
            #####
            ori_cat_ids[ii] = target_id #####
            
        return results