import cv2
import os
import shutil
import json
import numpy as np
import tqdm

def crop_instance(cnt,img):
    rect = cv2.minAreaRect(cnt)
    cx = rect[0][0]
    cy = rect[0][1]
    w = rect[1][0]
    h = rect[1][1]
    theta = rect[2]

    scale_factor = (max(w,h)/min(w,h))**0.5
    if w > h:
        rect = ((cx, cy), (w, h * scale_factor), theta)
    else:
        rect = ((cx, cy), (w * scale_factor, h), theta)
    # the order of the box points: bottom left, top left, top right,
    box = cv2.boxPoints(rect)


    box = np.int0(box)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    # print(img,M,(width,height))
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

# img_path = "/data2/zlx/fair1m/fair1m_v2/trainval/"
# out_path = "/data2/zlx/fair1m/fair1m_v2/crop/"
# json_set = ["/data2/zlx/fair1m/fair1m_v2/anns/coco/FAIR1M_trainval2022.json"]
# invalid_path = "/data2/zlx/fair1m/fair1m_v2/invalid/"

img_path = "/data2/zlx/GF/airplane/GF2/trainval_new/"
out_path = "/data2/zlx/GF/airplane/GF2/crop/"
json_set = ["/data2/zlx/GF/airplane/ann_new/instances_trainval2022.json"]
# invalid_path = "/data2/zlx/fair1m/fair1m_v2/invalid/"

try:
    os.mkdir(out_path)
except:
    pass

# try:
#     os.mkdir(invalid_path)
# except:
#     pass

# ann_file = open("./annotations/plane_train_K2.json")
# ann_json = json.load(ann_file)
# annotations = ann_json["annotations"]
# num_bridges = len(annotations)
# images = ann_json["images"]
sum_bridges = 0
cnt = 0
for j in range(len(json_set)):
    ann_file = open(json_set[j])
    ann_json = json.load(ann_file)
    annotations = ann_json["annotations"]
    num_bridges = len(annotations)
    images = ann_json["images"]
    for i in tqdm.trange(num_bridges):
        instance = annotations[i]
        pointobb = instance["pointobb"]
        bbox = instance['bbox']
        # pointobb = instance["segmentation"]
        cat_id = str(instance["category_id"]-1)          ##### To make the first number of the file name the class number, we add '-1' here
                                                             # then the range of the class number is 0~9

        ### for fair1m dataset
        # cat_id = cat_id.zfill(2)
        ###

        print(cat_id)
        # if cat_id > 9:
        #     continue
        image_id = instance["image_id"]
        image_name = images[image_id - 1]["file_name"]

        # if bbox[2]/bbox[3] < 0.8 or bbox[3]/bbox[2] < 0.8 or bbox[2] < 20 or bbox[3] < 20:
        #     cnt += 1
        #     img = cv2.imread(os.path.join(img_path, image_name.split('.')[0] + '.tif'))
        #     pointobb = np.array(pointobb,np.int32).reshape((4,1,2))
        #     bridge = crop_instance(pointobb,img)
        #     cv2.imwrite(os.path.join(invalid_path,cat_id + str(j) + str(image_id).zfill(4) + str(i + sum_bridges) + ".png"), bridge)
        #     continue


        img = cv2.imread(os.path.join(img_path, image_name.split('.')[0] + '.png')) ##### pay attention to the suffix of the file_name
        # print(img)
        # cv2.imshow('img',img)

        pointobb = np.array(pointobb,np.int32).reshape((4,1,2))
        bridge = crop_instance(pointobb,img)

        ### for fair1m dataset
        # cv2.imwrite(os.path.join(out_path,cat_id + str(j) + str(image_id).zfill(4) + str(i + sum_bridges) + ".png"), bridge)  # + str(i + sum_bridges) +
        ###

        cv2.imwrite(os.path.join(out_path, cat_id + str(j) + str(i + sum_bridges) + ".png"), bridge)  # + str(i + sum_bridges) +

    sum_bridges = sum_bridges + num_bridges  # in view of the consistency of the filename
print(f'The total number of instances is {sum_bridges} ')

# print(f'The invalid instance number is {cnt}.')