import aitool

def drop_invalid_pointobb_w_cat_id(pointobbs_w_cat_id, min_area=2, min_side=2):
    """drop the invalid bboxes

    Args:
        bboxes (list): list of bboxes, (coco format, [xmin, ymin, xmax, ymax])
        min_area (int, optional): min area of bboxes. Defaults to 2.
        min_side (int, optional): min side of bboxes. Defaults to 2.
    
    Return:
        list: list of keep bboxes (coco format, [xmin, ymin, xmax, ymax])
    """
    results = []
    for obb in pointobbs_w_cat_id:
        xyxy = aitool.pointobb2bbox(obb[0])
        cx, cy, w, h = aitool.xyxy2cxcywh(xyxy)

        if w * h < min_area:
            continue

        if w < min_side or w < min_side:
            continue
        
        result_w_cat_id = []
        result_w_cat_id.append([int(_) for _ in obb[0]])
        result_w_cat_id.append(obb[1])
        results.append(result_w_cat_id)
        # results.append([int(_) for _ in bbox])
    
    return results

def drop_invalid_bboxes_w_cat_id(bboxes_w_cat_id, min_area=2, min_side=2):
    """drop the invalid bboxes

    Args:
        bboxes (list): list of bboxes, (coco format, [xmin, ymin, xmax, ymax])
        min_area (int, optional): min area of bboxes. Defaults to 2.
        min_side (int, optional): min side of bboxes. Defaults to 2.
    
    Return:
        list: list of keep bboxes (coco format, [xmin, ymin, xmax, ymax])
    """
    results = []
    for bbox in bboxes_w_cat_id:
        cx, cy, w, h = aitool.xyxy2cxcywh(bbox[0])

        if w * h < min_area:
            continue

        if w < min_side or w < min_side:
            continue
        
        result_w_cat_id = []
        result_w_cat_id.append([int(_) for _ in bbox[0]])
        result_w_cat_id.append(bbox[1])
        results.append(result_w_cat_id)
        # results.append([int(_) for _ in bbox])
    
    return results

def drop_invalid_pointobb(pointobbs, min_area=2, min_side=2):
    """drop the invalid bboxes

    Args:
        bboxes (list): list of bboxes, (coco format, [xmin, ymin, xmax, ymax])
        min_area (int, optional): min area of bboxes. Defaults to 2.
        min_side (int, optional): min side of bboxes. Defaults to 2.
    
    Return:
        list: list of keep bboxes (coco format, [xmin, ymin, xmax, ymax])
    """
    results = []
    for obb in pointobbs:
        xyxy = aitool.pointobb2bbox(obb)
        cx, cy, w, h = aitool.xyxy2cxcywh(xyxy)

        if w * h < min_area:
            continue

        if w < min_side or w < min_side:
            continue
        
        results.append([int(_) for _ in obb])
    
    return results

def drop_invalid_bboxes(bboxes, min_area=2, min_side=2):
    """drop the invalid bboxes

    Args:
        bboxes (list): list of bboxes, (coco format, [xmin, ymin, xmax, ymax])
        min_area (int, optional): min area of bboxes. Defaults to 2.
        min_side (int, optional): min side of bboxes. Defaults to 2.
    
    Return:
        list: list of keep bboxes (coco format, [xmin, ymin, xmax, ymax])
    """
    results = []
    for bbox in bboxes:
        cx, cy, w, h = aitool.xyxy2cxcywh(bbox)

        if w * h < min_area:
            continue

        if w < min_side or w < min_side:
            continue
        
        results.append([int(_) for _ in bbox])
    
    return results
    
    