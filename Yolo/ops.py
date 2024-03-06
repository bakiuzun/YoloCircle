
import torch
import fractions

# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TORCH_1_10



def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size


    nc = nc or (prediction.shape[1] - 3)  # number of classes
    nm = prediction.shape[1] - nc - 3
    mi = 3 + nc  # mask start index
    xc = prediction[:, 3:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)


    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

    #prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    print("prediction shape = ",prediction.shape)
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        print("X = ",x.shape)
        print("XC = ",xc.shape)
        print("Xi = ",xi)
        x = x[xc[xi]]  # confidence
        print("X SHAPE = ",x.shape)
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            print("GIRDIMMM")
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((3, nc, nm), 1)
        print("box shape = ",box.shape)
        print("cls shape = ",cls.shape)
        print("mask shape = ",mask.shape)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            print(cls)
            conf, j = cls.max(1, keepdim=True)
            print(conf)

            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            print("X SHAPE = ",x.shape)
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        print("N = ",n)
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 4:5] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :3] + c, x[:, 3]  # boxes (offset by class), scores

        print("ALOOOOO = ",scores)
        order = torch.argsort(-scores)
        #order = torch.arange(boxes.shape[0])
        indices = torch.arange(boxes.shape[0])
        keep = torch.ones_like(indices, dtype=torch.bool)
        #circleIOU(boxes,boxes)

        #indices = indices[:100]
        #ious = circleIOU(boxes[order[:1]],boxes[order[1:5]])


        for i in indices:
             if keep[i]:
                 bbox = boxes[order[i:i+1]]
                 iou = circleIOU(bbox,boxes[order[i+1:]])
                 iou = torch.tensor(iou).squeeze()
                 overlapped = torch.nonzero(iou > iou_thres)
                 keep[overlapped + i + 1] = 0
        print("KEEP = ",keep)


        """
        for i in indices:
            if keep[i]:
                bbox = boxes[order[i]]
                #iou = box_iou(bbox,boxes[order[i+1:])
                iou = box_iou(bbox,boxes[order[i+1:]]) * keep[i+1:]
                overlapped = torch.nonzero(iou > iou_thres)
                keep[overlapped + i + 1] = 0
        """
        i = order[keep]
        i = i[:max_det]

        output[xi] = x[i][:200]
        #output[xi] = x[:6]

        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output



# YOLO METHOD, Should be modified
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""

    print("Distance shape = ",distance.shape)
    print("anchor points shape = ",anchor_points.shape)
    #distance shape = (1,4,2800)
    #->>>> (1,3,2800)
    #->> x_center,y_center = [:,:2,:] 2
    #->> radius = [:,3,2800]
    xy_center = distance[:,:2,:]
    radius = distance[:,1:2,:]

    #lt, rb = distance.chunk(2, dim)

    #x1y1 = anchor_points - lt
    #x2y2 = anchor_points + rb

    #top_left = xy_center
    bbox_absolute = xy_center + anchor_points

    return torch.cat((bbox_absolute, radius), dim=1)
    """
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
    """


# YOLO METHOD, Should be modified
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)



def box_iou(boxes_1,bboxes):

    res = torch.zeros((bboxes.shape[0]))

    center_x_1 = boxes_1[0]
    center_y_1 = boxes_1[1]
    radius_1 = boxes_1[2]

    dist_x = (bboxes[:,0] - center_x_1)**2
    dist_y = (bboxes[:,1] - center_y_1)**2
    distance_boxes = torch.sqrt(dist_x + dist_y)

    radi_minus = abs(radius_1 - bboxes[:,2])
    radi_add = abs(radius_1 + bboxes[:,2])


    condition = (torch.abs(boxes_1[2] - boxes_1[2]) <= distance_boxes) & (distance_boxes <= torch.abs(boxes_1[2] + boxes_1[2]))

    matching_indices = torch.nonzero(condition).squeeze(1)
    non_matching_indices = torch.nonzero(~condition).squeeze(1)

    for i in matching_indices:

        r2,r1= bboxes[i][-1],radius_1
        d_squared = distance_boxes[i]**2
        overlap = solve(r1, r2, d_squared)
        union = math.pi * (r1**2) + math.pi * (r2**2) -  overlap
        if union == 0:
            res[i] = 0
        else:
            res[i] = overlap/union
    return res



def circleIOU(d,g):
    ious = np.zeros((len(d), len(g)))
    for di in range(len(d)):
        center_d_x = d[di][0]
        center_d_y = d[di][1]
        center_d_r = d[di][2]
        for gi in range(len(g)):
            center_g_x = g[gi][0]
            center_g_y = g[gi][1]
            center_g_r = g[gi][2]
            distance = math.sqrt((center_d_x - center_g_x)**2 + (center_d_y - center_g_y)**2)
            if center_d_r <=0 or center_g_r <=0 or distance > (center_d_r + center_g_r) :
                ious[di, gi] = 0
            else:
                overlap = solve(center_d_r, center_g_r, distance**2)
                union = math.pi * (center_d_r**2) + math.pi * (center_g_r**2) -  overlap
                if union == 0:
                    ious[di,gi] = 0
                else:
                    ious[di, gi] = overlap/union
    return ious


def f(x):
    """
    Compute  x - sin(x) cos(x)  without loss of significance
    """
    if abs(x) < 0.01:
        return 2 * x ** 3 / 3 - 2 * x ** 5 / 15 + 4 * x ** 7 / 315
    return x - math.sin(x) * math.cos(x)


def acos_sqrt(x, sgn):
    """
    Compute acos(sgn * sqrt(x)) with accuracy even when |x| is close to 1.
    http://www.wolframalpha.com/input/?i=acos%28sqrt%281-y%29%29
    http://www.wolframalpha.com/input/?i=acos%28sqrt%28-1%2By%29%29
    """
    assert isinstance(x, fractions.Fraction)

    y = 1 - x
    if y < 0.01:
        # pp('y < 0.01')
        numers = [1, 1, 3, 5, 35]
        denoms = [1, 6, 40, 112, 1152]
        ans = fractions.Fraction('0')
        for i, (n, d) in enumerate(zip(numers, denoms)):
            ans += y ** i * n / d
        assert isinstance(y, fractions.Fraction)
        ans *= math.sqrt(y)
        if sgn >= 0:
            return ans
        else:
            return math.pi - ans

    return math.acos(sgn * math.sqrt(x))


def solve(r1, r2, d_squared):

    #print("TYPE = ",r1,r2,d_squared)
    r1, r2 = min(r1, r2), max(r1, r2)
    d = math.sqrt(d_squared)
    if d >= r1 + r2:  # circles are far apart
        return 0.0
    if r2 >= d + r1:  # whole circle is contained in the other
        return math.pi * r1 ** 2

    r1f, r2f, dsq = map(fractions.Fraction, [str(r1.detach().numpy()), str(r2.detach().numpy() ), str(d_squared)])
    #r1f, r2f, dsq = map(fractions.Fraction, [str(r1), str(r2), str(d_squared)])
    r1sq, r2sq = map(lambda i: i * i, [r1f, r2f])
    numer1 = r1sq + dsq - r2sq
    cos_theta1_sq = numer1 * numer1 / (4 * r1sq * dsq)
    numer2 = r2sq + dsq - r1sq
    cos_theta2_sq = numer2 * numer2 / (4 * r2sq * dsq)
    theta1 = acos_sqrt(cos_theta1_sq, math.copysign(1, numer1))
    theta2 = acos_sqrt(cos_theta2_sq, math.copysign(1, numer2))
    result = r1 * r1 * f(theta1) + r2 * r2 * f(theta2)

    return result
