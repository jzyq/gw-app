import time
from functools import wraps

import os
import base64
import cv2
import math
import numpy as np
from core.basehandler import BaseHandler
from utils.log_config import get_logger

logger = get_logger()

class ImageHandler(BaseHandler):

    class UniqueImageSet:
        def __init__(self):
            self.image_dir = './tmp/imgset'
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)

        def load_set(self, prefix=None):    # prefix in 'YYYYMMDD'
            out_data = []
            for iname in os.listdir(self.image_dir):
                # name format YYYYMMDD-HHMMSS_x1-y1-x2-y2.jpg
                if isinstance(prefix, str) and prefix:
                    if iname.startswith(prefix):
                        x1, y1, x2, y2 = iname.replace('.jpg', '').split('_')[-1].split('-')
                        out_data.append((cv2.imread(os.path.join(self.image_dir, iname)), (int(x1), int(y1), int(x2), int(y2))))
                    else:
                        os.remove(os.path.join(self.image_dir, iname))
                else:
                    x1, y1, x2, y2 = iname.replace('.jpg', '').split('_')[-1].split('-')
                    out_data.append((cv2.imread(os.path.join(self.image_dir, iname)), (int(x1), int(y1), int(x2), int(y2))))
            return out_data

        def add_img(self, image_data=None, prefix=None):
            if image_data is not None and prefix is not None:   # treat prefix as full filename
                return cv2.imwrite(os.path.join(self.image_dir, prefix + '.jpg'), image_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
            return

        def delete_set(self, prefix=None):
            if prefix is not None:  # prefix in 'YYYYMMDD'
                for iname in os.listdir(self.image_dir):
                    if iname.startswith(prefix):
                        os.remove(os.path.join(self.image_dir, iname))
            return
    def __init__(self):
        self.num_masks = 32
        self.num_classes = 1
        self.new_shape = [640, 640]
        self.classes = ['A_baise_beiyong']
        self.unique_set = self.UniqueImageSet()

    # 图片读取，从urlsafe base64 string返回cv2 image
    def base64_to_cv2(self, base64string: str):
        try:
            img0 = np.frombuffer(base64.urlsafe_b64decode(base64string), np.uint8)
            img0 = cv2.imdecode(img0, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f'base64字符串解析错误：{e.__str__()}')

        if img0 is not None:
            return img0
        else:
            raise ValueError(f'base64字符串解析错误：图片格式非法')

    # 图片预处理，返回图片numpy
    def prepare_input(self, image, swapRB=True):
        self.img_height, self.img_width = image.shape[:2]

        # Input shape
        # create new images of desired size and color (blue) for padding

        r = min(self.new_shape[0] / self.img_height, self.new_shape[1] / self.img_width, 1)

        new_unpad = int(round(self.img_height * r)), int(round(self.img_width * r))
        dh, dw = self.new_shape[0] - new_unpad[0], self.new_shape[1] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.resize(image, (new_unpad[1], new_unpad[0]))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        if swapRB:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        input_image = image_rgb / 255.0
        input_image = input_image.transpose(2, 0, 1)[::-1]
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    # nms iou xywh2xyxy
    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original images dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def extract_boxes_segment(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original images dimensions
        boxes = self.rescale_boxes(boxes, )

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the images
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    # 推理后预测框选择&rescale目标检测
    def process_output(self, output, conf, iou):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf, :]
        scores = scores[scores > conf]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object

        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)

        boxes0 = boxes[class_ids == 0, :]
        scores0 = scores[class_ids == 0]
        class_ids0 = class_ids[class_ids == 0]
        indices0 = self.nms(boxes0, scores0, iou)

        box_out = boxes0[indices0]
        class_out = class_ids0[indices0]
        score_out = scores0[indices0]

        for i in range(1, self.num_classes):
            boxes1 = boxes[class_ids == i, :]
            scores1 = scores[class_ids == i]
            class_ids1 = class_ids[class_ids == i]
            indices1 = self.nms(boxes1, scores1, iou)
            box_out = np.append(box_out, boxes1[indices1], axis=0)
            class_out = np.append(class_out, class_ids1[indices1], axis=0)
            score_out = np.append(score_out, scores1[indices1], axis=0)
        return box_out, score_out, class_out

    def process_box_output(self, prediction, confidence, iou_thre, max_nms=30000, max_wh=7680, max_det=300):
        """目标框处理"""
        bs = prediction.shape[0]
        nc = len(self.classes)
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc
        xc = prediction[:, 4:mi].max(1) > confidence

        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):
            x = x.transpose(1, 0)[xc[xi]]

            if not x.shape[0]:
                continue
            box = x[:, :4]
            cls = x[:, 4: 4 + nc]
            mask = x[:, 4 + nc: 4 + nc + nm]
            box = self.xywh2xyxy(box)
            conf = cls.max(1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(dtype=np.float32), mask), 1)[conf.reshape(-1) > confidence]

            n = x.shape[0]
            if not n:
                continue
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

            c = x[:, 5:6] * max_wh
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.nms(boxes, scores, iou_thre)
            i = i[:max_det]

            output[xi] = x[i]
        return output

    def process_mask_output(self, boxes, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []
        mask_output = np.squeeze(mask_output)
        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        n = 1
        mask_output1 = mask_output.reshape((num_mask, -1))
        masks1 = (mask_predictions @ mask_output1[:, 0]).reshape((-1, 1))
        for i in range(1, mask_output1.shape[1]):
            masks1 = np.concatenate((masks1, (mask_predictions @ mask_output1[:, i]).reshape((-1, 1))), 1)
        masks = self.sigmoid(masks1.reshape((num_mask, -1)))

        # n > 1
        # mask_output1 = mask_output.reshape((num_mask, -1))
        # masks1 = (mask_predictions @ mask_output1[:, 0]).reshape((-1, 1))
        # n = 1000
        # for i in range(math.ceil(mask_output1.shape[1] / n)):
        #     masks1 = np.concatenate((masks1, mask_predictions @ mask_output1[:, i * n: (i + 1) * n]), 1)
        # masks = self.sigmoid(masks1[:, 1:])
        # masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size

        r = min(mask_height / self.img_height, mask_width / self.img_width)
        dh, dw = (mask_height - (r * self.img_height)) / 2, (mask_width - (r * self.img_width)) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        boxes0 = boxes * np.array([r, r, r, r])
        scale_boxes = boxes0
        scale_boxes[..., [1, 3]] = boxes0[..., [1, 3]] + top
        scale_boxes[..., [0, 2]] = boxes0[..., [0, 2]] + left
        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / (r * self.img_width)), int(self.img_height / (r * self.img_height)))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask >= 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    # 分割
    def process_box_output_seg(self, output, conf, iou):
        predictions = np.squeeze(output).T
        num_classes = output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        predictions = predictions[scores > conf, :]
        scores = scores[scores > conf]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes_segment(box_predictions)

        # Scale boxes to original image dimensions
        boxes0 = boxes[class_ids == 0, :]
        scores0 = scores[class_ids == 0]
        class_ids0 = class_ids[class_ids == 0]
        mask_predictions0 = mask_predictions[class_ids == 0, :]
        indices0 = self.nms(boxes0, scores0, iou)

        box_out = boxes0[indices0]
        class_out = class_ids0[indices0]
        score_out = scores0[indices0]
        mask_out = mask_predictions0[indices0]

        for i in range(1, num_classes):
            boxes1 = boxes[class_ids == i, :]
            scores1 = scores[class_ids == i]
            class_ids1 = class_ids[class_ids == i]
            mask_predictions1 = mask_predictions[class_ids == i, :]
            indices1 = self.nms(boxes1, scores1, iou)

            box_out = np.append(box_out, boxes1[indices1], axis=0)
            class_out = np.append(class_out, class_ids1[indices1], axis=0)
            score_out = np.append(score_out, scores1[indices1], axis=0)
            mask_out = np.append(mask_out, mask_predictions1[indices1], axis=0)

        return box_out, score_out, class_out, mask_out

    def rescale_boxes(self, boxes):
        r = min(self.new_shape[0] / self.img_height, self.new_shape[1] / self.img_width, 1)
        new_unpad = int(round(self.img_height * r)), int(round(self.img_width * r))
        dh, dw = self.new_shape[0] - new_unpad[0], self.new_shape[1] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        boxes[..., 0] = boxes[..., 0] - left

        boxes[..., 1] = boxes[..., 1] - top

        boxes[..., :4] /= r
        # boxes = np.divide(boxes, r, dtype=np.float32)
        return boxes

    def clip_boxes(self, boxes, shape):
        """限制目标框位置"""
        boxes[..., 0].clip(0, shape[1])
        boxes[..., 1].clip(0, shape[0])
        boxes[..., 2].clip(0, shape[1])
        boxes[..., 3].clip(0, shape[0])

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """将目标框映射到原图"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

        boxes[..., [0, 2]] -= pad[0]
        boxes[..., [1, 3]] -= pad[1]
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def is_in_poly(self, p, poly):
        '''判断点是否在多边形内内'''
        px, py = p
        is_in = False
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:
                    is_in = True
                    break
                elif x > px:
                    is_in = not is_in
        return is_in

    def clip_coords(self, coords, shape):
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y

    def scale_coords(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
        coords[..., 0] /= gain
        coords[..., 1] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def template_matching(self, cv2_img, cv2_img_crop, thres=0.8):
        # 检查图片是否为RGB格式
        if cv2_img.shape[2] != 3:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        result = cv2.matchTemplate(cv2_img, cv2_img_crop, cv2.TM_CCOEFF_NORMED)
        locations = np.unravel_index(np.argmax(result), result.shape)
        #print('最大相似度：', round(float(result[locations]), 3))
        det = []
        if float(result[locations]) >= 0.8:
            locations = locations[::-1]
            det = [locations[0], locations[1], locations[0] + cv2_img_crop.shape[1], locations[1] + cv2_img_crop.shape[0]]
        return det


    def image_deduplication(self, image_new, alert_xyxy, template_thres=0.8, iou_thres=0.5, do_dedup=True, freq='d'):
        '''
            将未重复的告警区域与新图片进行模板匹配，判断最大的相似度，小于template_thres认为不相似，并返回结果
            如果>=template_thres，再计算iou，iou < iou_thres，认为不相似并返回结果
            如果>=iou_threshold，则认为相似，并返回结果

            img_new: numpy.ndarray  需要判断告警是否为重复告警的图片  速度慢可以考虑缩小图片
            alert_xyxy: list[x1, y1, x2, y2]  img_new图片中告警区域
            alert_database: tuple(tuple(image, coord), ...)  已有的告警库，用于与之后的报警进行比对，判断是否重复，如果为空img_new作为第一条数据
                image: numpy.ndarray  第一次出现的告警图片
                coord: tuple(x1, y1, x2, y2)  area在image中的坐标
            do_dedup: Bool, 是否进行去重
            freq: str, 图片库重置频率，默认为d=天，只能是m=分钟，h=小时，d=天

            return:
                False 为新告警
                True 为重复告警
        '''
        if do_dedup:
            # time freq setting
            if freq == 'd':
                time_format = "%Y%m%d"
                time_delta = 86400
            elif freq == 'h':
                time_format = "%Y%m%d-%H"
                time_delta = 3600
            elif freq == 'm':
                time_format = "%Y%m%d-%H%M"
                time_delta = 60
            else:
                raise ValueError('重复图片库重置频率错误')
            # time stamp
            tstruct = time.localtime()
            former_day_str = time.strftime(time_format, time.localtime(time.mktime(tstruct) - time_delta))  # YYYYMMDD
            # clean old images
            _ = self.unique_set.delete_set(prefix=former_day_str)
            # load image set
            alert_database = self.unique_set.load_set(prefix=time.strftime(time_format, tstruct))
            # process
            if len(alert_database) == 0:
                _ = self.unique_set.add_img(image_data=image_new,
                                            prefix=time.strftime("%Y%m%d-%H%M%S", tstruct) + '_' + "-".join(list(map(str, alert_xyxy))))
                return False
            else:
                for image, coord in alert_database:
                    det = self.template_matching(image_new, image[coord[1]: coord[3], coord[0]: coord[2]], template_thres)
                    if det and self.filter_by_iou(alert_xyxy, det, filter_iou=iou_thres, reverse=True):
                        return True
                _ = self.unique_set.add_img(image_data=image_new,
                                            prefix=time.strftime("%Y%m%d-%H%M%S", tstruct) + '_' + "-".join(list(map(str, alert_xyxy))))
                return False
        else:
            return False

    '''
    __________________________________________
        filter of small objects
    __________________________________________
    '''

    def filter_by_size(self, obj_list, filter_size=48, reverse=None):
        # obj_list: output from self.process_output -> (box, score, class)
        box, score, cls = obj_list
        box_out, score_out, cls_out = [], [], []
        if len(box) > 0:
            for a, b, c in zip(box.tolist(), score.tolist(), cls.tolist()):
                if reverse:
                    if abs(a[0] - a[2]) <= filter_size and abs(a[1] - a[3]) <= filter_size:
                        box_out.append(a)
                        score_out.append(b)
                        cls_out.append(c)
                else:
                    if abs(a[0] - a[2]) >= filter_size and abs(a[1] - a[3]) >= filter_size:
                        box_out.append(a)
                        score_out.append(b)
                        cls_out.append(c)
        return box_out, score_out, cls_out

    '''
    __________________________________________
        filter of un-connected object pairs by iou
        eg. consider one hat should belong to one person
    __________________________________________
    '''

    def filter_by_iou(self, box1, box2, filter_iou=0.5, mode=None, reverse=None):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        w1 = abs(x11 - x12)
        h1 = abs(y11 - y12)
        w2 = abs(x21 - x22)
        h2 = abs(y21 - y22)
        a1 = w1 * h1
        a2 = w2 * h2
        # intersection
        xi1 = max(x11, x21)
        yi1 = max(y11, y21)
        xi2 = min(x12, x22)
        yi2 = min(y12, y22)
        wi = max(0, xi2 - xi1)
        hi = max(0, yi2 - yi1)
        ai = wi * hi
        if mode:
            if mode in ['min']:
                if reverse:
                    return round(ai / min(a1, a2), 3) >= filter_iou
                else:
                    return round(ai / min(a1, a2), 3) <= filter_iou
            elif mode in ['max']:
                if reverse:
                    return round(ai / max(a1, a2), 3) >= filter_iou
                else:
                    return round(ai / max(a1, a2), 3) <= filter_iou
            elif mode in ['both']:
                if reverse:
                    if round(ai / a1, 3) >= filter_iou and round(ai / a2, 3) >= filter_iou:
                        return True
                    else:
                        return False
                else:
                    if round(ai / a1, 3) <= filter_iou and round(ai / a2, 3) <= filter_iou:
                        return True
                    else:
                        return False
            else:
                return
        else:
            if reverse:
                return round(ai / (a1 + a2 - ai), 3) >= filter_iou
            else:
                return round(ai / (a1 + a2 - ai), 3) <= filter_iou


    def order_4points(self, pts):  # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


    def four_point_transform(self, image, pts, x_offset=3, y_offset=3):  # 透视变换得到矫正后的图像
        rect = self.order_4points(pts)
        (tl, tr, br, bl) = rect
        # 扩大一下车牌范围，保留边缘
        tl = np.array([max(0, tl[0] - x_offset), max(0, tl[1] - y_offset)])
        tr = np.array([min(image.shape[1], tr[0] + x_offset), max(0, tr[1] - y_offset)])
        bl = np.array([max(0, bl[0] - x_offset), min(image.shape[0], bl[1] + y_offset)])
        br = np.array([min(image.shape[1], br[0] + x_offset), min(image.shape[0], br[1] + y_offset)])
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

def dec_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        request_id = kwargs.get('request_id')
        logger.info(f"|{request_id}| Function `{func.__name__}` cost_time(s)：|{time.time() - start_time:.6f}")
        return result

    return wrapper


def get_dict_all_keys(target: dict):
    '''

    :param target: 要获取所有键的dict
    :return: 所有key list
    '''
    output = []

    def get_key(_dict):
        if isinstance(_dict, dict):
            for key in _dict.keys():
                get_key(_dict[key])
                output.append(key)
        if isinstance(_dict, (tuple, list)):
            for i in range(len(_dict)):
                get_key(_dict[i])
        return output

    return get_key(target)