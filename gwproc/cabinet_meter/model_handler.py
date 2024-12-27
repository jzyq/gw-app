import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import time
import math
import yaml

import cv2
import numpy as np
from core.exceptions import error_handler
from utils.commons import ImageHandler
from utils.log_config import get_logger

_cur_dir_=os.path.dirname(__file__)

try:
    from .paddle_ocr.pdocr_onnx import OcrOnnx
except:
    from paddle_ocr.pdocr_onnx import OcrOnnx

logger = get_logger()

# 从中心点掩码中提取中心点坐标
def find_c_loc(c_point_mask, mode='center'):
    # 从掩码轮廓中心点提取坐标
    if mode == 'center':
        contours, hierarchy = cv2.findContours(c_point_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # 面积需大于25,以防噪点影响
            if area < 25:
                continue
            approx = cv2.approxPolyDP(cnt, 2, True)
            x, y, w, h = cv2.boundingRect(approx)
        c_loc = (int(x + 0.5 * w), int(y + 0.5 * h))

    # 从掩码底部中心提取坐标
    elif mode == 'bottom':
        contours, hierarchy = cv2.findContours(c_point_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # 面积需大于25,以防噪点影响
            if area < 25:
                continue
            approx = cv2.approxPolyDP(cnt, 2, True)
            x, y, w, h = cv2.boundingRect(approx)
        c_loc = (int(x + 0.5 * w), int(y + h))
    return c_loc


# 根据一个点和一个角度以及图片的长和宽，计算从此点出发的射线与图片的交点
def find_end_point(h, w, point, angle):
    x, y = point[0], point[1]

    while angle >= 360:
        angle -= 360
    while angle < 0:
        angle += 360

    if angle >= 0 and angle < 90:
        angle_in_radians = math.radians(angle)
        tan_value = math.tan(angle_in_radians)
        if (w - x) * tan_value < y:
            end_point = (w, int(y - (w - x) * tan_value))
        else:
            end_point = (int(x + y / tan_value), 0)

    if angle == 90:
        end_point = (x, 0)

    if angle > 90 and angle < 180:
        angle_in_radians = math.radians(angle - 90)
        tan_value = math.tan(angle_in_radians)
        if x > (y * tan_value):
            end_point = (int(x - y * tan_value), 0)
        else:
            end_point = (0, int(y - x / tan_value))

    if angle == 180:
        end_point = (0, y)

    if angle > 180 and angle < 270:
        angle_in_radians = math.radians(angle - 180)
        tan_value = math.tan(angle_in_radians)
        if (y + x * tan_value) < h:
            end_point = (0, int(y + x * tan_value))
        else:
            end_point = (int(x - (h - y) / tan_value), h)

    if angle == 270:
        end_point = (x, h)

    if angle > 270:
        angle_in_radians = math.radians(angle - 270)
        tan_value = math.tan(angle_in_radians)
        if x + (h - y) * tan_value < w:
            end_point = (int(x + (h - y) * tan_value), h)
        else:
            end_point = (w, int(y + (w - x) / tan_value))

    return end_point


# 输入指针、中心点、刻度的掩码，输出指针的值。
# 可调节参数如下：
# 直线检测阈值（指针的针部分长度需高于此值，尾部需短于此值）、
# 表盘读数范围（按照顺时针顺序）、
# 起始角度（需小于表盘读数第一个刻度对应的角度，但不得越过表盘读数最后一个刻度。可为负数）、
# 角度分辨率（越小则读数约精确，但计算量越大）
def detect_pointer(pointer_mask, c_point_mask, dial_mask, mode, pointer_thre, dial_range, start_angle=0,
                   resolution=1.0):
    # 读取图片大小信息，三个mask的shape应该相同
    h, w = dial_mask.shape[0], dial_mask.shape[1]

    # 创建待扫描角度列表
    angle, angles = start_angle, []
    while angle < 360 + start_angle:
        angles.append(angle)
        angle += resolution

    # 获取中心点坐标
    # c_loc = find_c_loc(c_point_mask)
    c_loc = find_c_loc(c_point_mask, mode=mode)  # ocr部分

    # 获取刻度覆盖的角度范围以及指针的角度
    dial_angles, pointer_angles = [], []
    for angle in angles:
        # 以中心点为中心扫描全图
        end_point = find_end_point(h, w, c_loc, angle)
        temp_img = np.zeros((h, w, 3), np.uint8)
        # cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 2)
        cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 4)
        # cv2.imshow('image', temp_img)
        # cv2.waitKey(0)
        line_img = cv2.threshold(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

        # 获取扫描线与刻度、指针的重合区域
        dial_mixed = cv2.bitwise_and(dial_mask, line_img)
        pointer_mixed = cv2.bitwise_and(pointer_mask, line_img)
        # cv2.imshow('image', pointer_mixed)
        # cv2.waitKey(0)

        # 若扫描线与刻度有重合区域，则认为扫描线的角度处于刻度的范围之内
        contours, hierarchy = cv2.findContours(dial_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            dial_angles.append(angle)

        # 若扫描线与指针有重合区域且直线检测能检测出指针部分的长度，则认为扫描线的角度处于指针的范围之内
        contours, hierarchy = cv2.findContours(pointer_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            lines = cv2.HoughLines(pointer_mixed, 1, np.pi / 180, int(pointer_thre * math.sqrt(w ** 2 + h ** 2)))
            try:
                if len(lines) > 0:
                    pointer_angles.append(angle)
            except:
                pass

    # 获取刻度所对应的角度范围以及指针对应的角度
    dial_value_range = (dial_angles[0], dial_angles[-1])
    pointer_value = np.mean(pointer_angles)

    # 计算指针读数
    result_value = dial_range[0] + (dial_range[1] - dial_range[0]) * (pointer_value - dial_value_range[0]) / (
            dial_value_range[1] - dial_value_range[0])

    return result_value


# 输入指针、中心点、刻度的掩码，输出指针的值。
# 可调节参数如下：
# 直线检测阈值（指针的针部分长度需高于此值，尾部需短于此值）、
# 表盘读数范围（按照顺时针顺序）、
# 起始角度（需小于表盘读数第一个刻度对应的角度，但不得越过表盘读数最后一个刻度。可为负数）、
# 角度分辨率（越小则读数约精确，但计算量越大）
def detect_pointer_oil_level(pointer_mask, c_point_mask, dial_mask, mode, pointer_thre, dial_range, start_angle=0,
                             resolution=1.0):
    # 读取图片大小信息，三个mask的shape应该相同
    h, w = dial_mask.shape[0], dial_mask.shape[1]

    # 创建待扫描角度列表
    angle, angles = start_angle, []
    while angle < 360 + start_angle:
        angles.append(angle)
        angle += resolution

    # 获取中心点坐标
    # c_loc = find_c_loc(c_point_mask)
    c_loc = find_c_loc(c_point_mask, mode=mode)  # ocr部分

    # 获取刻度覆盖的角度范围以及指针的角度
    dial_angles, pointer_angles = [], []
    for angle in angles:
        # 以中心点为中心扫描全图
        end_point = find_end_point(h, w, c_loc, angle)
        temp_img = np.zeros((h, w, 3), np.uint8)
        # cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 2)
        cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 4)
        # cv2.imshow('image', temp_img)
        # cv2.waitKey(0)
        line_img = cv2.threshold(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

        # 获取扫描线与刻度、指针的重合区域
        dial_mixed = cv2.bitwise_and(dial_mask, line_img)
        pointer_mixed = cv2.bitwise_and(pointer_mask, line_img)
        # cv2.imshow('image', pointer_mixed)
        # cv2.waitKey(0)

        # 若扫描线与刻度有重合区域，则认为扫描线的角度处于刻度的范围之内
        contours, hierarchy = cv2.findContours(dial_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            dial_angles.append(angle)

        # 若扫描线与指针有重合区域且直线检测能检测出指针部分的长度，则认为扫描线的角度处于指针的范围之内
        contours, hierarchy = cv2.findContours(pointer_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            lines = cv2.HoughLines(pointer_mixed, 1, np.pi / 180, int(pointer_thre * math.sqrt(w ** 2 + h ** 2)))
            try:
                if len(lines) > 0:
                    pointer_angles.append(angle)
            except:
                pass

    # 获取刻度所对应的角度范围以及指针对应的角度
    dial_value_range = (dial_angles[0], dial_angles[-1])
    # 去除刻度范围外的指针角度
    pointer_angle = []
    for i in pointer_angles:
        if dial_value_range[0] < i < dial_value_range[1]:
            pointer_angle.append(i)
    pointer_value = np.mean(pointer_angle)

    # 计算指针读数
    result_value = dial_range[0] + (dial_range[1] - dial_range[0]) * (pointer_value - dial_value_range[0]) / (
            dial_value_range[1] - dial_value_range[0])

    return result_value


# 对不均匀表盘识别读数
def detect_pointer_uneven(pointer_mask, c_point_mask, dial_mask, mode, pointer_thre,
                          dial_ranges, start_angle=0, resolution=1):
    # 读取图片大小信息，三个mask的shape应该相同
    h, w = dial_mask.shape[0], dial_mask.shape[1]
    dial_ranges = sorted(dial_ranges)
    # 创建待扫描角度列表
    angle, angles = start_angle, []
    while angle < 360 + start_angle:
        angles.append(angle)
        angle += resolution

    # 获取中心点坐标
    # c_loc = find_c_loc(c_point_mask)
    c_loc = find_c_loc(c_point_mask, mode=mode)

    # 获取刻度覆盖的角度范围以及指针的角度
    dial_angles, pointer_angles = [], []
    for angle in angles:
        # 以中心点为中心扫描全图
        end_point = find_end_point(h, w, c_loc, angle)
        temp_img = np.zeros((h, w, 3), np.uint8)
        cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 2)
        # cv2.imshow('image', temp_img)
        # cv2.waitKey(0)
        line_img = cv2.threshold(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

        # 获取扫描线与刻度、指针的重合区域
        dial_mixed = cv2.bitwise_and(dial_mask, line_img)
        pointer_mixed = cv2.bitwise_and(pointer_mask, line_img)

        # 若扫描线与刻度有重合区域，则认为扫描线的角度处于刻度的范围之内
        contours, hierarchy = cv2.findContours(dial_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            dial_angles.append(angle)

        # 若扫描线与指针有重合区域且直线检测能检测出指针部分的长度，则认为扫描线的角度处于指针的范围之内
        contours, hierarchy = cv2.findContours(pointer_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            lines = cv2.HoughLines(pointer_mixed, 1, np.pi / 180, int(pointer_thre * math.sqrt(w ** 2 + h ** 2)))
            try:
                if len(lines) > 0:
                    pointer_angles.append(angle)
            except:
                pass

    # 获取刻度所对应的角度范围以及指针对应的角度
    dial_value_range = (dial_angles[0], dial_angles[-1])
    pointer_value = np.mean(pointer_angles)

    # 计算指针相对于顺时针初始刻度旋转的比例
    pointer_rate = (pointer_value - dial_value_range[0]) / (dial_value_range[1] - dial_value_range[0])
    # 计算指针读数
    for index in range(len(dial_ranges) - 1):
        # 根据指针旋转比例分配到小区间处理
        if pointer_rate > dial_ranges[index][0] and pointer_rate < dial_ranges[index + 1][0]:
            result_value = dial_ranges[index][1] + (dial_ranges[index + 1][1] - dial_ranges[index][1]) * (
                    pointer_rate - dial_ranges[index][0]) / (dial_ranges[index + 1][0] - dial_ranges[index][0])
    return result_value


def detect_pointer2(pointer_mask, c_point_mask, dial_mask, mode, pointer_thre, dial_range, start_angle=0,
                    resolution=1.0):
    # 读取图片大小信息，三个mask的shape应该相同
    h, w = dial_mask.shape[0], dial_mask.shape[1]

    # 创建待扫描角度列表
    angle, angles = start_angle, []
    while angle < 360 + start_angle:
        angles.append(angle)
        angle += resolution

    # 获取中心点坐标
    # c_loc = find_c_loc(c_point_mask)
    c_loc = find_c_loc(c_point_mask, mode=mode)  # ocr部分

    # 获取刻度覆盖的角度范围以及指针的角度
    dial_angles, pointer_angles = [], []
    for angle in angles:
        # 以中心点为中心扫描全图
        end_point = find_end_point(h, w, c_loc, angle)
        temp_img = np.zeros((h, w, 3), np.uint8)
        # cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 2)
        cv2.line(temp_img, c_loc, end_point, (255, 255, 255), 4)
        # cv2.imshow('image', temp_img)
        # cv2.waitKey(0)
        line_img = cv2.threshold(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

        # 获取扫描线与刻度、指针的重合区域
        dial_mixed = cv2.bitwise_and(dial_mask, line_img)
        # cv2.imshow('image', pointer_mixed)
        # cv2.waitKey(0)

        # 若扫描线与刻度有重合区域，则认为扫描线的角度处于刻度的范围之内
        contours, hierarchy = cv2.findContours(dial_mixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            dial_angles.append(angle)

    # 获取刻度所对应的角度范围以及指针对应的角度
    dial_value_range = (dial_angles[0], dial_angles[-1])
    # pointer_value = np.mean(pointer_angles)
    pointer_angles = []
    lines = cv2.HoughLinesP(pointer_mask, 1, np.pi / 180, 40, minLineLength=10, maxLineGap=1000)
    for line in lines:
        # newlines1 = lines[:, 0, :]
        x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        cv2.line(pointer_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
        # 转换为浮点数，计算斜率
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        if x2 - x1 == 0:
            result = 90
        elif y2 - y1 == 0:
            result = 0
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
        pointer_angles.append(result)
    pointer_value = np.mean(pointer_angles)
    pointer_value = pointer_value if pointer_value > 0 else pointer_value + 180

    # 计算指针读数
    result_value = dial_range[0] + (dial_range[1] - dial_range[0]) * (pointer_value - dial_value_range[0]) / (
            dial_value_range[1] - dial_value_range[0])

    return result_value


class IndicatorMeterDetectCabinetMeterHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()
        
        self.model_name = 'cabinet_meter'

        self.read_config()

        # 目标检测
        self.classes_detect = ['cabinet_meter',
                               'lightning_rod',
                               'lightning_rod_ammeter',
                               'lightning_rod_ammeter_dial',
                               'lightning_rod_dial',
                               'oil_level',
                               'oil_temperature',
                               'pressure',
                               'pressure1']
        self.num_classes = len(self.classes_detect)

        # 表盘
        self.classes_dial = ['scale', 'centre']

        # 指针
        self.classes_pointer = ['pointer']

        # 表盘分类
        self.standard_word_list = [
            ['uneven_20A', ['0', '5', '10', '15', '20', 'A']],
            ['uneven_40A', ['0', '10', '20', '30', '40', 'A']],
            ['uneven_50A', ['0', '20', '30', '40', '50', 'A']],
            ['uneven_100A', ['0', '40', '60', '80', '100', 'A']],
            ['even_40kV', ['0', '10', '20', '30', '40', 'kV', 'KV', 'Kv', 'kv']],
            ['even_200A', ['0', '50', '100', '150', '200', 'A']],
            ['even_1200V', ['0', '400', '800', '1200', 'V', 'v']],
        ]

        # 读数计算
        self.resolution = 0.5
        self.start_angle = 0
        self.meter_infos = {
            'even_40kV': [0.14, [40, 0], 1],
            'even_200A': [0.11, [200, 0], 0],
            'even_1200V': [0.10, [1200, 0], -1],
            'uneven_20A': [0.15,
                           [[1.0, 0.0], [0.9196961326782471, 5.0], [0.8698657325101639, 6.0], [0.803664114548678, 7.0],
                            [0.7340883786067162, 8.0], [0.656733578566477, 9.0], [0.5755832689414397, 10.0],
                            [0.5002280863674992, 11.0], [0.43268309757626683, 12.0], [0.365590668904685, 13.0],
                            [0.3003673691189864, 14.0], [0.23786130053072435, 15.0], [0.18437426272345891, 16.0],
                            [0.12907842492979632, 17.0], [0.07995477109475717, 18.0], [0.03990529762186365, 19.0],
                            [0.0, 20.0]], 1],
            'uneven_40A': [0.12,
                           [[1.0, 0.0], [0.9186263376262647, 10.0], [0.8652887562340307, 12.0],
                            [0.8016411669632394, 14.0],
                            [0.7300759794292425, 16.0], [0.6567485027052288, 18.0], [0.57127197209844, 20.0],
                            [0.4986446815140031, 22.0], [0.4288409623562486, 24.0], [0.36205482641146275, 26.0],
                            [0.29739666031198103, 28.0], [0.23391892489886632, 30.0], [0.18178935625023937, 32.0],
                            [0.1293989575704947, 34.0], [0.08009972870407879, 36.0], [0.03987430904522452, 38.0],
                            [0.0, 40.0]], 0],
            'uneven_50A': [0.10,
                           [[1.0, 0.0], [0.9347583253167938, 10.0], [0.9017334574989727, 12.0],
                            [0.8607399954458526, 14.0],
                            [0.8212884228699239, 16.0], [0.7768165276460272, 18.0], [0.7223820769570708, 20.0],
                            [0.6647369538800743, 22.0], [0.6007635057130876, 24.0], [0.5377961811816979, 26.0],
                            [0.4759652748708359, 28.0], [0.41999464405531184, 30.0], [0.36755825908147705, 32.0],
                            [0.318269197209844, 34.0], [0.27062792908120303, 36.0], [0.22544797978931233, 38.0],
                            [0.18711495394531916, 40.0], [0.17214569875816382, 42.0], [0.1074959815478507, 44.0],
                            [0.06902683270092508, 46.0], [0.03303868020058539, 48.0], [0.0, 50.0]], 0],
            'uneven_100A': [0.10,
                            [[1.0, 0.0], [0.937888753910799, 20.0], [0.8849313631713571, 25.0],
                             [0.8320162457339485, 30.0],
                             [0.7808781669044043, 35.0], [0.7286373752090639, 40.0], [0.6519438824202003, 45.0],
                             [0.574811136125124, 50.0], [0.49901121661488557, 55.0], [0.4237929192501577, 60.0],
                             [0.3626045604142394, 65.0], [0.3029849097426346, 70.0], [0.24454818129219105, 75.0],
                             [0.18853447162798825, 80.0], [0.1392109451409674, 85.0], [0.08786173529182338, 90.0],
                             [0.04512213960901985, 95.0], [0.0, 100.0]], 0],
        }

        self.platform = platform
        
        if self.platform == 'ONNX':
            import onnxruntime as ort

            self.inference_sessions = {
                'detect': ort.InferenceSession(os.path.join(_cur_dir_, 'models/detect.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'dial': ort.InferenceSession(os.path.join(_cur_dir_, 'models/dial.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pointer': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pointer.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pdocr_cls': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_cls.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pdocr_det': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_det.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pdocr_rec': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_rec.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
            }

            logger.info(f'Model {self.model_name} Loaded')

        elif self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel
            import onnxruntime as ort

            self.device_id = device_id
            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            self.inference_sessions = {
                'detect': AclLiteModel(os.path.join(_cur_dir_, 'models/detect.om')),
                'dial': AclLiteModel(os.path.join(_cur_dir_, 'models/dial.om')),
                'pointer': AclLiteModel(os.path.join(_cur_dir_, 'models/pointer.om')),
                'pdocr_cls': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_cls.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pdocr_det': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_det.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pdocr_rec': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pdocr_rec.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
            }

            logger.info(f'Model {self.model_name} Loaded on {self.device}')
        else:
            # TO DO: should not be here, should report an error
            pass       
            
    def read_config(self):
        # 读取配置文件
        _config_file = os.path.join(_cur_dir_, 'config.yaml')
        with open(_config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 访问参数
        self.conf_detect = config['detect']['conf_thres']
        self.iou_detect = config['detect']['iou_thres']
        self.filter_size = config['detect']['filter_size']

        self.conf_dial = config['dial']['conf_thres']
        self.iou_dial = config['dial']['iou_thres']

        self.conf_pointer = config['pointer']['conf_thres']
        self.iou_pointer = config['pointer']['iou_thres']

    def release(self):
        if self.platform == 'ASCEND':
            for _sess in self.inference_sessions:
                del _sess
            del self.resource
            
            logger.info(f'Model {self.model_name} Relased on {self.device}')

        else:
            logger.info(f'Model {self.model_name} Relased')


    def run_inference(self, image_files, extra_args=None):
        _images_data = []
        
        image_result = True
        for _image_file in image_files:
            try:
                from ..gwproc import GWProc
            except:
                from gwproc import GWProc

            _result, _data = GWProc.read_image(_image_file)
            
            if _result:
                _encoded_str = base64.urlsafe_b64encode(_data)
                _images_data.append(_encoded_str.decode('utf8'))
            else:
                image_result = False
        if image_result == False:
            #TODO: generate error response for image failure
            pass

        if extra_args is not None:
            logger.info(f'{self.model_name}不支持extra_args, 忽略')

        payload = {
            'task_tag': 'indicator_meter_detect',
            'image_type': 'base64',
            'images': _images_data,
        }

        data = self.preprocess(payload)
        data = self.inference(data)
        data = self.postprocess(data)

        return json.dumps(data, indent=4, ensure_ascii=False)

    def preprocess(self, data, *args, **kwargs):
        return data

    @error_handler
    def inference(self, data, *args, **kwargs):
        image_type = data.get('image_type')
        images = data.get('images')
        extra_args = data.get('extra_args')
        conf = self.conf_detect
        iou = self.iou_detect
        filter_size = self.filter_size
        if extra_args:
            for extra_arg in extra_args:
                if extra_arg.get('model') == self.model_name:
                    param = extra_arg.get('param')
                    conf = param.get("conf")
                    iou = param.get("iou")
                    filter_size = param.get("filter_size")
                    if conf is None:
                        conf = self.conf_detect
                    if iou is None:
                        iou = self.iou_detect
                    if filter_size is None:
                        filter_size = self.filter_size
                    break

        assert image_type == 'base64', f'image_type: {image_type}'

        data = {"code": 200, "data": [], "message": "", "time": 0}

        sess_detect = self.inference_sessions.get('detect')
        sess_dial = self.inference_sessions.get('dial')
        sess_pointer = self.inference_sessions.get('pointer')
        sess_pdocr_cls = self.inference_sessions.get('pdocr_cls')
        sess_pdocr_rec = self.inference_sessions.get('pdocr_rec')
        sess_pdocr_det = self.inference_sessions.get('pdocr_det')

        text_sys = OcrOnnx(cls_sess=sess_pdocr_cls, det_sess=sess_pdocr_det, rec_sess=sess_pdocr_rec)
        
        if self.platform =='ONNX':
            input_name_detect = sess_detect.get_inputs()[0].name
            label_name_detect = [i.name for i in sess_detect.get_outputs()]
            # 指针分割
            input_name_pointer = sess_pointer.get_inputs()[0].name
            label_name_pointer = [i.name for i in sess_pointer.get_outputs()]
            # 刻度和中心点分割
            input_name_center = sess_dial.get_inputs()[0].name
            label_name_center = [i.name for i in sess_dial.get_outputs()]

        for image_num, base64_str in enumerate(images):
            img0 = self.base64_to_cv2(base64_str)
            img = self.prepare_input(img0, swapRB=False)

            if self.platform == 'ONNX':
                output = sess_detect.run(label_name_detect, {input_name_detect: img}, **kwargs)[0]
            elif self.platform == 'ASCEND':
                output = sess_detect.execute([img])[0]
            box_out, score_out, class_out = self.filter_by_size(self.process_output(output, conf, iou),
                                                                filter_size=filter_size)
            box_out = np.array(box_out)
            score_out = np.array(score_out)
            class_out = np.array(class_out)
            defect_data = []
            if len(class_out):
                box_out0 = box_out[class_out == 0]
                score_out0 = score_out[class_out == 0]

                if len(box_out0) > 0:
                    for idx in range(len(box_out0)):
                        conf_detc = int(score_out0[idx] * 100)
                        xyxy = box_out0[idx]

                        # 去除检测到的位于图片边缘不可识别读数的表
                        if img0.shape[1] - 100 < (xyxy[0] + xyxy[2]) / 2 or 150 > (xyxy[0] + xyxy[2]) / 2 or img0.shape[
                            0] - 100 < (xyxy[1] + xyxy[3]) / 2 or 150 > (xyxy[1] + xyxy[3]) / 2:
                            continue

                        '''切图，分割，算数'''
                        # 切图
                        img_seg0 = img0[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                        shape = img_seg0.shape
                        if shape[0] < 640 and shape[1] < 640:
                            r = min(640 / shape[0], 640 / shape[1])
                            img_seg0 = cv2.resize(img_seg0, None, None, r, r)
                        # 预处理
                        img_seg = self.prepare_input(img_seg0, swapRB=False)
                        # 指针分割
                        if self.platform == 'ONNX':
                            output_pointer = sess_pointer.run(label_name_pointer, {input_name_pointer: img_seg}, **kwargs)
                        elif self.platform == 'ASCEND':
                            output_pointer = sess_pointer.execute([img_seg])
                        

                        box_out, score_out, class_out, mask_pred = self.process_box_output_seg(output_pointer[0],
                                                                                               self.conf_pointer,
                                                                                               self.iou_pointer)
                        if len(box_out) > 0:
                            mask_maps = self.process_mask_output(box_out, mask_pred, output_pointer[1])
                            # 指针mask
                            mask_pointer = np.where(mask_maps[0] > 0, 255, 0)

                            # 刻度和中心点分割
                            if self.platform == 'ONNX':
                                output_dial = sess_dial.run(label_name_center, {input_name_center: img_seg}, **kwargs)
                            elif self.platform == 'ASCEND':
                                output_dial = sess_dial.execute([img_seg])
                            box_out, score_out, class_out, mask_pred = self.process_box_output_seg(output_dial[0],
                                                                                                   self.conf_dial,
                                                                                                   self.iou_dial)
                            if 0 in class_out and 1 in class_out:
                                mask_maps = self.process_mask_output(box_out, mask_pred, output_dial[1])
                                # 刻度和中心点mask
                                for i, mask in enumerate(mask_maps):
                                    if class_out[i] == 0:
                                        # 刻度mask
                                        mask_scale = np.where(mask > 0, 255, 0)
                                    elif class_out[i] == 1:
                                        # 中心点mask
                                        mask_centre = np.where(mask > 0, 255, 0)

                                # ocr
                                ocr_result = text_sys(img_seg0, **kwargs)

                                # 判断表类别
                                word_list = []
                                for i in ocr_result.get('results')[0]:
                                    word_list.append(i.get('text'))
                                sim_word_count = []
                                for standard_word in self.standard_word_list:
                                    sim_word_count.append(len(set(standard_word[1]).intersection(set(word_list))))
                                meter_type = self.standard_word_list[sim_word_count.index(max(sim_word_count))][0]

                                # 计算读数
                                mask_pointer = mask_pointer.astype(np.uint8)
                                mask_centre = mask_centre.astype(np.uint8)
                                mask_scale = mask_scale.astype(np.uint8)

                                pointer_thre, dial_range, ndigits = self.meter_infos[meter_type]
                                if meter_type.split('_')[0] == 'even':
                                    reading = detect_pointer(mask_pointer, mask_centre, mask_scale, mode='bottom',
                                                             pointer_thre=pointer_thre, dial_range=dial_range,
                                                             start_angle=self.start_angle, resolution=self.resolution)
                                else:
                                    reading = detect_pointer_uneven(mask_pointer, mask_centre, mask_scale,
                                                                    mode='bottom',
                                                                    pointer_thre=pointer_thre, dial_ranges=dial_range,
                                                                    start_angle=self.start_angle,
                                                                    resolution=self.resolution)

                                if ndigits == 0:
                                    reading = round(reading)
                                elif ndigits < 0:
                                    reading = int(round(reading, ndigits))
                                else:
                                    reading = round(reading, ndigits)
                                result_dict = {"defect_name": "indicator_meter_detect",
                                               "defect_desc": "指针表计识别：柜体表计",
                                               "confidence": conf_detc,
                                               "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                                               "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                                               "class": self.model_name,
                                               "extra_info": {'reading': reading,
                                                              'range': meter_type.split('_')[1]}
                                               }
                                defect_data.append(result_dict)

            data['data'].append({'defect_data': defect_data, 'image_tag': 'img' + str(image_num + 1)})
        data['time'] = int(round(time.time() * 1000))
        return data

    def postprocess(self, data, *args, **kwargs):
        return data

# Note: ftp服务器在首次实例化GWProc时设置，无法在此单独进行测试

"""
if __name__ == '__main__':
    input_images = [os.path.join(_cur_dir_,'test_case/cabinet_meter_20A.jpg')]

    obj = IndicatorMeterDetectCabinetMeterHandler(platform='ONNX')
    
    results = obj.run_inference(input_images)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
"""