import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import time
import math
import yaml

import numpy as np
from core.exceptions import error_handler
from utils.commons import ImageHandler
from utils.log_config import get_logger

logger = get_logger()

_cur_dir_=os.path.dirname(__file__)

class PointerMeterDetectLightningRodCurrentMeterHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()
        self.model_name = 'lightning_rod_current_meter'
        
        self.read_config()
        
        # 目标检测
        self.classes_detect = ['lightning_rod_current_meter0',
                               'dial0',
                               'lightning_rod_current_meter1',
                               'dial1',
                               'lightning_rod_current_meter2',
                               'lightning_rod_current_meter3']
        self.num_classes = len(self.classes_detect)

        # 雷击计数器
        self.classes_digit = ['0', '1', '2', '5', '6', '8']

        '''新增避雷针表计2'''
        self.classes_pose2 = ['dian0', 'dian1', '0', '1', '2', '3']  # 'dian0' 指针远刻度端
        self.kpt_shape3 = [len(self.classes_pose2), 3]

        self.platform = platform
        
        if self.platform == 'ONNX':
            import onnxruntime as ort

            self. inference_sessions = {
                'detect': ort.InferenceSession(os.path.join(_cur_dir_, 'models/detect.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'digit': ort.InferenceSession(os.path.join(_cur_dir_, 'models/digit.onnx'), providers=['AzureExecutionProvider', 'CPUExecutionProvider']),
                'pose2': ort.InferenceSession(os.path.join(_cur_dir_, 'models/pose2.onnx'),providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
            }

            logger.info(f'Model {self.model_name} Loaded')

        elif self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device_id = device_id
            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            self. inference_sessions = {
                'detect': AclLiteModel(os.path.join(_cur_dir_, 'models/detect.om')),
                'digit': AclLiteModel(os.path.join(_cur_dir_, 'models/digit.om')),
                'pose2': AclLiteModel(os.path.join(_cur_dir_, 'models/pose2.om'))
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

        # 目标检测
        self.conf_detect = config['detect']['conf_thres']
        self.iou_detect = config['detect']['iou_thres']
        self.filter_size = config['detect']['filter_size']

        # 雷击计数器
        self.conf_digit = config['digit']['conf_thres']
        self.iou_digit = config['digit']['iou_thres']

        # 避雷针表记2
        self.conf_pose2 = config['pose2']['conf_thres']
        self.iou_pose2 = config['pose2']['iou_thres']

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
            'task_tag': 'pointer_meter_detect',
            'image_type': 'base64',
            "images": _images_data
        }

        data = self.preprocess(payload)
        data = self.inference(data)
        data = self.postprocess(data)

        return json.dumps(data, indent=4, ensure_ascii=False)

    def angel_calculate(self, point_yuandian, point):
        point_yuandian = point_yuandian[::-1] * -1
        point = point[::-1] * -1
        angle = math.atan2(-point[1] + point_yuandian[1], point[0] - point_yuandian[0])
        angle = math.degrees(angle)
        if angle < 0:
            angle += 360
        return angle

    def calc_abc_from_line_2d(self, x0, y0, x1, y1):
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    def get_line_cross_point(self, line1, line2):
        a0, b0, c0 = self.calc_abc_from_line_2d(*line1)
        a1, b1, c1 = self.calc_abc_from_line_2d(*line2)
        D = a0 * b1 - a1 * b0
        if D == 0:
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return x, y

    def get_distance(self, p0, p1):
        return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def cal(self, preds_kpts, defect_data, conf_detc, x1, y1, x2, y2, reading_count):
        for pred_kpts in preds_kpts:
            if pred_kpts[0][2] < 0.5 or pred_kpts[1][2] < 0.5:
                continue
            kpt_dian0 = pred_kpts[0][:2]
            kpt_dian1 = pred_kpts[1][:2]
            kpt_scale = pred_kpts[2:]
            scale_index = kpt_scale[:, -1] > 0.5
            for i, kpt in enumerate(kpt_scale):
                if kpt[-1] > 0.5:
                    kpt[-1] = i
            kpt_scale = kpt_scale[scale_index]

            left_index = np.argmax(kpt_dian1[0] - kpt_scale[:, 0])
            left = kpt_scale[left_index]
            if len(kpt_scale) < (left_index + 1):
                continue
            right = kpt_scale[left_index + 1]
            node = self.get_line_cross_point(np.append(kpt_dian0, kpt_dian1),
                                             np.append(left[:2], right[:2]))
            ratio = (right[-1] - left[-1]) / self.get_distance(left[:2], right[:2])
            reading = left[-1] + self.get_distance(node, left[:2]) * ratio
            defect_data.append({"defect_name": "pointer_meter_detect",
                                "defect_desc": "指针表计识别：避雷针电流表",
                                "confidence": conf_detc,
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "class": self.model_name,
                                "extra_info": {
                                    'reading': round(reading, 2),
                                    'reading_count': int(reading_count)
                                },
                                })
        return defect_data

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

        # 表类别检测模型
        sess_detect = self.inference_sessions.get('detect')
        sess_digit = self.inference_sessions.get('digit')
        sess_pose2 = self.inference_sessions.get('pose2')
        
        if self.platform =='ONNX':
            input_name_detect = sess_detect.get_inputs()[0].name
            label_name_detect = [i.name for i in sess_detect.get_outputs()]
            # 雷击数检测
            input_name = sess_digit.get_inputs()[0].name
            label_name = [i.name for i in sess_digit.get_outputs()]

            '''新增避雷针表2'''
            input_name_pose2 = sess_pose2.get_inputs()[0].name
            label_name_pose2 = [i.name for i in sess_pose2.get_outputs()]

        for image_num, base64_str in enumerate(images):
            img0 = self.base64_to_cv2(base64_str)
            img = self.prepare_input(img0, swapRB=False)

            # 表类别检测
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

                # 新增避雷针表2
                box_out4 = box_out[class_out == 4]
                score_out4 = score_out[class_out == 4]
                if len(box_out4) > 0:
                    for idx, box in enumerate(box_out4):
                        conf_detc = int(score_out4[idx] * 100)
                        # 如果xmin或ymin小于0，则设置等于0
                        if box[0] < 0:
                            box[0] = 0
                        if box[1] < 0:
                            box[1] = 0
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        img_pose = img0[y1:y2, x1:x2]

                        img = self.prepare_input(img_pose, swapRB=False)
                        # 雷击数检测
                        if self.platform == 'ONNX':
                            output_digit = sess_digit.run(label_name, {input_name: img}, **kwargs)
                        elif self.platform == 'ASCEND':
                            output_digit = sess_digit.execute([img])

                        box_out_digit, _, class_out_digit = self.filter_by_size(
                            self.process_output(output_digit, self.conf_digit, self.iou_digit),
                            filter_size=filter_size)
                        if len(box_out_digit) > 1:
                            x_list = []
                            cls_list = []
                            for class_index, class_num in enumerate(class_out_digit):
                                cls = self.classes_digit[class_num]
                                x = box_out_digit[class_index][0]
                                cls_list.append(cls)
                                x_list.append(x)
                            cls_sorted_index = [i for i, _ in sorted(enumerate(x_list), key=lambda x: x[1])]
                            cls_list = sorted(cls_list, key=lambda x: cls_sorted_index[cls_list.index(x)])
                            reading_count = ''
                            for cls in cls_list:
                                reading_count += cls
                            reading_count = int(reading_count)
                        else:
                            reading_count = 0

                        img_pose = self.prepare_input(img_pose, swapRB=False)
                        if self.platform == 'ONNX':
                            preds_pose = sess_pose2.run(label_name_pose2, {input_name_pose2: img_pose})[0]
                        elif self.platform == 'ASCEND':
                            preds_pose = sess_pose2.execute([img])[0]
                        preds_pose = self.process_box_output(preds_pose, self.conf_pose2, self.iou_pose2)[0]
                        if len(preds_pose) > 0:
                            preds_kpts = preds_pose[:, 6:].reshape([len(preds_pose)] + self.kpt_shape3)
                            preds_kpts = self.scale_coords(self.new_shape, preds_kpts, img_pose.shape)

                            self.cal(preds_kpts, defect_data, conf_detc, x1, y1, x2, y2, reading_count)

            data['data'].append({'defect_data': defect_data, 'image_tag': 'img' + str(image_num + 1)})
        data['time'] = int(round(time.time() * 1000))
        return data

    def postprocess(self, data, *args, **kwargs):
        return data

# Note: ftp服务器在首次实例化GWProc时设置，无法在此单独进行测试

"""

if __name__ == '__main__':
    input_images = [os.path.join(_cur_dir_,'test_case/lightning_rod_current_meter2.png')]

    obj = PointerMeterDetectLightningRodCurrentMeterHandler(platform='ONNX')
    
    results = obj.run_inference(input_images)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
"""