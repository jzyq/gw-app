import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import yaml
import time
import numpy as np

from core.exceptions import error_handler
from utils.commons import ImageHandler, dec_timer
from utils.log_config import get_logger

logger = get_logger()

_cur_dir_=os.path.dirname(__file__)

class HatDetectHatHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()
        self.read_config()
        
        self.new_shape = [960, 960]
        self.model_name = 'hat'
        self.classes = ['helmet', "no_helmet"]
        self.num_classes = 2
        
        self.platform = platform
        
        if self.platform == 'ONNX':
            import onnxruntime as ort

            sess1 = ort.InferenceSession(
                os.path.join(_cur_dir_, 'models/hat.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            sess_person = ort.InferenceSession(
                os.path.join(_cur_dir_, 'models/person.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            logger.info(f'Model {self.model_name} Loaded')

        elif self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device_id = device_id
            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            sess1 = AclLiteModel(os.path.join(_cur_dir_, 'models/hat.om'))
            sess_person = AclLiteModel(os.path.join(_cur_dir_, 'models/person.om'))

            logger.info(f'Model {self.model_name} Loaded on {self.device}')
        else:
            # TO DO: should not be here, should report an error
            pass
        
        self.inference_sessions = {'hat': sess1, 'person': sess_person}
       
    def read_config(self):
        # 读取配置文件
        _config_file = os.path.join(_cur_dir_, 'config.yaml')
        with open(_config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 访问参数
        self.conf = config['detect']['conf_thres']
        self.iou = config['detect']['iou_thres']

        self.filter_size = config['filter']['filter_size']
        
        self.do_dedup = (config['dedup']['enable']==1)
        self.dedup_template_thres = config['dedup']['template_thres']
        self.dedup_iou_thres = config['dedup']['iou_thres']
        self.dedup_freq = config['dedup']['time_freq']
            
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
        for _image_file in image_files:
            with open(_image_file, 'rb') as file:
                encoded_str = base64.urlsafe_b64encode(file.read())
                _images_data.append(encoded_str.decode('utf8'))

        if extra_args is not None:
            logger.info(f'{self.model_name}不支持extra_args, 忽略')

        payload = {
            "task_tag": "hat_detect",
            "image_type": "base64",
            "images": _images_data,
        }

        data = self.preprocess(payload)
        data = self.inference(data)
        data = self.postprocess(data)

        return json.dumps(data, indent=4, ensure_ascii=False)

    def preprocess(self, data, **kwargs):
        return data

    @error_handler
    def inference(self, data, *args, **kwargs):
        """推理"""
        return_datas = []
        image_type = data.get("image_type")
        images = data.get("images")
        extra_args = data.get("extra_args")
        filter_size = data.get("filter_size")
        if filter_size is None:
            filter_size = self.filter_size

        sess_hat = self.inference_sessions.get('hat')
        sess_person = self.inference_sessions.get('person')

        if self.platform == 'ONNX':
            input_name = sess_hat.get_inputs()[0].name
            label_name = [i.name for i in sess_hat.get_outputs()]

            input_name0 = sess_person.get_inputs()[0].name
            label_name0 = [i.name for i in sess_person.get_outputs()]

        confidence = self.conf
        iou_thre = self.iou
        filter_size = self.filter_size
        do_dedup = self.do_dedup
        do_freq = self.dedup_freq

        if extra_args:
            for model_param in extra_args:
                model = model_param.get("model")
                if model == self.model_name:
                    param = model_param.get('param')
                    confidence = param.get('conf')
                    iou_thre = param.get('iou')
                    filter_size2 = param.get('filter_size')
                    do_dedup = param.get('do_dedup')
                    do_freq = param.get('time_freq')
                    if confidence is None:
                        confidence = self.conf
                    if iou_thre is None:
                        iou_thre = self.iou
                    if ( filter_size2 is not None ) and ( filter_size != filter_size2 ):
                        filter_size = filter_size2

        if image_type == "base64":
            for i, base64_str in enumerate(images):
                img0 = self.base64_to_cv2(base64_str)
                img = self.prepare_input(img0, swapRB=False)
                
                if self.platform == 'ONNX':
                    output = sess_hat.run(label_name, {input_name: img}, **kwargs)[0]
                    output0 = sess_person.run(label_name, {input_name: img}, **kwargs)[0]
                elif self.platform == 'ASCEND':
                    output = sess_hat.execute([img])[0]
                    output0 = sess_person.execute([img])[0]
                    
                return_datas.append(["img" + str(i + 1), img0, output, output0])

                # return_datas.append(["img" + str(i + 1), img0.shape, img, img0])

        else:
            result = {}
            result["code"] = 400
            result["message"] = f"'model': '{image_type}'"
            result["time"] = int(time.time() * 1000)
            result["data"] = []
            return result

        return return_datas, (confidence, iou_thre, filter_size, do_dedup, do_freq)



    @error_handler
    def postprocess(self, data, *args, **kwargs):
        """后处理"""

        if isinstance(data, dict) and data['code'] == 400:
            return data

        finish_datas = {"code": 200, "message": "", "time": 0, "data": []}

        if data:
            data, param = data
            confidence, iou_thre, filter_size, do_dedup, do_freq = param

            for i, img_data in enumerate(data):
                img_tag, img_raw, preds, preds_person = img_data
                box_out, score_out, class_out = self.filter_by_size(self.process_output(preds, confidence, iou_thre), filter_size=filter_size)
                box_out_person, score_out_person, class_out_person = self.filter_by_size(self.process_output(preds_person, confidence, iou_thre),
                                                                    filter_size=filter_size)

                box_out0 = []
                score_out0 = []
                class_out0=[]
                for i in range(len(class_out_person)):
                    if class_out_person[i] == 0:
                        class_out0.append(class_out_person[i])
                        score_out0.append(score_out_person[i])
                        box_out0.append(box_out_person[i])

                defect_data = []
                if len(box_out) > 0 and len(box_out0) > 0:
                    for idx in range(len(box_out)):
                        if class_out[idx] == 1:

                            if self.image_deduplication(img_raw, list(map(int, box_out[idx])), template_thres=self.dedup_template_thres, iou_thres=self.dedup_iou_thres, do_dedup=do_dedup,
                                                        freq=do_freq):
                                continue

                            conf_hat = int(score_out[idx] * 100)
                            xyxy = box_out[idx]

                            ious = self.compute_iou(xyxy, np.array(box_out0))
                            if np.max(ious) > 0:
                                cls = self.classes[class_out[idx]]

                                xyxy = box_out[idx]
                                defect_data.append({"confidence": conf_hat,
                                                    "defect_desc": "场站有人员未穿戴安全帽,请及时警告!",
                                                    "defect_name": "no_helmet",
                                                    "class": cls,
                                                    "extra_info": {},
                                                    "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                                                    "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                                                    })
                finish_datas["data"].append({"image_tag": img_tag, "defect_data": defect_data})

        finish_datas["time"] = int(round(time.time() * 1000))

        return finish_datas


if __name__ == '__main__':
    input_images = [os.path.join(_cur_dir_,'test_case/0.jpg'),os.path.join(_cur_dir_,'test_case/1.jpg'),os.path.join(_cur_dir_,'test_case/2.jpg')]

    obj = HatDetectHatHandler(platform='ONNX')
    
    results = obj.run_inference(input_images)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
