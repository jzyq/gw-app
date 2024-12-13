import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import time

from utils.log_config import get_logger
logger = get_logger()

from utils.commons import ImageHandler, dec_timer
from core.exceptions import error_handler

_cur_dir_=os.path.dirname(__file__)

class IntrusionDetectIntrusionHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()
        self.conf = 0.5
        self.iou = 0.5
        self.new_shape = [1920, 1920]
        self.model_name = 'intrusion'
        self.classes = ['person']
        self.kpt_shape = [17, 3]  # 17个关键点 (x,y,是否可见)
        self.filter_size = 0
        self.do_dedup = 1
        self.time_freq = 'd'
        self.areas = [
            {
                "area_id": 0,
                "points": [[0, 0], [999999, 0], [999999, 999999], [0, 999999]]
            }
        ]
        
        self.platform = platform
        
        if self.platform == 'ONNX':
            import onnxruntime as ort

            sess = ort.InferenceSession(os.path.join(_cur_dir_, 'models/intrusion.onnx'),providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

            logger.info(f'Model {self.model_name} Loaded')

        elif self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device_id = device_id
            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            sess = AclLiteModel(os.path.join(_cur_dir_, 'models/intrusion.om'))

            logger.info(f'Model {self.model_name} Loaded on {self.device}')
        else:
            # TO DO: should not be here, should report an error
            pass
        
        self.inference_sessions = {'intrusion': sess}

    def release(self):
        if self.platform == 'ASCEND':
            for _sess in self.inference_sessions:
                del _sess
            del self.resource
            
            logger.info(f'Model {self.model_name} Relased on {self.device}')

        else:
            logger.info(f'Model {self.model_name} Relased')


    def run_inference(self, image_files):
        _images_data = []
        for _image_file in image_files:
            with open(_image_file, 'rb') as file:
                encoded_str = base64.urlsafe_b64encode(file.read())
                _images_data.append(encoded_str.decode('utf8'))

        payload = {
            "task_tag": "intrusion_detect",
            "image_type": "base64",
            "images": _images_data,
            "extra_args": [
                {
                    "model": "intrusion",
                    'param': {
                        "do_dedup": 0,
                        "time_freq": 'd',
                        "filter_size": 0,
                        "areas": [
                            {"area_id": 1, "points": [[196, 139], [447, 79], [1155, 550], [764, 650]]},
                            {"area_id": 2, "points": [[196, 139], [447, 79], [1155, 550], [764, 650]]},
                        ],
                        'conf': 0.5, 'iou': 0.5
                    }
                }
            ]
        }

        data = self.preprocess(payload)
        data = self.inference(data)
        data = self.postprocess(data)
        
        return json.dumps(data, indent=4, ensure_ascii=False)

    def filter_by_size(self, output, filter_size=25, reverse=None):
        # obj_list: output from self.plate_postprocessing [box(4), score(1), landmark(8), class(0)]
        box, score, cls = output, output[:, 4], output[:, 5]
        idx = []
        if len(box) > 0:
            for i, z in enumerate(zip(box.tolist(), score.tolist(), cls.tolist())):
                if reverse:
                    if abs(z[0][0] - z[0][2]) <= filter_size and abs(z[0][1] - z[0][3]) <= filter_size:
                        idx.append(i)
                else:
                    if abs(z[0][0] - z[0][2]) >= filter_size and abs(z[0][1] - z[0][3]) >= filter_size:
                        idx.append(i)
        return box[idx, :], score[idx,], cls[idx,]

    def preprocess(self, data, **kwargs):
        return data

    @error_handler
    def inference(self, data, *args, **kwargs):
        """推理"""
        return_datas = []
        areas = []
        image_type = data.get("image_type")
        images = data.get("images")
        filter_size = data.get("filter_size")
        extra_args = data.get("extra_args")
        if filter_size is None:
            filter_size = self.filter_size
            
        sess = self.inference_sessions.get('intrusion')

        if self.platform == 'ONNX':
            input_name = sess.get_inputs()[0].name
            label_name = [i.name for i in sess.get_outputs()]

        if extra_args:
            for model_param in extra_args:
                model = model_param.get("model")
                if model == self.model_name:
                    param = model_param.get('param')
                    confidence = param.get('conf')
                    iou_thre = param.get('iou')
                    areas = param.get('areas')
                    filter_size2 = param.get('filter_size')
                    do_dedup = param.get('do_dedup')
                    time_freq = param.get('time_freq')
                    if confidence is None:
                        confidence = self.conf
                    if iou_thre is None:
                        iou_thre = self.iou
                    if ( filter_size2 is not None ) and ( filter_size != filter_size2 ):
                        filter_size = filter_size2
                    if not areas:
                        areas = self.areas

        if image_type == "base64":
            for i, base64_str in enumerate(images):
                img0 = self.base64_to_cv2(base64_str)
                img = self.prepare_input(img0, swapRB=False)
                
                if self.platform == 'ONNX':
                    #output0 = sess.run(label_name, {input_name: img}, **kwargs)
                    output = sess.run(label_name, {input_name: img}, **kwargs)[0]
                elif self.platform == 'ASCEND':
                    output = sess.execute(img)[0]
              
                return_datas.append(["img" + str(i + 1), img0, output])

        else:
            result = {}
            result["code"] = 400
            result["message"] = f"'model': '{image_type}'"
            result["time"] = int(time.time() * 1000)
            result["data"] = []
            return result

        return return_datas, (confidence, iou_thre, areas, filter_size, do_dedup, time_freq)

    @error_handler
    def postprocess(self, data, *args, **kwargs):
        """后处理"""

        if isinstance(data, dict) and data['code'] == 400:
            return data

        finish_datas = {"code": 200, "message": "", "time": 0, "data": []}

        if data:
            data, param = data
            confidence, iou_thre, areas, filter_size, do_dedup, time_freq = param

            for i, img_data in enumerate(data):
                img_tag, img_raw, preds = img_data
                preds, _, _ = self.filter_by_size(self.process_box_output(preds, confidence, iou_thre)[0],
                                                  filter_size=filter_size)
                box_out, score_out, class_out = preds[:, :4], preds[:, 4], preds[:, 5]
                box_out = self.scale_boxes(self.new_shape, box_out, img_raw.shape)
                pred_kpts = preds[:, 6:].reshape([len(preds)] + self.kpt_shape)
                pred_kpts = self.scale_coords(self.new_shape, pred_kpts, img_raw.shape)

                defect_data = []
                for area in areas:
                    area_id = area.get('area_id')
                    points = area.get('points')

                    person_counts = 0
                    for i, pred in enumerate(preds):
                        # 增加重复图片检测，识别为True直接跳过
                        if self.image_deduplication(img_raw, list(map(int, box_out[i])), do_dedup=do_dedup,
                                                    freq=time_freq):
                            continue
                        pred_kpt = pred_kpts[i]
                        # 左下和右下(最后两个关键点)任意一个点在区域内
                        xyz1 = pred_kpt[-1]
                        xyz2 = pred_kpt[-2]
                        if xyz1[-1] > 0.5 and self.is_in_poly(xyz1[:2], points):
                            person_counts += 1
                            continue
                        if xyz2[-1] > 0.5 and self.is_in_poly(xyz2[:2], points):
                            person_counts += 1
                    if person_counts:
                        conf = int(pred[4] * 100)
                        defect_data.append({"defect_name": "intrusion",
                                            'defect_desc': "场站有人员入侵，请及时警告!",
                                            "confidence": conf,
                                            "class": self.model_name,
                                            "extra_info": {"area_id": area_id,
                                                           "person_counts": person_counts}
                                            })
                finish_datas["data"].append({"image_tag": img_tag, "defect_data": defect_data})

        finish_datas["time"] = int(round(time.time() * 1000))

        return finish_datas


if __name__ == '__main__':
    input_images = [os.path.join(_cur_dir_,'test_case/0.jpg'),os.path.join(_cur_dir_,'test_case/2.jpg'),os.path.join(_cur_dir_,'test_case/4.jpg')]

    obj = IntrusionDetectIntrusionHandler(platform='ONNX')
    
    results = obj.run_inference(input_images)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
