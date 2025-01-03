import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import time
import yaml

from utils.log_config import get_logger
logger = get_logger()

from utils.commons import ImageHandler, dec_timer
from core.exceptions import error_handler

_cur_dir_=os.path.dirname(__file__)

class IntrusionDetectIntrusionHandler(ImageHandler):
    def __init__(self, platform='ASCEND', device_id=None):
        super().__init__()

        self.model_name = 'intrusion'
        self.new_shape = [1920, 1920]
        
        # 设置缺省侵入控制区域为全图像, 配置文件和接口调用中若有明确定义将覆盖缺省定义
        self.areas=[{"area_id": 0, "points": [[0,0], [self.new_shape[1]-1, self.new_shape[0]-1]]}]

        self.read_config()
        
        self.classes = ['person']
        self.kpt_shape = [17, 3]  # 17个关键点 (x,y,是否可见)
        
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

    def read_config(self):
        # 读取配置文件
        _config_file = os.path.join(_cur_dir_, 'config.yaml')
        with open(_config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 访问参数
        self.conf = config['detect']['conf_thres']
        self.iou = config['detect']['iou_thres']
        self.kpt_thres = config['detect']['kpt_thres']
        self.filter_size = config['detect']['filter_size']
        
        self.do_dedup = (config['dedup']['enable']==1)
        self.dedup_template_thres = config['dedup']['template_thres']
        self.dedup_iou_thres = config['dedup']['iou_thres']
        self.dedup_freq = config['dedup']['time_freq']
            
        if config['areas'] is not None and config['areas'].items() is not None and len(config['areas'].items()) > 0:
            _areas = []
            
            for _key, _value in config['areas'].items():
                # if len(_value) == 2: # (left,top) - (right,bottom)
                #     _areas.append({"area_id": int(_key), "points": self.__class__.points2box(_value[0], _value[1])})
                # else:
                #     _areas.append({"area_id": int(_key), "points": _value})
                _areas.append({"area_id": int(_key), "points": _value})
            
            if len(_areas) != 0:
                self.areas=_areas
        """
        [
            {
                "area_id": 0,
                "points": [[0, 0], [999999, 0], [999999, 999999], [0, 999999]]
            }
        ]
        """

    def release(self):
        if self.platform == 'ASCEND':
            for _sess in self.inference_sessions:
                del _sess
            del self.resource
            
            logger.info(f'Model {self.model_name} Relased on {self.device}')

        else:
            logger.info(f'Model {self.model_name} Relased')

    def run_inference(self, images_data, extra_args=None):
        _areas=[]
        if extra_args is None or extra_args.get('pos') is None:
            logger.info(f'{self.model_name}收到无效extra_args, 忽略')
        else:
            _pos_items = extra_args.get('pos')
                    
            for _i,_pos in enumerate(_pos_items):
                if _pos.get('areas') is not None:
                    _point_list = _pos.get('areas')
                    
                    _area_points=[]
                    for _p in _point_list:
                        _area_points.append([_p["x"], _p["y"]])
            
                _areas.append({"area_id": (_i+1), "points": _area_points})
                            
        if len(_areas) == 0: # 没有extra_args将会采用初始化中的areas
            _areas=self.areas
            
        #检查每个区域，若为2点则转为4点坐标
        _payload_areas = []
        for _area in _areas:
            if len(_area['points']) == 2:
                _payload_areas.append({"area_id": _area['area_id'], "points": self.__class__.points2box(_area['points'][0], _area['points'][1])})
            else:
                _payload_areas.append(_area)

        payload = {
            "task_tag": "intrusion_detect",
            "image_type": "base64",
            "images": images_data,
            "extra_args": [
                {
                    "model": self.model_name,
                    'param': {
                        "areas": _payload_areas
                    }
                }
            ]
        }

        try:
            try:
                from ..gwproc import GWProc,GWProc_Result
            except:
                from gwproc import GWProc,GWProc_Result

            data = self.preprocess(payload)
            data = self.inference(data)
            data = self.postprocess(data)

            #当前仅支持单输入单输出, 仅取[0]号元素
            _defects = data['data'][0]['defect_data']
            if len(_defects) == 0:
                return GWProc.result_json(self.model_name, GWProc_Result.INFER_ZERO_DETECT)
            else:
                _max_conf = 0.0
                _pos = []
                for _def in _defects:
                    _aid = _def['extra_info']['area_id']
                    #_defect_area=None
                    for _area in _areas:
                        if _area['area_id']== _aid:
                            _defect_points=_area['points']
                            break
                    if _defect_points is None:
                        raise ValueError(f'找到返回区域代码{_aid}对应监控区域, 要求监控区域为{_areas} ')
                    else:
                        _defect_areas_points=[]
                        for _p in _defect_points:
                            _defect_areas_points.append({"x":_p[0],"y":_p[1]})
                        _defect_areas={"areas": _defect_areas_points}

                    if _def['confidence'] > _max_conf:
                        _max_conf=_def['confidence']
                        _desc = _def['defect_desc']
                        _pos=[_defect_areas]+_pos
                    else:
                        _pos=_pos+[_defect_areas]

                return GWProc.result_json(self.model_name, GWProc_Result.INFER_AND_DETECT, value="1", desc=_desc, conf=_max_conf/100.0, pos=_pos)
        except Exception as e:
            _data = f"{self.model_name}: An error occurred: {e}"
            logger.error(_data)
            return GWProc.result_json(self.model_name, GWProc_Result.INFER_FAIL, _data)
        # data = self.preprocess(payload)
        # data = self.inference(data)
        # data = self.postprocess(data)
        
        # return json.dumps(data, indent=4, ensure_ascii=False)

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
                    kpt_thres = param.get('kpt_thres')
                    areas = param.get('areas')
                    filter_size2 = param.get('filter_size')
                    do_dedup = param.get('do_dedup')
                    time_freq = param.get('time_freq')
                    if confidence is None:
                        confidence = self.conf
                    if iou_thre is None:
                        iou_thre = self.iou
                    #if kpt_thres is None:
                    #    kpt_thres = self.kpt_thres
                    if ( filter_size2 is not None ) and ( filter_size != filter_size2 ):
                        filter_size = filter_size2
                    if do_dedup is None:
                        do_dedup = self.do_dedup
                    if time_freq is None:
                        time_freq = self.dedup_freq
                    if not areas:
                        areas = self.areas
        else:
            confidence = self.conf
            iou_thre = self.iou
            filter_size = self.filter_size
            do_dedup = self.do_dedup
            time_freq = self.dedup_freq
            areas = self.areas
            #kpt_thres = self.kpt_thres

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
                        """
                        # 增加重复图片检测，识别为True直接跳过
                        # TODO: 这里有个问题，检测重复目标前没有先检测该目标是否在指定区域内，区域外目标也会被记录，在检测另外区域时此目标会被认为重复目标而忽略
                        #       此问题会导致同一画面，多个监控区域都有目标情况下， 只返回第一个发现，其他发现会被认为是重复目标
                        if self.image_deduplication(img_raw, list(map(int, box_out[i])), 
                                                    template_thres=self.dedup_template_thres, iou_thres=self.dedup_iou_thres, 
                                                    do_dedup=do_dedup, freq=time_freq):
                            continue
                        pred_kpt = pred_kpts[i]
                        # 左下和右下(最后两个关键点)任意一个点在区域内
                        xyz1 = pred_kpt[-1]
                        xyz2 = pred_kpt[-2]
                        if xyz1[-1] > self.kpt_thres and self.is_in_poly(xyz1[:2], points):
                            person_counts += 1
                            continue
                        if xyz2[-1] > self.kpt_thres and self.is_in_poly(xyz2[:2], points):
                            person_counts += 1
                        """
                        pred_kpt = pred_kpts[i]
                        # 左下和右下(最后两个关键点)任意一个点在区域内
                        xyz1 = pred_kpt[-1]
                        xyz2 = pred_kpt[-2]
                        if (xyz1[-1] > self.kpt_thres and self.is_in_poly(xyz1[:2], points)) or (xyz2[-1] > self.kpt_thres and self.is_in_poly(xyz2[:2], points)):
                            if self.image_deduplication(img_raw, list(map(int, box_out[i])), 
                                                        template_thres=self.dedup_template_thres, iou_thres=self.dedup_iou_thres, 
                                                        do_dedup=do_dedup, freq=time_freq):
                                continue
                            person_counts += 1
                            continue
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

json_str = '''{
    "requestHostIp": "10.11.120.39",
    "requestHostPort": "8766",
    "requestId": "1234abcd-1a2b-4444-3c4d-1a2b3c4d5e6f",
    "objectList": [
        {
            "objectId": "123-1",
            "typeList": [
                "wcaqm",
                "wcgz",
                "hxq_gjbs",
                "xy"
            ],
            "imageUrlList": [
                "wcaqm1.jpg"
            ],
            "imageNormalUrlPath": "",
            "pos": [{
                "areas": [
                    {"x": 0, "y": 0},
                    {"x": 100, "y": 100}
                ]
            }]
        }
    ]
}'''

# Note: ftp服务器在首次实例化GWProc时设置，无法在此单独进行测试

"""
if __name__ == '__main__':
    input_images = [os.path.join(_cur_dir_,'test_case/0.jpg'),os.path.join(_cur_dir_,'test_case/2.jpg'),os.path.join(_cur_dir_,'test_case/4.jpg')]

    obj = IntrusionDetectIntrusionHandler(platform='ONNX')
    
    #extra_args= json.loads(json_str)
    results = obj.run_inference(input_images, extra_args=None)
    print("Inference Results:", results)

    obj.release()
    print("Done!")
"""