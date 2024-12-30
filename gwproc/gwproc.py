import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#import json
from pathlib import Path

#import cv2
import numpy as np
#import torch
from enum import Enum

import os
import json
#import requests
from PIL import Image
from io import BytesIO
from ftplib import FTP, error_perm

from hat.model_handler import HatDetectHatHandler
from intrusion.model_handler import IntrusionDetectIntrusionHandler
from wandering.model_handler import BehaviorDetectWanderingHandler
from lightning_rod_current_meter.model_handler import PointerMeterDetectLightningRodCurrentMeterHandler
from cabinet_meter.model_handler import IndicatorMeterDetectCabinetMeterHandler

from utils.log_config import get_logger

logger = get_logger()

model_classes = { "hat": HatDetectHatHandler, 
                  "intrusion": IntrusionDetectIntrusionHandler,
                  "wandering": BehaviorDetectWanderingHandler, 
                  "lightning_rod_current_meter": PointerMeterDetectLightningRodCurrentMeterHandler,
                  "cabinet_meter": IndicatorMeterDetectCabinetMeterHandler
}   

class GWProc_Result(Enum):
    IMAGE_FAIL=1
    INVALID_PROC=2
    INFER_FAIL=3
    INFER_ZERO_DETECT=4
    INFER_AND_DETECT=5

GWProc_result_dict = {
    GWProc_Result.IMAGE_FAIL:"2001", 
    GWProc_Result.INVALID_PROC:"2002", 
    GWProc_Result.INFER_FAIL:"2002", 
    GWProc_Result.INFER_ZERO_DETECT:"2000", 
    GWProc_Result.INFER_AND_DETECT:"2000"
}

PLATFORM = ['ONNX', 'ASCEND']
    
class GWProc:
    ftp_config = None #shared ftp config
    
    def __init__(self, model_name, platform='ASCEND', device_id=None, ftp=None):
        if ftp is None:
            if self.__class__.ftp_config is None:
                raise ValueError(
                    f"GWProc initialization failed, missing ftp configuration!")
        else:
            self.__class__.ftp_config = ftp
            
        # Validate the model type
        if model_name not in model_classes:
            raise ValueError(
                f"Invalid model type '{model_name}'. Available types: {list(self.model_classes.keys())}")

        if platform not in PLATFORM:
            raise ValueError(
                f"Invalid platform type '{platform}'. Available platforms: {PLATFORM}")

        self.model_instance = model_classes[model_name](platform=platform, device_id=device_id)

    def run_inference(self, input_image, extra_args=None):
        """Run inference using the specific model instance."""
        return self.model_instance.run_inference(input_image, extra_args=extra_args)

    def release(self) -> None:
        self.model_instance.release()

    # 图像通过ftp服务服务器提供，API调用中URL没有ftp头，仅包括服务器内的路径
    @classmethod
    def read_image(cls, path):
        try:
            _connected = False
            _result = False
            _data = None

            # Connect to the FTP server
            ftp = FTP()
            ftp.connect(cls.ftp_config['ip'], cls.ftp_config['port'])
            ftp.login(user=cls.ftp_config['user'],passwd=cls.ftp_config['password'])
            _connected = True

            image_data = BytesIO()
            ftp.retrbinary(f'RETR {path}', image_data.write)
            #ftp.quit()

            image_data.seek(0)
            _data = image_data.read()
            _result = True

        except error_perm as e:
            if str(e).startswith('550'):
                _data = f"File not found on FTP server: {path}"
            else:
                _data = f"Permission error: {e}"
            logger.error(_data)
        except FileNotFoundError:
            _data =f"The specified image file {path} was not found."
            logger.error(_data)
        except Exception as e:
            _data = f"An error occurred: {e}"
            logger.error(_data)
        finally:
            # Ensure the FTP connection is closed
            if _connected:
                ftp.quit()
                
            return _result, _data

    @classmethod
    def result_json(cls, type, result, value="0", desc="正常", conf=0.0, pos=[]):
        assert isinstance(result,GWProc_Result), f'返回结果类型{result}非法'

        json_data = {
            "type": type,
            "value": value,
            "code": GWProc_result_dict[result],
            "resImageUrl": "",
            "pos": pos,
            "conf": f'{conf:.4f}',
            "desc":desc
        }
        
        return json.dumps(json_data, indent=4, ensure_ascii=False)

# Note: 存在循环import, 不要在此进行测试
"""
# Example usage
ftpDict = {
    "ip": "192.168.0.164",
    "port": 2121,
    "user": "gw",
    "password": "gwPasw0rd"
}

if __name__ == "__main__":
    # Load the configuration for the specific task
    import time

    t0 = time.time()
    #model = GWProc(model_name='hat', platform='ONNX')
    #model = GWProc(model_name='intrusion', platform='ONNX')
    #model = GWProc(model_name='wandering', platform='ONNX')
    #model = GWProc(model_name='lightning_rod_current_meter', platform='ONNX')
    #model = GWProc(model_name='cabinet_meter', platform='ONNX')

    #model = GWProc(model_name='hat', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='intrusion', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='wandering', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='lightning_rod_current_meter', platform='ASCEND', device_id=0)
    model = GWProc(model_name='cabinet_meter', platform='ONNX', device_id=0,ftp=ftpDict)
    t1 = time.time()
    
    # Load an image for testing (replace with actual image path)
    #input_images = ['hat/test_case/0.jpg','hat/test_case/1.jpg','hat/test_case/2.jpg']
    #input_images = ['intrusion/test_case/0.jpg','intrusion/test_case/2.jpg','intrusion/test_case/4.jpg']
    #input_images = ['wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-8d4376beeb5048feb0cfb7ed798b6d71.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-27fb1e1bd3844b07a18f7d2aac8bbc71.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-93bfd984746c49e0b02cf33b7575bcb3.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-ef7e24fc0da949f4af6a471209904250.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-f325e0299dbd4c77ae71f5b4ebf7ad38.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-faf3ca75c7014e55ac307d8eddcf8616.png'
    #                ]
    #input_images = ['lightning_rod_current_meter/test_case/lightning_rod_current_meter2.png']
    input_images = ['cabinet_meter/test_case/cabinet_meter_20A.jpg']

    # Run inference
    results = model.run_inference(input_images)
    t2 = time.time()

    print(f'TIME: {t1-t0:.4f}, {t2-t1:.4f}')

    # Process and visualize the results (example)
    print("Inference Results:", results)

    model.release()

    print("Done!")
"""