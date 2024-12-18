import os
import cv2
import numpy as np
from loguru import logger

from typing import List, Union

import torch

# Define the specific YOLOv8 detection model class
class YOLOv8_DET:
    def __init__(self, config: dict):
        self.model_path = config['model']['path']
        self.names = {int(k):v for k,v in config['model']['names'].items()}
        self.nc = len(self.names)
        
        self.input_shape = (config['model']['height'],config['model']['width']) # shape: h,w

        self.conf_thres = config['params'].get('conf_thres', 0.5)
        self.iou_thres = config['params'].get('iou_thres', 0.4)

        self.platform = config['platform']
        self.device_id = config['device_id']

        if self.platform == 'ASCEND':
            from acllite.acllite_resource import AclLiteResource
            from acllite.acllite_model import AclLiteModel

            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            self.model = AclLiteModel(self.model_path)
        else:
            from ultralytics import YOLO

            if self.device_id < torch.cuda.device_count():
               self.device = f'cuda:{self.device_id}'
            else:
               self.device = 'cpu'

            self.model=YOLO(self.model_path)
            #self.model.eval()  # Set the model to evaluation mode

        logger.info(f'Model Loaded: {self.model_path}, {self.device}')

    def release(self):
        if self.platform == 'ASCEND':
            del self.model
            del self.resource
        else:
            pass

        logger.info(f'Model Relase: {self.model_path}, {self.device}')
        
    def preprocess_input(self, im):
        if self.platform == 'ASCEND':
            im = np.stack(self.pre_transform([im]))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous

            im = (im / 255).astype(np.float32)  # 0 - 255 to 0.0 - 1.0
            
            return im
        else:
            pass
        
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        from utils import LetterBox

        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.input_shape) #, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def predict(self, input_data):
        """Make predictions based on the input data."""
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        if self.platform == 'ASCEND':
            predictions = self.model.execute([input_data])[0]
            #predictions = np.squeeze(predictions)
            #print(f'1. predictions type:{type(predictions)}, shape: {predictions.shape}')
        else:
            pass

        return predictions

    def postprocess_output(self, predictions, img1, img0, img0_path, format='json'):
        """Process the model's predictions."""
        if self.platform == "ASCEND":
            import ops
            #from results import Results
            """Post-processes predictions and returns a list of Results objects."""
            predictions=torch.tensor(predictions)
            #print(f'2. predictions type: {type(predictions)}, shape: {predictions.shape}')
            pred = ops.non_max_suppression(
                predictions,
                self.conf_thres,
                self.iou_thres,
                #agnostic=self.args.agnostic_nms,
                #max_det=self.args.max_det,
                #classes=self.args.classes,
            )[0]
            
            pred[:, :4] = ops.scale_boxes(img1.shape[2:], pred[:, :4], img0.shape)
            #results = Results(img0, path=img0_path, names=self.model.names, boxes=pred)

            if format == 'json':
                import json

                detections = []
                for det in pred:
                    x_min, y_min, x_max, y_max, confidence, class_id = det
                    detections.append({
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "confidence": float(f'{confidence:.4f}'),
                        "class_id": int(class_id)
                    })
                    
                return json.dumps(detections, indent=4)

            return pred
        else:
            pass

    def run_inference(self, image_file_path):
        if self.platform == 'ASCEND':
            import time
            import cv2

            #t0 = time.time()
            img0=cv2.imread(image_file_path)
            img1 = self.preprocess_input(img0)
            #print(f'preprocessed_data type: {type(img1)}, shape: {img1.shape}')
            #t1 = time.time()
            predictions = self.predict(img1)
            #t2 = time.time()
            results = self.postprocess_output(predictions, img1, img0, image_file_path)
            #t3 = time.time()

            #print(f'INFERENCE TIME: {t1-t0:.4f}, {t2-t1:.4f}, {t3-t2:.4f}')
            return results
        else:
            return self.model.predict(image_file_path,imgsz=self.input_shape)
                       
if __name__ == "__main__":
    import json
    import time
    
    config_path='config/yolov8n-det.json'
    with open(config_path, 'r') as file:
        config = json.load(file) 

    config['platform'] = 'ASCEND'
    config['device_id'] = 0

    model=YOLOv8_DET(config)
    
    # To test preprocessing and infering
    imgfile='data/bus.jpg'
    results = model.run_inference(imgfile)
    #print(results)
    
    # To test postrocessing and infering
    """
    model.platform="ASCEND"

    t0 = time.time()
    image_file_path='data/bus.jpg'
    img0=cv2.imread(image_file_path)
    img1 = model.preprocess_input(img0)
    print(f'preprocessed_data type: {type(img1)}, shape: {img1.shape}')
    t1 = time.time()
    predicted_data = torch.load('det-preds.pt')
    results = model.postprocess_output(predicted_data, img1, img0, image_file_path)
    t2 = time.time()
    
    print(f'TIME: {t1-t0:.4f}, {t2-t1:.4f}')

    
    print('Done')
    """
