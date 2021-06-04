import cv2 as cv
import numpy as np
import torch
from werkzeug.datastructures import FileStorage

from mask_detector.dataset import get_valid_transforms
from mask_detector.model import EfficientBase

model = None

def model_initialize():
    global model
    model = EfficientBase(num_classes=3, pretrained_model_name='efficientnet-b5')
    model.load_state_dict(torch.load(f"./assets/results/checkpoint/MaskEff/best.pth"))
    

def predict(target_raw: FileStorage):
    transform = get_valid_transforms(384)

    data = target_raw.stream.read()
    encoded_img = np.fromstring(data, dtype = np.uint8)
    image_raw = cv.imdecode(encoded_img, cv.IMREAD_COLOR)
    image = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)

    target = transform(image=image)
    outputs = model(target['image'].unsqueeze(0))
    outputs = torch.argmax(outputs, dim=-1)
    
    return outputs.item()


def model_finalize():
    global model
    del model
