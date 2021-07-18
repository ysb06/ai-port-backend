from typing import Tuple
import cv2 as cv
import numpy as np
import torch
from aiport_server.mask_detector.model import EfficientBase
from werkzeug.datastructures import FileStorage
import albumentations as A
from albumentations.augmentations import SmallestMaxSize
from albumentations.pytorch import ToTensorV2

model = None

def model_initialize():
    global model
    model = EfficientBase(num_classes=3, pretrained_model_name='efficientnet-b5')
    model.load_state_dict(torch.load(f"./assets/results/checkpoint/MaskEff/best.pth", map_location=torch.device('cpu')))
    model.eval()
    

def predict(target_raw: FileStorage):
    transform = _get_valid_transforms(384)

    data = target_raw.stream.read()
    encoded_img = np.fromstring(data, dtype = np.uint8)
    image_raw = cv.imdecode(encoded_img, cv.IMREAD_COLOR)
    image = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)

    target = transform(image=image)
    with torch.no_grad():
        outputs = model(target['image'].unsqueeze(0))
        # print(outputs)
        outputs = torch.argmax(outputs, dim=-1)
    
    return outputs.item()


def model_finalize():
    global model
    del model


def _get_valid_transforms(
        image_side_length: int,
        mean: Tuple[float, float, float] = (0.548, 0.504, 0.479), 
        std: Tuple[float, float, float] = (0.237, 0.247, 0.246)
    ):
    val_transforms = A.Compose([
        SmallestMaxSize(max_size=image_side_length, always_apply=True),
        A.CenterCrop(image_side_length, image_side_length, always_apply=True),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return val_transforms