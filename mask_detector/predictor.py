from mask_detector.model import EfficientBase
from werkzeug.datastructures import FileStorage
import cv2 as cv
from mask_detector.dataset import get_valid_transforms
import matplotlib.pyplot as plt

model = None

def initialize():
    global model
    model = EfficientBase(num_classes=3, pretrained_model_name='efficientnet-b5')
    

def predict(target_raw: FileStorage):
    target = cv.imread(target_raw)
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
    transform = get_valid_transforms(384)
    target = transform(image=target)


def finalize():
    global model
    del model