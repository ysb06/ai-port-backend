import os
from dataclasses import dataclass
from enum import Enum
from glob import glob
from random import Random
from typing import Any, Dict, List, Tuple
import logging

import cv2 as cv
import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.frame import DataFrame
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import SmallestMaxSize
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Gender(Enum):
    Male = 0
    Female = 1

class MaskState(Enum):
    Good = 0
    Bad = 1
    NoMask = 2

@dataclass
class MaskImage():
    image_raw: Any
    gender: Gender
    mask_state: MaskState

# 클래스 명이 Person인 것은 원래는 사진 하나마다 다른 사람으로 처리할 생각이었음
# 지금은 한 사람이 아닌 Picture 하나라고 보면 됨


class MaskedFaceDataset(Dataset):
    def __init__(self, data: List[MaskImage]) -> None:
        self.data = data
        self.transform: A.Compose = None

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple:
        target: MaskImage = self.data[index]

        source = self.transform(image=target.image_raw)["image"]
        label = target.mask_state.value

        return source, label


def get_dataloader_folds(      # 다른 모델을 생성할 경우 dataset타입을 인수로 받는 거도 고려
        data_root_path: str,
        label_file_path: str,
        kfold_n_splits: int,
        image_side_length: int,
        cross_validation: bool,
        train_loader_args: Dict,
        valid_loader_args: Dict,
        seed: int = None,
    ) -> List[Tuple[DataLoader, DataLoader]]:
    mask_label_path = label_file_path
    label_raw = pd.read_csv(mask_label_path)
    age_group = np.where(label_raw['age'] < 40, 'young', 'old')
    label_raw['group'] = label_raw['gender'].str.cat(age_group, sep='_')
    
    dataloader_folds = []
    train_transform = __get_basic_train_transforms(image_side_length)
    valid_transform = __get_valid_transforms(image_side_length)

    skf = StratifiedKFold(n_splits=kfold_n_splits, shuffle=seed != None, random_state=seed)
    for index, (train_idx, valid_idx) in enumerate(skf.split(X=label_raw, y=label_raw['group'])):
        logger.info(f'Generating dataset fold {index}...')
        train_data, valid_data = label_raw.iloc[train_idx], label_raw.iloc[valid_idx]
        # StratifiedKFold에 따라 나뉜 인덱스에 따라 train_data, valid_data로 DataFrame 재생성

        train_dataset: MaskedFaceDataset = __generate_dataset(train_data, data_root_path, seed)
        train_dataset.transform = train_transform
        valid_dataset: MaskedFaceDataset = __generate_dataset(valid_data, data_root_path, seed)
        valid_dataset.transform = valid_transform
        # 데이터셋 생성
        # Transform을 각 데이터셋에 맞게 적용

        dataset_group = (
            DataLoader(train_dataset, **train_loader_args),
            DataLoader(valid_dataset, **valid_loader_args),
        )
        dataloader_folds.append(dataset_group)

        if cross_validation is not True:
            break

    return dataloader_folds

def __generate_dataset(raw: DataFrame, root_path: str, seed: int):  
    data: List[MaskImage] = []
    rand = Random(seed)

    for row in tqdm(raw.iloc, total=len(raw)):
        image_dir = os.path.join(root_path, 'images', row['path'])
        good_mask_images_paths = glob(os.path.join(image_dir, "mask*"))
        bad_mask_image_path = glob(os.path.join(image_dir, "incorrect_mask.*"))[0]
        no_mask_image_path = glob(os.path.join(image_dir, "normal.*"))[0]

        good_mask_image_raw = cv.imread(good_mask_images_paths[rand.randrange(0, 5)])
        good_mask_image_raw = cv.cvtColor(good_mask_image_raw, cv.COLOR_BGR2RGB)
        # Good 이미지는 수가 많으므로 데이터 불균형을 맞춰주기 위해 5개 중 1나만 선택
        # 일종의 Under Sampling, 그러나 fold 별로 다른 이미지
        bad_mask_image_raw = cv.imread(bad_mask_image_path)
        bad_mask_image_raw = cv.cvtColor(bad_mask_image_raw, cv.COLOR_BGR2RGB)
        no_mask_image_raw = cv.imread(no_mask_image_path)
        no_mask_image_raw = cv.cvtColor(no_mask_image_raw, cv.COLOR_BGR2RGB)

        data.append(MaskImage(
            good_mask_image_raw,
            Gender.Male if row['gender'] == 'male' else Gender.Female,
            MaskState.Good
        ))

        data.append(MaskImage(
            bad_mask_image_raw,
            Gender.Male if row['gender'] == 'male' else Gender.Female,
            MaskState.Bad
        ))

        data.append(MaskImage(
            no_mask_image_raw,
            Gender.Male if row['gender'] == 'male' else Gender.Female,
            MaskState.NoMask
        ))

    return MaskedFaceDataset(data)


def __get_basic_train_transforms(
        image_side_length: int, 
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):  
    # https://github.com/lukemelas/EfficientNet-PyTorch
    # Mean, Std는 위 링크를 참조했으며 Efficientnet Pretrained Model의 설정 값으로 추정
    train_transforms = A.Compose([
        SmallestMaxSize(max_size=image_side_length, always_apply=True),
        A.CenterCrop(image_side_length, image_side_length, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=15),
        A.HueSaturationValue(hue_shift_limit=0.2,
                             sat_shift_limit=0.2, 
                             val_shift_limit=0.2, 
                             p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return train_transforms


def __get_valid_transforms(
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