import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from mask_detector.dataset import get_dataset_folds
from pytz import timezone
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nn.utils import seed_everything

# AdamW와 Cosine Anealing을 같이 써보고 Test Error를 다음 링크와 비교
# https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0

logger = logging.getLogger(__name__)

@dataclass
class SavePath:
    checkpoint: str
    tensorboard: str
    log: str

class TraineeBase:
    def __init__(
        self,
        trainee_name: str,
        data_paths: Dict[str, str],
        save_paths: Dict[str, str],
        hyperparameters: Dict,
        device: torch.device,
    ) -> None:
        logger.info("Initializing trainee...")

        self.name = trainee_name
        self.device = device
        self.hyperparameters = hyperparameters
        
        self.paths = data_paths
        checkpoint_path = os.path.join(save_paths['root_dir'], save_paths['checkpoints_dir'], self.name)
        tensorboard_path = os.path.join(save_paths['root_dir'], save_paths['tensorboard_dir'], self.name)
        log_path = os.path.join(save_paths['root_dir'], save_paths['yaml_dir'], self.name)

        self.save_paths = SavePath(
            checkpoint_path, tensorboard_path, log_path
        )

        now = datetime.now(timezone('Asia/Seoul'))
        current_time = now.strftime("%y%m%d_%H%M%S")
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.save_paths.tensorboard, current_time))
        logger.info(f"Tensorboard initialized [{self.tensorboard.log_dir}]")

        seed_everything(hyperparameters['seed'])

    def train(self):
        print()
        self.on_train_begin()
        logger.info(f"{self.name}({type(self).__name__}) Traininig...")
        self.on_train()
        logger.info("Traininig finished")
        self.on_train_end()

    def on_train_begin(self):
        pass

    def on_train(self):
        raise NotImplementedError("on_train must be defined")

    def on_train_end(self):
        pass


class MaskModelTrainee(TraineeBase):
    def on_train_begin(self):
        dataset_folds = get_dataset_folds(
            self.paths['train_root_dir'],
            self.paths['training_data_file'],
            **self.hyperparameters['dataloader']['args']
        )

        print(dataset_folds)
    
    def on_train(self):
        pass
