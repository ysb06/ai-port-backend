import logging
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

# AdamW와 Cosine Anealing을 같이 써보고 Test Error를 다음 링크와 비교
# https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0

logging.basicConfig(
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class Path:
    def __init__(
            self, 
            data_paths: Dict[str, str], 
            save_paths: Dict[str, str],
            trainee_name: str
        ) -> None:
        # Raw
        self.data_path_raw = data_paths
        self.save_path_raw = save_paths
        self.trainee_name = trainee_name

        # Data
        self.train_data_root = data_paths['train_root_dir']
        self.test_data_root = data_paths['test_root_dir']
        self.train_data = os.path.join(data_paths['train_root_dir'], data_paths['training_data_file'])
        self.test_data = os.path.join(data_paths['test_root_dir'], data_paths['test_data_file'])

        # Save
        self.checkpoint = os.path.join(save_paths['root_dir'], save_paths['checkpoints_dir'], trainee_name)
        self.tensorboard = os.path.join(save_paths['root_dir'], save_paths['tensorboard_dir'], trainee_name)
        self.log = os.path.join(save_paths['root_dir'], save_paths['yaml_dir'], trainee_name)
        
        self.make_all_dir()
    
    def set_sp_label(self, sp_label: str = None):
        if sp_label:
            self.checkpoint = os.path.join(self.checkpoint, sp_label)
            self.tensorboard = os.path.join(self.tensorboard, sp_label)
            self.log = os.path.join(self.log, sp_label)
        else:
            self.checkpoint = os.path.join(self.save_path_raw['root_dir'], self.save_path_raw['checkpoints_dir'], self.trainee_name)
            self.tensorboard = os.path.join(self.save_path_raw['root_dir'], self.save_path_raw['tensorboard_dir'], self.trainee_name)
            self.log = os.path.join(self.save_path_raw['root_dir'], self.save_path_raw['yaml_dir'], self.trainee_name)
        
        self.make_all_dir()
        
    def make_all_dir(self):
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)
            logger.info(f'Folder [{self.checkpoint}] created')
        if not os.path.isdir(self.tensorboard):
            os.makedirs(self.tensorboard, exist_ok=True)
            logger.info(f'Folder [{self.tensorboard}] created')
        if not os.path.isdir(self.log):
            os.makedirs(self.log, exist_ok=True)
            logger.info(f'Folder [{self.log}] created')


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
        
        self.paths = Path(data_paths, save_paths, trainee_name)

        seed_everything(hyperparameters['seed'])

    def train(self):
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

    def create_tensorboard(self):
        now = datetime.now(timezone('Asia/Seoul'))
        current_time = now.strftime("%y%m%d_%H%M%S")
        tensorboard = SummaryWriter(log_dir=os.path.join(self.paths.tensorboard, current_time))

        logger.info(f"Tensorboard initialized [{tensorboard.log_dir}]")
        return tensorboard


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)