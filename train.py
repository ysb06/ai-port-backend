import argparse
import logging
import os
from typing import Dict, Type

import torch
import yaml

import nn.trainer
from nn.trainer import TraineeBase

logging.basicConfig(
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_config(config_file_name: str, config_file_folder: str='./'):
    path = os.path.join(config_file_folder, config_file_name)
    training_config = {}
    with open(path, 'r', encoding='utf8') as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)
        logger.info("config.yaml loaded")

    return training_config


def initialize_folders(save_paths: Dict[str, str]):
    root_path = save_paths["root_dir"]
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
        logger.info(f"{root_path} created")

    for key in save_paths:
        if key != "root_dir":
            target_path = os.path.join(root_path, save_paths[key])

            if not os.path.isdir(target_path):
                os.mkdir(target_path)
                logger.info(f"{target_path} created")


if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    parser.add_argument("--config_file", type=str, default="config.yaml")
    option_args = parser.parse_args()

    logger.info(option_args)

    # Get target device
    logger.info(f"PyTorch version: [{torch.__version__}]")
    if torch.cuda.is_available():   # 무조건 cuda만 사용
        target_device = torch.device("cuda:0")
    else:
        raise Exception("No CUDA Device")
    logger.info(f"Target device: [{target_device}]")

    # Load config file
    config = load_config(option_args.config_file, config_file_folder=option_args.target)
    config["device"] = target_device

    # 필요한 폴더 생성
    initialize_folders(config["save_paths"])
    
    # Training Code
    trainee_class: Type[TraineeBase] = getattr(nn.trainer, config["trainee_type"])
    del config["trainee_type"]

    trainee = trainee_class(**config)
    trainee.train()
    
    logger.info("Process Finished!")
            
