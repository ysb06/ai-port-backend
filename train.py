import argparse
import logging
import os
from typing import Type
import importlib

import torch
import yaml
from aiport_server.trainee import TraineeBase


logging.basicConfig(
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_config(config_file_name: str, config_file_folder: str='./'):
    path = os.path.join('aiport_server', config_file_folder, config_file_name)
    training_config = {}
    with open(path, 'r', encoding='utf8') as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)
        logger.info(f"{path} loaded")

    return training_config


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
    
    # Training Code
    trainee_module = importlib.import_module(f"aiport_server.{option_args.target}.trainer")
    trainee_class: Type[TraineeBase] = getattr(trainee_module, config["trainee_type"])
    
    del config["trainee_type"]

    trainee = trainee_class(**config)
    trainee.train()
    
    logger.info("Process Finished!")
