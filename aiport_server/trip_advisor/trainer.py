import logging
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from aiport_server.trainee import TraineeBase
from aiport_server.trip_advisor.training.dataset import OpenVocabDSTDataset, collate_TRADE_batch
from aiport_server.trip_advisor.training.klue_dst_loader import load_open_vocab_data

logger = logging.getLogger(__name__)


class DSTTrainee(TraineeBase):
    def __init__(self, trainee_name: str, data_paths: Dict[str, str], save_paths: Dict[str, str], hyperparameters: Dict, device: torch.device) -> None:
        super().__init__(trainee_name, data_paths, save_paths, hyperparameters, device)
        self.paths.ontology_data = os.path.join(self.paths.data_path_raw['train_root_dir'], self.paths.data_path_raw['ontology_data_file'])

    def on_train_begin(self):
        data_folds = load_open_vocab_data(self.paths.train_data, self.paths.ontology_data, self.hyperparameters)
        dataset_folds = [(OpenVocabDSTDataset(train_data), OpenVocabDSTDataset(valid_data)) for train_data, valid_data in data_folds]

        for fold in dataset_folds:
            train_loader = DataLoader(fold[0], batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_TRADE_batch)
            valid_loader = DataLoader(fold[1], batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_TRADE_batch)

            epochs = 10

            for epoch in range(epochs):
                for batch in train_loader:
                    break

        

    def on_train(self):
        print('OK')
