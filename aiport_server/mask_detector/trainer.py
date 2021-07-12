import logging

import numpy as np
import torch
from aiport_server.mask_detector.dataset import get_dataloader_folds
from aiport_server.mask_detector.loss import FocalLoss
from aiport_server.mask_detector.model import EfficientBase
from aiport_server.trainee import TraineeBase
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

logger = logging.getLogger(__name__)

# AdamW와 Cosine Anealing을 같이 써보고 Test Error를 다음 링크와 비교
# https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0

class MaskModelTrainee(TraineeBase):
    def on_train_begin(self):
        self.logging_inteval = 10

        self.dataloader_folds = get_dataloader_folds(
            data_root_path=self.paths.train_data_root,
            label_file_path=self.paths.train_data,
            **self.hyperparameters['dataloader']['args']
        )   # About 4 Gb (800Mb * 5)

        self.model = EfficientBase(**self.hyperparameters['model']['args'])
        self.model.to(self.device)
        self.criterion = FocalLoss(**self.hyperparameters['loss']['args'])
        self.optimizer = AdamW(self.model.parameters(), **self.hyperparameters['optimizer']['args'])
        self.scheduler = CosineAnnealingLR(self.optimizer, **self.hyperparameters['scheduler']['args'])
    
    def on_train(self):
        for fold, loaders in enumerate(self.dataloader_folds):
            target_fold = self.hyperparameters['dataloader']['target_fold']    
            if target_fold >= 0 and target_fold != fold:
                continue 
            
            if target_fold != -1:
                self.paths.set_sp_label(fold)
            self.tensorboard = self.create_tensorboard()

            train_loader = loaders[0]
            valid_loader = loaders[1]

            best_valid_accuracy = 0
            best_valid_accuracy_epoch = 0
            best_valid_loss = np.inf

            epochs = self.hyperparameters['epochs']
            for current_epoch in range(epochs):
                logger.info(f"Start epoch {current_epoch + 1}")
                self.model.train()

                loss_value = 0
                tp_matches = 0
                for step, (source_batch, label_batch) in enumerate(train_loader):
                    source_batch: Tensor = source_batch.to(self.device)
                    label_batch: Tensor = label_batch.to(self.device, dtype=torch.long)

                    # Initialize Gradient
                    self.optimizer.zero_grad()

                    outputs = self.model(source_batch)
                    predicts = torch.argmax(outputs, dim=-1)
                    loss: Tensor = self.criterion(outputs, label_batch)

                    loss.backward()
                    self.optimizer.step()

                    # Examination
                    loss_value += loss.item()
                    tp_matches += (predicts == label_batch).sum().item()
                    if (step + 1) % self.logging_inteval == 0:
                        # Calc
                        train_loss = loss_value / self.logging_inteval
                        train_accuracy = tp_matches / (train_loader.batch_size * self.logging_inteval)
                        current_lr = self._get_lr(self.optimizer)

                        # Print examination result
                        logger.info(f"Epoch: [{current_epoch + 1} / {epochs}] ({step + 1} / {len(train_loader)})")
                        print(f"Training loss: {train_loss:4.4}")
                        print(f"Training accuracy: {train_accuracy:4.2%}")
                        print(f"Learning Rate: {current_lr}")
                        print()

                        # Write at Tensorboard
                        self.tensorboard.add_scalar("Train/loss", train_loss, current_epoch * len(train_loader) + step)
                        self.tensorboard.add_scalar("Train/accuracy", train_accuracy, current_epoch * len(train_loader) + step)
                        self.tensorboard.add_scalar("Train/learning_rate", current_lr, current_epoch * len(train_loader) + step)

                        # Initialize examination
                        loss_value = 0
                        tp_matches = 0
            
                self.scheduler.step()

                # Validation
                with torch.no_grad():    
                    logger.info("Validating...")
                    self.model.eval()

                    loss_value = 0
                    tp_matches = 0
                    pred_list = []
                    label_list = []
                    for sources, labels in tqdm(valid_loader, total=len(valid_loader)):
                        sources: Tensor = sources.to(self.device)
                        labels: Tensor = labels.to(self.device, dtype=torch.long)

                        outputs = self.model(sources)
                        predicts = torch.argmax(outputs, dim=-1)
                        loss = self.criterion(outputs, labels)
                        loss_value += loss.item()
                        tp_matches += (predicts == labels).sum().item()

                        pred_list.extend(predicts.cpu().tolist())
                        label_list.extend(labels.cpu().tolist())
                    
                    # Validation 결과 계산
                    valid_loss = loss_value / len(valid_loader)
                    valid_accuracy = tp_matches / len(valid_loader.dataset)
                    best_valid_loss = min(best_valid_loss, valid_loss)

                    # 모델 저장
                    # 추후 f1-score로 바꾸기
                    if valid_accuracy > best_valid_accuracy:
                        best_valid_accuracy = valid_accuracy
                        best_valid_accuracy_epoch = current_epoch
                        logger.info(f"Saving best accuracy model: {valid_accuracy:4.2%}...")
                        torch.save(self.model.state_dict(), f"{self.paths.checkpoint}/best.pth")

                    torch.save(self.model.state_dict(), f"{self.paths.checkpoint}/last.pth")
                    self.tensorboard.add_scalar("Val/loss", valid_loss, current_epoch)
                    self.tensorboard.add_scalar("Val/accuracy", valid_accuracy, current_epoch)
                    logger.info(f'Epoch {current_epoch + 1} end.')

        # Training End

    
    def _get_lr(self, optimizer: Optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
