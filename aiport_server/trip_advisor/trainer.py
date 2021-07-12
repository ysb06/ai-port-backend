import logging

from aiport_server.trainee import TraineeBase
from aiport_server.trip_advisor.training.klue_dst_loader import load_dataset

logger = logging.getLogger(__name__)


class DSTTrainee(TraineeBase):
    def on_train_begin(self):
        datasets = load_dataset(self.paths.train_data)
        logger.info(datasets)

    def on_train(self):
        print('OK')
