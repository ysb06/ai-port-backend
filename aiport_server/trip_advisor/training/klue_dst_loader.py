import json
import logging

logger = logging.getLogger(__name__)


def load_dataset(data_path: str):
    logger.info(f'Loading data from {data_path}')
    with open(data_path, 'r', encoding='utf8') as fr:
        data = json.load(fr)
    
    logger.info(f'Data Length: {len(data)}')

    for key in data[0]:
        print(key)
    return []
