from typing import Dict, List
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class DataElement:
    ref_example: Dict
    dialogue_index: int


class OpenVocabDSTDataset(Dataset):
    def __init__(self, raw: List) -> None:
        self.raw: List[Dict] = raw
        self.index_list: List[DataElement] = []

        for example in self.raw:
            for dialogue_index, dialogue in enumerate(example['dialogue']):
                if dialogue['role'] == 'user':
                    self.index_list.append(DataElement(
                        example,
                        dialogue_index
                    ))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index) -> List[Dict]:
        element = self.index_list[index]
        return element.ref_example['dialogue'][:element.dialogue_index + 1]


def collate_TRADE_batch(batch: List[List[Dict]]):
    for elem in batch:
        print(elem)
