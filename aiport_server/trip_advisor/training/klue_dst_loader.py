import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_open_vocab_data(
    dialogue_path: str,
    ontology_path: str,
    hyperparameters: Dict
) -> List:
    with open(dialogue_path, 'r', encoding='utf8') as fr:
        dialogue_data = json.load(fr)
        logger.info(f'{dialogue_path} Loaded: {len(dialogue_data)}')
    with open(ontology_path, 'r', encoding='utf8') as fr:
        ontology_data: Dict = json.load(fr)
        logger.info(f'{ontology_path} Loaded')

    slot_meta = [key for key, _ in ontology_data.items()]
    slot_meta.sort()
    gate_list = {
        'none': 0,
        'dontcare': 1,
        'ptr': 2,
        'no': 3,
        'yes': 4,
    }

    data_folds = _create_folds(dialogue_data, **hyperparameters['skf']['args'])
    processed_data_folds: List[Tuple[List, List]] = []

    tokenizer = BertTokenizer.from_pretrained(hyperparameters['tokenizer_pretrained_model_name_or_path'])
    for fold_index, fold in enumerate(data_folds):
        logger.info(f'Preprocessing training data of {fold_index} fold...')
        train_data = _preprocess(fold[0], slot_meta, gate_list, tokenizer)
        logger.info(f'Preprocessing validation data of {fold_index} fold...')
        valid_data = _preprocess(fold[1], slot_meta, gate_list, tokenizer)

        processed_data_folds.append((train_data, valid_data))

    return processed_data_folds

def _preprocess(
        data: List[Dict], 
        slot_meta: List[str], 
        gate_list: Dict[str, int], 
        tokenizer: BertTokenizer,
    ) -> List[Dict]:
    """원 데이터에 Feature 추가

    Args:
        data (List[Dict]): 데이터 원본
        slot_meta (List[str]): Slot 전체 목록 (해당 슬롯 순서로 Feature 생성)
        gate_list (Dict[str, int]): 정의된 Gate 목록
        tokenizer (BertTokenizer): 텍스트를 변환할 Tokenizer

    Returns:
        List[Dict]: 변경된 데이터 (주의: 데이터 원본을 변경시키고 원본 참조를 그대로 반환)
    """
    none_value = tokenizer.encode('none', add_special_tokens=False) + [tokenizer.sep_token_id]
    empty_slot_values = {slot:none_value for slot in slot_meta}
    empty_gates = {slot:gate_list['none'] for slot in slot_meta}

    for example in tqdm(data):
        dialogue: List[Dict] = example['dialogue']
        user_count = 0
        for turn in dialogue:
            turn['text_idx'] = tokenizer.encode(turn['text'])
            
            slot_values = deepcopy(empty_slot_values)
            gates = deepcopy(empty_gates)

            if turn['role'] == 'user':
                user_count = user_count + 1
                for state_raw in turn['state']:
                    state_raw: List[str] = state_raw.split('-')
                    slot = state_raw[0] + '-' + state_raw[1]
                    value = state_raw[2]
                    
                    slot_values[slot] = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.sep_token_id]
                    gates[slot] = gate_list.get(value, gate_list['ptr'])
            
                turn['slot'] = slot_values
                turn['gate'] = gates

        example['count'] = user_count

    return data


def _create_folds(
        data: List[Dict], 
        split_k: int, 
        seed: int = None, 
        target_fold: int = None
    ) -> List[Tuple[List[Dict], List[Dict]]]:
    """도메인 조합을 기준으로 Straitified K Fold를 적용하여 데이터를 분리한다.

    Args:
        data (List): 원본 KLUE DST 데이터
        split_k (int): 분리할 Fold 갯수
        seed (int, optional): 데이터를 Shuffle 할때 사용할 Seed 값. None일 경우 데이터를 Shuffle하지 않는다.
        target_fold (int, optional): Cross Validation을 사용하지 않고 특정 Fold를 지정. None일 경우 모든 Fold를 사용하여 Cross Validation

    Returns:
        List[Tuple[List, List]]: K-Fold로 분리된 데이터 Fold 묶음. Fold를 특정할 경우 해당 데이터 하나만 들어있는 리스트 반환
    """
    logger.info('Splitting by domain combination...')
    domain_comb_list = []
    domain_comb_set = set()

    for dial in data:
        domain_comb = " ".join(sorted(dial["domains"]))
        domain_comb_list.append(domain_comb)
        domain_comb_set.add(domain_comb)

    k_fold_splitter = StratifiedKFold(
        n_splits=split_k, shuffle=(seed != None), random_state=seed)
    folds_indexes = k_fold_splitter.split(X=data, y=domain_comb_list)

    results: List[Tuple[List[Dict], List[Dict]]] = []

    for fold, (train_indexes, valid_indexes) in enumerate(folds_indexes):
        if target_fold is None or target_fold == fold:
            train_data: List[Dict] = []
            valid_data: List[Dict] = []
            train_domains_comb_set: Set[str] = set()
            valid_domains_comb_set: Set[str] = set()

            for index in train_indexes:
                train_data.append(data[index])
                train_domains_comb_set.add(
                    " ".join(sorted(data[index]["domains"])))

            for index in valid_indexes:
                valid_data.append(data[index])
                valid_domains_comb_set.add(
                    " ".join(sorted(data[index]["domains"])))

            # 데이터셋에 없는 도메인 조합 검색 및 출력
            missing_set = domain_comb_set.difference(train_domains_comb_set)
            if len(missing_set) > 0:
                logger.info(f"{missing_set} is not in training set of {fold} fold")

            missing_set = domain_comb_set.difference(valid_domains_comb_set)
            if len(missing_set) > 0:
                logger.info(
                    f"{missing_set} is not in validation set of {fold} fold")

            results.append((train_data, valid_data))

    return results
