import pandas as pd
import torch
import copy
from typing import Dict, List, Tuple
from torch import LongTensor
from torch.tensor import Tensor
from torch.utils.data import Dataset
from konlpy.tag import Kkma
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


class ConversationDataset(Dataset):
    def __init__(self, kor2idx: dict, idx2kor: dict, eng2idx: dict, idx2eng: dict, tensor_sentence_data: list) -> None:
        super().__init__()
        self.kor2idx = kor2idx
        self.idx2kor = idx2kor
        self.eng2idx = eng2idx
        self.idx2eng = idx2eng

        self.tensor_sentences = tensor_sentence_data

    def __len__(self):
        return len(self.tensor_sentences)

    def __getitem__(self, index) -> Tuple[LongTensor, LongTensor]:
        return self.tensor_sentences[index]


def generate_conversation_dataset(
    path: str,
    shuffle: bool = True,
    seed: int = None,
    batch_size: int = None,
    sort_by_len=True,
    test_ratio: float = 0.2, validation_ratio: float = 0.2
):
    kor2idx = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
    idx2kor = {0: '<unk>', 1: '<pad>', 2: '<sos>', 3: '<eos>'}
    eng2idx = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
    idx2eng = {0: '<unk>', 1: '<pad>', 2: '<sos>', 3: '<eos>'}

    raw_data = pd.read_csv(path)
    if shuffle:
        if seed is None:
            raw_data = raw_data \
                .sample(frac=1) \
                .reset_index(drop=True)
        else:
            raw_data = raw_data \
                .sample(frac=1, random_state=seed) \
                .reset_index(drop=True)

    raw_kor = raw_data["kor_sent"].to_numpy()
    raw_eng = raw_data["eng_sent"].to_numpy()

    # 영어는 단어 단위로 토큰화하는 데 이는 번역 결과를 도출하는 데 있어 편리성을 위한 것임.
    # 정확한 번역을 위해서는 더 자세한 수준으로 토큰화한 후 재조립 과정이 필요할 것으로 보임.
    kkma = Kkma()

    kor_morphs_set = set()
    eng_words_set = set()
    tokenized_sentences = []

    print("Torkenizing and generating vocabs...")
    for kor_sentence, eng_sentence in tqdm(zip(raw_kor, raw_eng), total=len(raw_kor)):
        kor_sentence_morphs = kkma.morphs(kor_sentence)
        eng_sentence_words = word_tokenize(eng_sentence.lower())    # 소문자 후 토큰화

        kor_morphs_set.update(kor_sentence_morphs)  # 한국어
        eng_words_set.update(eng_sentence_words)    # 영어 Vocab 대상 집합

        tokenized_sentences.append(
            (['<sos>'] + kor_sentence_morphs + ['<eos>'],
             ['<sos>'] + eng_sentence_words + ['<eos>'])
        )  # 문장 전처리

    kor_morphs_set = sorted(kor_morphs_set)
    eng_words_set = sorted(eng_words_set)

    # 한글, 영어 vocab 사전 생성, 사전 초기화를 하지 않는 부분을 유의 (에러 가능성)
    for morph in kor_morphs_set:
        # 아래 조건문은 이전의 코드에서 생긴 문제에서 비롯됨.
        if morph not in kor2idx:
            if len(kor2idx) != len(idx2kor):
                raise Exception("Korean dictionary is broken.")
            idx = len(kor2idx)
            kor2idx[morph] = idx
            idx2kor[idx] = morph

    for word in eng_words_set:
        # 여기 조건문도 위와 마찬가지.
        if word not in eng2idx:
            if len(eng2idx) != len(idx2eng):
                raise Exception("English dictionary is broken.")
            idx = len(eng2idx)
            eng2idx[word] = idx
            idx2eng[idx] = word

    print("Spliting dataset...")
    test_size = int(len(tokenized_sentences) * test_ratio)
    validation_size = int(len(tokenized_sentences) * validation_ratio)

    test_tokenized_sentence = tokenized_sentences[:test_size]
    validation_tokenized_sentence = tokenized_sentences[test_size:test_size + validation_size]
    train_tokenized_sentence = tokenized_sentences[test_size + validation_size:]

    test_set_tensor = []
    validation_set_tensor = []
    train_set_tensor = []

    if sort_by_len:
        test_tokenized_sentence.sort(key=lambda x: len(x[0]), reverse=True)
        validation_tokenized_sentence.sort(key=lambda x: len(x[0]), reverse=True)
        train_tokenized_sentence.sort(key=lambda x: len(x[0]), reverse=True)

    print("Indexing Senteces (Test set)...")
    test_set_tensor = __generate_batch(test_tokenized_sentence, batch_size, kor2idx, eng2idx)

    print("Indexing Senteces (Validation set)...")
    validation_set_tensor = __generate_batch(validation_tokenized_sentence, batch_size, kor2idx, eng2idx)

    print("Indexing Senteces (Train set)...")
    train_set_tensor = __generate_batch(train_tokenized_sentence, batch_size, kor2idx, eng2idx)

    test_set = ConversationDataset(
        copy.deepcopy(kor2idx), copy.deepcopy(idx2kor), 
        copy.deepcopy(eng2idx), copy.deepcopy(idx2eng),
        test_set_tensor
    )

    validation_set = ConversationDataset(
        copy.deepcopy(kor2idx), copy.deepcopy(idx2kor), 
        copy.deepcopy(eng2idx), copy.deepcopy(idx2eng),
        validation_set_tensor
    )

    train_set = ConversationDataset(
        kor2idx, idx2kor, 
        eng2idx, idx2eng,
        train_set_tensor
    )

    return train_set, validation_set, test_set


def __generate_batch(dataset: list, batch_size: int, kor2idx: dict, eng2idx: dict) -> Tuple[Tensor, Tensor]:
    result = []

    sentence_code_cache: List[Tuple[List, List]] = []
    max_kor_length = 0
    max_eng_length = 0

    for index, (kor_tokenized_sentence, eng_tokenized_sentence) in enumerate(dataset):
        kor_encoded_sentence = [kor2idx[token]
                                for token in kor_tokenized_sentence]
        eng_encoded_sentence = [eng2idx[token]
                                for token in eng_tokenized_sentence]

        # deque를 쓰는 게 효율적일까? 일단 별 차이 없을 것 같아 그냥 둠.
        sentence_code_cache.append((kor_encoded_sentence, eng_encoded_sentence))
        max_kor_length = max(max_kor_length, len(kor_encoded_sentence))
        max_eng_length = max(max_eng_length, len(eng_encoded_sentence))

        # 배치 및 Shape 처리 코드
        if index is None or index % batch_size == batch_size - 1 or index == batch_size - 1:
            kor_sentence_tensor_batch = None
            eng_sentence_tensor_batch = None

            for kor_sentence_code, eng_sentence_code in sentence_code_cache:
                while len(kor_sentence_code) < max_kor_length:
                    kor_sentence_code.append(kor2idx["<pad>"])

                while len(eng_sentence_code) < max_eng_length:
                    eng_sentence_code.append(kor2idx["<pad>"])

                kor_sentence_tensor = LongTensor(kor_sentence_code).unsqueeze(1)
                eng_sentence_tensor = LongTensor(eng_sentence_code).unsqueeze(1)

                if kor_sentence_tensor_batch is None:
                    kor_sentence_tensor_batch = kor_sentence_tensor
                else:
                    kor_sentence_tensor_batch = torch.cat((kor_sentence_tensor_batch, kor_sentence_tensor), 1)

                if eng_sentence_tensor_batch is None:
                    eng_sentence_tensor_batch = eng_sentence_tensor
                else:
                    eng_sentence_tensor_batch = torch.cat((eng_sentence_tensor_batch, eng_sentence_tensor), 1)

            result.append((kor_sentence_tensor_batch, eng_sentence_tensor_batch))
            sentence_code_cache.clear()
            max_kor_length = 0
            max_eng_length = 0

    return result
