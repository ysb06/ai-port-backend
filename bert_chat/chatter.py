from typing import List
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

MODEL_NAME = "bert-base-multilingual-cased"
print(f"Initialize Model: [{MODEL_NAME}]")
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
model: BertModel = AutoModel.from_pretrained(MODEL_NAME)

intent_cls_hidden = []
mapped_answers = []

def train_chat(questions: List[str], answers: List[str]):
    intent_cls_hidden_result = []
    for question in questions:
        intent_cls_hidden_result.append(get_cls_token(question, model, tokenizer))
    
    intent_cls_hidden.extend(np.array(intent_cls_hidden_result).squeeze(axis=1))
    mapped_answers.extend(answers)
    

def listenChat(text):
    print(f"PyTorch version:[{torch.__version__}].")
    target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device:[{target_device}].")
    
    query_cls_hidden = get_cls_token(text, model, tokenizer)

    cos_sim = cosine_similarity(query_cls_hidden, intent_cls_hidden)
    top_question = np.argmax(cos_sim)

    print(f"Query Text: {text}")
    answer = mapped_answers[top_question]
    print(f"Answer: {answer}")
    return {
        "text": answer
    }


def get_cls_token(sent_A: str, model: nn.Module, tokenizer: PreTrainedTokenizerBase):
    model.eval()
    
    tokenized_sent: BatchEncoding = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )

    with torch.no_grad():
        outputs = model( 
            input_ids=tokenized_sent['input_ids'],              # input text를 tokenizing한 후 vocab의 id
            attention_mask=tokenized_sent['attention_mask'],    # special token (pad, cls, sep) or not, 
                                                                # 특수 토큰이 아닌지 여부, 특수 토큰일 경우 0
            token_type_ids=tokenized_sent['token_type_ids']     # segment id (sentA or sentB)
        )                                                       # sep 토큰 기준으로 앞 문장은 0, 뒷 문장은 1
    
    # Output 설명
    # last_hidden_state: 문장의 토큰들에 대한 각각의 벡터 값
    # pooler_output: CLS 토큰에 대한 벡터 값, 추가로 BertPooler를 통과한 결과 값 (Tanh 활성화+Linear)

    # 즉, 여기서는 BertPooler를 통과하지 않은 CLS 토큰 결과 값을 얻었다.
    # 그런데 왜 굳이 이런 값을 사용했을까?
    # pooler_output은 다음 문장 예측에 필요한 가중치들을 학습한 Linear 레이어를 통과해 얻은 값이다.
    # 즉, 현재 문장을 분류하려는 목적에는 맞지 않는 값이기 때문
    logits = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    return logits

def initialize():
    print("Initial training...")
    train_chat(
        ['기차 타고 여행 가고 싶어','꿈이 이루어질까?','내년에는 더 행복해질려고 이렇게 힘든가봅니다', '간만에 휴식 중', '오늘도 힘차게!'],
        ['꿈꾸던 여행이네요.','현실을 꿈처럼 만들어봐요.','더 행복해질 거예요.', '휴식도 필요하죠', '아자아자 화이팅!!']
    )


if __name__ == "__main__":
    initialize()
    listenChat("아 여행가고 싶다~")
