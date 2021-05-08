import math
import torch
from torch import nn, Tensor

# d_model이 Embedding 크기

class TransformerTransModel(nn.Module):
    def __init__(
        self,
        input_vocab_size: int, output_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feed_forward: int = 2048,
        dropout: int = 0.1,
        padding_vocab_index = 1
    ):
        super(TransformerTransModel, self).__init__()
        self.padding_vocab_index = padding_vocab_index

        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_vocab_size)
        self.pre_transformer_dropout = nn.Dropout(dropout, inplace=True)

        self.transformer = nn.Transformer(d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feed_forward, dropout)

        # Linear 구성 어떻게??
        self.fc_linear = nn.Linear(d_model, output_vocab_size)

    def forward(self, source: Tensor, target: Tensor):
        """
        Source의 차원은 (S, N, E).
        Target의 차원은 (T, N, E).
        여기서 S는 Source의 길이 T는 Target의 길이, N은 배치 크기, E는 Feature의 크기이다.
        Feature는 텍스트처리에서는 보통 사용되지 않으며 일반적인 Source와 Target의 모습은 아래와 같다.

        Ex) 

        tensor([ [2,2,2,2], [389, 2505, 2341, 3487], [86, 913, 3020, 4103], [2651, 9, 2851, 3], [3, 3, 3, 1] ])

        Forward만 하는 경우 일반적으로 아래와 같은 시작 값만 넣고 종료 값이 나올 때까지 반복 출력한다.

        Ex) 시작 값이 2고 종료 값이 3일때,

        tensor([[2]]) -> tensor([[2, 23]]) -> tensor([[2, 23, 5634]]) -> tensor([[2, 23, 5634, 3]])

        Args:
            source (Tensor): Transformer 입력
            target (Tensor): Transformer 출력

        Returns:
            Tensor: Forward 출력 결과
        """
        source_embedding = self.input_embedding(source)
        source_input = self.pos_encoder(source_embedding)
        source_input = self.pre_transformer_dropout(source_input)

        target_embedding = self.output_embedding(target)
        target_input = self.pos_encoder(target_embedding)
        target_input = self.pre_transformer_dropout(target_input)

        source_padding_mask = generate_padding_mask(source, self.padding_vocab_index).to(source.device)
        target_mask = generate_square_subsequent_mask(target.shape[0]).to(target.device)
        output = self.transformer(source_input, target_input, src_key_padding_mask=source_padding_mask, tgt_mask=target_mask)

        output = self.fc_linear(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)    # 0부터 2칸씩
        pe[:, 1::2] = torch.cos(position * div_term)    # 1부터 2칸씩
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def _apply(self, fn):
        super(PositionalEncoding, self)._apply(fn)
        self.pe = fn(self.pe)
        return self

    def forward(self, x: Tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

def generate_padding_mask(src: Tensor, src_pad_idx: int):
    src_mask = src.transpose(0, 1) == src_pad_idx

    return src_mask
