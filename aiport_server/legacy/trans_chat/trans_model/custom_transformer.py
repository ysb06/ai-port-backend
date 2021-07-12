import torch
import math 
from torch import Tensor
from torch.nn import LayerNorm
from torch.nn import Module
from torch.nn import Embedding
from torch.nn.modules.activation import MultiheadAttention, ReLU
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

# 주요 모듈은 Self-Attention(Multi-Head, Masked Multi-Head), Position-Wise FFNN, Residual Connection, Normalization(LayerNorm)
# LayerNorm에 관해서는 https://jeongukjae.github.io/posts/apex-fused-layer-norm-vs-torch-layer-norm/


class Transformer(Module):
    def __init__(self, d_model: int = 512, num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048):
        super(Transformer, self).__init__()
        self.embedding = Embedding(d_model, num_layers)
        self.positional_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, num_layers, num_heads)
        self.mask = None
        self.decoder = TransformerDecoder(d_model, num_layers, num_heads)

    def forward(self, src: Tensor, target: Tensor):
        memory = self.encoder(src)
        output = self.decoder(target, memory)

        return output


class MultiHeadAttention(Module):
    """
    제작 중
    """

    def __init__(self, input_dimension: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.input_dimension = input_dimension


class TransformerEncoder(Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(
            [TransformerEncoderLayer(d_model, num_heads)
             for _ in range(num_layers)]
        )

    def forward(self, src: Tensor):
        output = src

        for layer in self.layers:
            output = layer(output)

        # norm 한 번 더 할수도 있다.

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dimension: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Multi-Head Self-Attention
        self.self_attention = MultiheadAttention(d_model, num_heads)
        self.dropout_a = Dropout(dropout)
        # Normalization after Self-Attention
        self.norm1 = LayerNorm(d_model)
        # Position-Wise Feed Forward NN
        self.linear1 = Linear(d_model, feedforward_dimension)
        self.relu = ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(feedforward_dimension, d_model)
        self.dropout2 = Dropout(dropout)
        # Normalization after PW-FFNN
        self.norm2 = LayerNorm(d_model)

    def forward(self, src: Tensor):
        attention_value = self.self_attention(src, src, src)[0]
        out = src + self.dropout_a(attention_value)
        out = self.norm1(out)

        out2 = self.linear2(self.dropout1(self.relu(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)

        return out


class TransformerDecoder(Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList(
            [TransformerDecoderLayer(d_model, num_heads)
             for _ in range(num_layers)]
        )

    def forward(self, src: Tensor):
        output = src

        for layer in self.layers:
            output = layer(output)

        # norm 한 번 더 할수도 있다.

        return output


class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, num_heads, feedforward_dimension: int = 2048, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        # Masked Multi-Head Self-Attention
        self.masked_self_attention = MultiheadAttention(
            d_model, num_heads, dropout=dropout)
        self.dropout_a1 = Dropout(dropout)

        # Normalization after Self-Attention
        self.norm1 = LayerNorm(d_model)

        # Encoder-Decoder Attention
        self.self_attention = MultiheadAttention(
            d_model, num_heads, dropout=dropout)
        self.dropout_a2 = Dropout(dropout)

        # Normalization after Attention
        self.norm2 = LayerNorm(d_model)

        # Position-Wise Feed Forward NN
        self.linear1 = Linear(d_model, feedforward_dimension)
        self.relu = ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(feedforward_dimension, d_model)
        self.dropout2 = Dropout(dropout)

        # Normalization after PW-FFNN
        self.norm3 = LayerNorm(d_model)

    def forward(self, target: Tensor, memory: Tensor):
        attention_value = self.masked_self_attention(target, target, target)[0]
        out = target + self.dropout_a1(attention_value)
        out = self.norm1(out)

        attention_value = self.self_attention(target, memory, memory)[0]
        out2 = out + self.dropout_a2(attention_value)
        out = self.norm2(out2)

        out2 = self.linear2(self.dropout1(self.relu(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm3(out)

        return out


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    마스크 생성
    """
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout=0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
    
    def forword(self, x: Tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        