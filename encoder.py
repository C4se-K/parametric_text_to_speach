import torch
import torch.nn as nn
import math
import copy

#duplication of identical copies of layers
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#module to convert output of the transformer to mel spectrogram frames
class MelSpectrogramGenerator(nn.Module):
    def __init__(self, d_model, num_mel_bins, static_seq_len):
        super(MelSpectrogramGenerator, self).__init__()
        self.linear = nn.Linear(d_model, num_mel_bins * static_seq_len)
        self.num_mel_bins = num_mel_bins
        self.static_seq_len = static_seq_len

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x)
        x = x.view(batch_size, -1, self.num_mel_bins, self.static_seq_len)
        x = x.mean(dim=1)
        return x

#sets positional encodings to inform the model about the positional relationship of the tokens
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#allows for the attendance of information from different subspaces
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

#used for processing. 
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

#applies layer normalization.applies to the data in the batches
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#helps mitigating the vanishing gradient problem
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#defines a layer of transformer encoder. with self attention and position wise feed forward
#networks
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

#a stack of sequential processing layers 
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#maps input tokens to vectors of a specified dimention
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

#combines the encoder with input embeddings
class TransformerEncoder(nn.Module):
    def __init__(self, encoder, src_embed):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def forward(self, src, src_mask):
        embedded_src = self.src_embed(src)
        return self.encoder(embedded_src, src_mask)

#integrates the transformer encoder with mel spectrogram generator
class CustomTransformerModel(nn.Module):
    def __init__(self, encoder, src_embed, mel_generator):
        super(CustomTransformerModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.mel_generator = mel_generator

    def forward(self, src, src_mask):
        embedded_src = self.src_embed(src)
        encoder_output = self.encoder(embedded_src, src_mask)
        mel_output = self.mel_generator(encoder_output)
        return mel_output


def make_model(src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, num_mel_bins=256, static_seq_len=1173):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, N)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    mel_generator = MelSpectrogramGenerator(d_model, num_mel_bins, static_seq_len)
    model = CustomTransformerModel(encoder, src_embed, mel_generator)
    return model


