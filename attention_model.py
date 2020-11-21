#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
# !/PycharmProjects/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Sublayers(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout,device):
        super(Sublayers, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, h, dropout,device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,device)
        #self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout,device) for i in range(2)])
        self.norm = LayerNorm(d_model, device)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask):
        # x = self.sublayer[0](x, lambda x: self.self_attn(x))
        # return self.sublayer[1](x, self.feed_forward(x))
        Z,weight=self.self_attn(self.norm(x), mask)
        Z1=x+self.dropout(Z)
        Z2=Z1+self.dropout(self.feed_forward(self.norm(Z1)))
        return Z2,weight


class Attention(nn.Module):
    def __init__(self, d_model, dropout,device):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model).to(device)
        self.W_k = nn.Linear(d_model, d_model).to(device)
        self.W_v = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input,mask=None):
        d_k = input.size(2)
        Q, K, V=self.W_q(input), self.W_k(input), self.W_v(input)

        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        weights = F.softmax(scores, dim=-1)
        S = torch.bmm(weights, V)
        return S,weights

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout,device):
        super().__init__()
        assert d_model % h == 0
        self.n_heads = h
        self.d_model = d_model
        self.head_dim = d_model // h  # 每个head空间维度
        self.linears = clones(nn.Linear(d_model, d_model).to(device), 4)
        self.dropout = nn.Dropout(dropout)
        self.scale= torch.sqrt(torch.FloatTensor([d_model // h])).to(device)

    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        """变形 Q K,V= [batch size, n heads, query/key/value  len, head dim]"""
        Q, K ,V= [l(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
         for l, x in zip(self.linears, (inputs, inputs,inputs))]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # matmul矩阵相乘
        if mask is not None:
            mask = mask.unsqueeze(1)#所有head
            scores = scores.masked_fill(mask == 0, -1e9)

        weights =self.dropout(F.softmax(scores, dim=-1))
        V1=torch.matmul(weights, V)
        V2 = V1.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * (self.d_model // self.n_heads))
        Z=self.linears[-1](V2)
        weights_normalization=torch.sum(weights,dim=1)/self.n_heads
        return Z,weights_normalization

class LayerNorm(nn.Module):

    def __init__(self, d, device):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d)).cuda(device)
        self.b_2 = nn.Parameter(torch.zeros(d)).cuda(device)
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x1 = self.a_2 * (x - mean)
        x2= x1/ (std + self.eps) + self.b_2
        return x2

"""为了使模型具有非线性，并考虑不同潜在维数之间的相互作用，输入输出维度d_model，内层维度d_ff"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout,device):
        super(PositionwiseFeedForward, self).__init__()
        self.f_1 = nn.Linear(d_model, d_ff,bias=True).to(device)
        self.f_2 = nn.Linear(d_ff, d_model,bias=True).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.f_1(x)
        x1 = self.dropout(F.relu(x1))
        x2 = self.f_2(x1)
        return x2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * ((math.log(10000.0) / d_model)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)