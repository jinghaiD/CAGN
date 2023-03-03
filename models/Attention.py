from os import sched_get_priority_max
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, conv=None, matx=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores * mask
        scores = scores.masked_fill(mask == 0, -1e9)
    if matx is not None:
        scores = torch.cat((scores, matx),dim=1)
        scores = conv(scores)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), scores


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.liners = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(2*h,2*h,1),
            nn.Conv2d(2*h,h,1),
            nn.Conv2d(h,h,1)
        )

    def forward(self, query, key, value, mask=None, matx=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value =[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.liners, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, conv=self.conv, matx=matx)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.liners[-1](x), self.attn
