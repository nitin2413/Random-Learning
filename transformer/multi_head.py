import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════════════════════
# VERSION 1 — Naive (loop over heads, easy to understand)
# ═══════════════════════════════════════════════════════════════

class NaiveMultiAttn(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        # If n_heads = 8, this creates 8 different weight matrices for Q, K, and V, one for each head.
        self.Wq = nn.ModuleList([nn.Linear(self.d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.Wk = nn.ModuleList([nn.Linear(self.d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.Wv = nn.ModuleList([nn.Linear(self.d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.Wo = nn.Linear(self.d_model , self.d_head , bias = False)

    def forward(self,x , mask=None):
        B , T , D = x.shape() # good practice to not not assign( d_model instead D ) as input can have different dimension
        head_outputs = []

        for i in range(self.n_heads):
            Q = self.Wq[i](x)
            K = self.Wk[i](x)
            V = self.Wv[i](x)

            scores = torch.matmul(Q , K.transpose(-2, -1)) / (self.d_head ** 0.5)
            if mask is not None :
                scores = scores.masked_fill(mask == 0, float('-inf'))
            weights = F.softmax(scores)
            weights = torch.nan_to_num(weights)
            out = weights @ V
            head_outputs.append(out)

        concat = torch.cat(head_outputs , dim =-1)

        return self.Wo(concat)

# ═══════════════════════════════════════════════════════════════
# VERSION 2 — Efficient (all heads in parallel via reshape)
# ═══════════════════════════════════════════════════════════════

class EffMultiAttn(nn.Module):
    def __init__(self,d_models,n_heads,dropout=0.1):
        super().__init__()
        assert d_models % n_heads == 0

        self.n_heads = n_heads
        self.d_models = d_models
        self.d_head = d_models / n_heads

        self.qkv = nn.Linear(d_models,d_models * 3,bias=False)
        self.Wo = nn.Linear(d_models , d_models , bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        B , T , D = x.shape()

        qkv = self.qkv(x)
        Q , K , V = qkv.chunk(3, dim = -1)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(2 ,1) # dim(2, 6, 128)->(2,6,4,32)transpose->(2,4,6,32)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(2, 1)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(2, 1)

        # Different heads have different weights,
        # but they are computed together using large matrix operations.

        scores = torch.matmul(Q , K.transpose(-2,-1)) / (self.d_head ** 0.5) # calculation happen here in one big matrix
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores)
        weights = torch.nan_to_num(weights)
        weights = self.dropout(weights)

        out = weights @ V # every head multiply their weights together with value
        out = out.transpose(2,1).contiguous().view(B, T, D) # merge heads

        return self.Wo(out)

# ENCODER ------------------------
# Input → embeddings → self-attention → output (encoder_out)
#
# DECODER ------------------------
# decoder_x → Q
#
# encoder_out → K, V
#
# Q interacts with K,V → cross-attention output

class CrossAttention(nn.Module):
    """
       Cross-attention: Q from decoder, K/V from encoder.
       Used in encoder-decoder transformers (T5, BART, original Transformer).
    """
    def __init__(self,d_models,n_heads,dropout=0.1):
        super().__init__()
        assert d_models % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_models // n_heads
        # Q comes from decoder — separate projection
        self.Wq = nn.Linear(d_models , d_models, bias= False)

        # K and V come from encoder — separate projections
        self.Wk = nn.Linear(d_models, d_models, bias=False)
        self.Wv = nn.Linear(d_models, d_models, bias=False)
        self.Wo = nn.Linear(d_models, d_models, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,decoder_x,encoder_out):
        B = decoder_x.size(0)
        Td = decoder_x.size(1) # target seq len (decoder)
        Te = encoder_out.size(1) # source seq len (encoder)
        D = decoder_x.size(2)

        Q = self.Wq(decoder_x).view(B ,Td , self.n_heads,self.d_head).transpose(1,2) # (B,H,Td,dh)
        K = self.Wk(encoder_out).view(B, Te, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Te,dh)
        V = self.Wv(encoder_out).view(B, Te, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,Te,dh)

        score = Q @ K.transpose(-2,-1) / (self.d_head ** 0.5)
        weights = F.softmax(score)
        weights = self.dropout(weights)

        out = weights @ V
        # ── Step 4: merge heads back
        # .transpose() swaps back → (B, T, H, dh)
        # .contiguous() ensures memory is laid out correctly for .view()
        # .view()       merges last two dims → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, Td, D)  # (B,Td,D)

        return self.Wo(out)

seq_len , tgt_len = 8, 5
decoder_x = torch.randn(B , tgt_len , D)
encoder_out = torch.randn(B , seq_len , D)

cross_attn  = CrossAttention(d_model=128, n_heads=4)
cross_attn.eval()