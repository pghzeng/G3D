# -----G3D-----
# This module contains the encoder and decoder modules which are components of the Orca models. Some contents have been modified with annotation.
# Added multi-head attention module which take gene expression as query. Source code modified from https://nn.labml.ai/transformers/mha.html
# Modified orca modules to get rid of normmats

# round 2 v1:
# Use cross-attention to capture 'regulation' of expression on seq, rather than directly output 'regulated seq'. So take seq as Q, exp as K/V, then use resnet to add the results.
# Only use one attention layer at the end of encoder rather than 3 layers jammed in original encoder.
# Not using positionial encoding.

import numpy as np
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# can be set to lower values to decrease memory usage
# at least 4000 * 50 recommended for performance
# Blocksize = 4000 * 200

# class FixedPositionalEncoding(nn.Module):
#     def __init__(self, channel, seq_len):
#         super(FixedPositionalEncoding, self).__init__()
        
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0)  # shape [1, seq_len]
#         div_term = torch.exp(torch.arange(0, channel, 2).float() * (-math.log(10000.0) / channel)).unsqueeze(1)  # shape [channel // 2, 1]

#         pe = torch.zeros(1, channel, seq_len)
#         pe[0, 0::2, :] = torch.sin(position * div_term)
#         pe[0, 1::2, :] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe
#         return x

class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, channel: int, heads: int, channel_per_head: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(channel, channel, bias=bias)
        self.heads = heads
        self.channel_per_head = channel_per_head

    def forward(self, x: torch.Tensor):
        # Input sequence encoding has shape [batch_size, channel, seq_len]
        x = x.permute(2, 0, 1)
        # [seq_len, batch_size, channel]
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.channel_per_head)
        # [seq_len, batch_size, heads, channel_per_head]
        return x

class G3DCrossAttention(nn.Module):
    def __init__(self, heads: int, sequence_length: int, sequence_channel: int, gene_number: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        assert (sequence_channel % heads == 0), "channel number need to be divisible by heads"
        self.heads = heads
        self.sequence_length = sequence_length
        self.gene_number = gene_number
        self.sequence_channel = sequence_channel
        self.channel_per_head = sequence_channel // heads
        
        self.gene_proj = nn.Linear(1, sequence_channel, bias=bias)
        self.query = PrepareForMultiHeadAttention(sequence_channel, heads, self.channel_per_head, bias=bias)
        self.key = PrepareForMultiHeadAttention(sequence_channel, heads, self.channel_per_head, bias=bias)
        self.value = PrepareForMultiHeadAttention(sequence_channel, heads, self.channel_per_head, bias=bias)
        
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(sequence_channel, sequence_channel)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.channel_per_head)
        
        self.norm1 = nn.LayerNorm(sequence_channel)
        self.norm2 = nn.LayerNorm(sequence_channel)
        
        self.ffn = nn.Sequential(
            nn.Linear(sequence_channel, 4 * sequence_channel),
            nn.ReLU(),
            nn.Linear(4 * sequence_channel, sequence_channel),
        )
        
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
        
    def forward(self, seq: torch.Tensor, exp: torch.Tensor):
        # seq has shape [batch_size, channel, seq_len]
        # exp has shape [batch_size, gene_number]

        batch_size, channel, seq_len = seq.shape
        assert seq_len == self.sequence_length
        assert channel == self.sequence_channel

        exp_proj = self.gene_proj(exp.unsqueeze(-1)) # [batch_size, gene_number, channel]
        exp_proj = exp_proj.permute(0, 2, 1) # [batch_size, channel, gene_number]
        query = self.query(seq) # get Q from seq
        key = self.key(exp_proj)
        value = self.value(exp_proj)
        # # [seq_len, batch_size, heads, channel_per_head]
        scores = self.get_scores(query, key)
        scores *= self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value).reshape(seq_len, batch_size, -1) # [seq_len, batch_size, channel]
        x = self.norm1(x + seq.permute(2, 0, 1)) # add & norm
        x2 = self.ffn(x) # feed forward
        x = self.norm2(x + x2) # add & norm
        x = self.output(x)
        x = x.permute(1, 2, 0)  # [batch_size, channel, seq_len] to keep the same dim as input
        return x

class Decoder(nn.Module):
    def __init__(self, upsample_mode='nearest'):
        """
        Orca decoder architecture.
        """
        super(Decoder, self).__init__()
        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )
        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0, dilation=1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0, dilation=1),
        )

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode=upsample_mode)
        self.lcombiner = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv2d(65, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
        )
        self.combiner = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.lcombinerD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1), # G3D: changed input channel 129 to 128
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
        )
        self.combinerD = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y=None):
        mat = x[:, :, :, None] + x[:, :, None, :]
        # mat = torch.cat([mat, distenc], axis=1)
        mat = self.lcombinerD(mat) # G3D: although named as combinerD, they are no longer used for combining distenc matrix
        mat = self.combinerD(mat) + mat

        if y is not None:
            mat = torch.cat([mat, self.upsample(y)], axis=1)
        cur = mat
        first = True
        for lm, m in zip(self.lconvtwos, self.convtwos):
            if first:
                if y is not None:
                    cur = self.lcombiner(cur)
                    cur = self.combiner(cur) + cur
                else:
                    cur = lm(cur)
                    cur = m(cur) + cur
                first = False
            else:
                lout = lm(cur)
                if lout.size() == cur.size():
                    cur = lout + cur
                else:
                    cur = lout
                cur = m(cur) + cur
        cur = self.final(cur)
        return 0.5 * cur + 0.5 * cur.transpose(2, 3)


class Decoder_1m(nn.Module):
    def __init__(self):
        """
        Decoder for training the 1Mb module. Used for pretraining
        the Encoder or used with the Encoder as a 
        standalone 1Mb model.
        """
        super(Decoder_1m, self).__init__()
        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )

        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )

    def forward(self, x):
        mat = x[:, :, :, None] + x[:, :, None, :]

        cur = mat
        first = True
        for lm, m in zip(self.lconvtwos, self.convtwos):
            if first:
                cur = lm(cur)
                cur = m(cur) + cur
                first = False
            else:
                lout = lm(cur)
                if lout.size() == cur.size():
                    cur = lout + cur
                else:
                    cur = lout
                cur = m(cur) + cur
        cur = self.final(cur)
        return 0.5 * cur + 0.5 * cur.transpose(2, 3)

class Encoder(nn.Module):
    def __init__(self):
        """
        The first section of the Orca Encoder (sequence at bp resolution
        to 4kb resolution)
        """
        super(Encoder, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        binsize = 4000
        x_padding = 112000
        x_block = Blocksize

        def run(x, dummy):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            return out7

        dummy = torch.Tensor(1)
        dummy.requires_grad = True

        segouts = []
        starts = np.arange(0, x.size(2), x_block)
        for start in starts:
            if start == starts[0]:
                segouts.append(
                    checkpoint(run, x[:, :, start : start + x_block + x_padding], dummy)[
                        :, :, : int(x_block / binsize)
                    ]
                )
            elif start == starts[-1]:
                segouts.append(
                    checkpoint(run, x[:, :, start - x_padding :], dummy)[
                        :, :, int(x_padding / binsize) :
                    ]
                )
            else:
                segouts.append(
                    checkpoint(
                        run, x[:, :, start - x_padding : start + x_block + x_padding], dummy,
                    )[
                        :, :, int(x_padding / binsize) : int((x_block + x_padding) / binsize),
                    ]
                )

        out = torch.cat(segouts, 2)
        return out


class Encoder_wo_checkpoint(nn.Module):
    def __init__(self):
        super(Encoder_wo_checkpoint, self).__init__()
        # self.positional_encoding = FixedPositionalEncoding(4, 2000000)

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        # binsize = 4000
        # x_padding = 112000
        # x_block = Blocksize
        # print("x", x.shape)
        def run(x):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            return out7

        out = run(x)
        return out

# -----G3D-----
# Only keep 2Mb level and get rid of checkpoint
class Encoder2(nn.Module):
    def __init__(self):
        """
        The second section of the Orca Encoder (4kb resolution to
        128kb resolution)
        """
        super(Encoder2, self).__init__()
        self.lblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(128, 128, kernel_size=9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, 128, kernel_size=9, padding=4),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(128, 128, kernel_size=9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.downlblocks = nn.ModuleList([
            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128))])

        self.downblocks = nn.ModuleList([
            nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)),
        ])

    def forward(self, x):
        """Forward propagation of a batch."""
        out = x

        encodings = [out]
        for lconv, conv in zip(self.lblocks, self.blocks):
            lout = lconv(out)
            out = conv(lout) + lout
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout) + lout
            out = enc + out
            encodings2.append(out)
        encodings2.reverse()
        return encodings2


class Net(nn.Module):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model. The trained model weighted can be
        loaded into Encoder and Decoder_1m modules.

        Parameters
        ----------
        num_1d : int or None, optional
            The number of 1D targets used for the auxiliary
            task of predicting ChIP-seq profiles.
        """
        super(Net, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )

        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )
        if num_1d is not None:
            self.final_1d = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, num_1d, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        self.num_1d = num_1d

    def forward(self, x):
        """Forward propagation of a batch."""

        def run0(x, dummy):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, dummy)
        else:
            cur = checkpoint(run0, x, dummy)

        def run1(cur):
            first = True
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur
