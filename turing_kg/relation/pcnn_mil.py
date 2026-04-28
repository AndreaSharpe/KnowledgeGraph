"""PCNN（字级）+ piecewise pooling + 经典 DS selective attention（Lin et al., 2016）。

本文件同时保留：
- 旧版：共享 attention + 多标签 BCE（PCNNMILAttention）
- 新版（经典路线）：逐关系 selective attention + 多分类 softmax（PCNNSelectiveAttention）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN(nn.Module):
    """
    字序列 -> embedding -> Conv1d -> 按实体位置分三段 piecewise max pool -> 全连接 -> num_classes。
    实体位置为字符下标 [0, L)；若无效则退化为整句单段 max（保证可跑）。
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        emb_dim: int = 128,
        num_filters: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.num_filters = num_filters
        self.hidden_dim = num_filters * 3
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(emb_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward_pooled(self, char_ids: torch.Tensor, pos_e1: torch.Tensor, pos_e2: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (batch, L)
        pos_e1, pos_e2: (batch,)  long，实体左边界字符下标
        返回句级 pooled 特征 (batch, hidden_dim)
        """
        b, L = char_ids.shape
        x = self.emb(char_ids)
        x = self.drop(x)
        x = x.transpose(1, 2)
        conv_out = F.relu(self.conv(x))
        conv_out = self.drop(conv_out)

        pooled_rows: list[torch.Tensor] = []
        for i in range(b):
            e1 = int(pos_e1[i].item())
            e2 = int(pos_e2[i].item())
            if e1 > e2:
                e1, e2 = e2, e1
            e1 = max(0, min(e1, L))
            e2 = max(0, min(e2, L))
            if e1 == e2:
                e2 = min(e1 + 1, L)
            row = conv_out[i]
            if e1 > 0:
                s1 = row[:, :e1].max(dim=-1).values
            else:
                s1 = row[:, :1].max(dim=-1).values
            if e2 > e1:
                s2 = row[:, e1:e2].max(dim=-1).values
            else:
                s2 = row[:, e1 : e1 + 1].max(dim=-1).values
            if e2 < L:
                s3 = row[:, e2:].max(dim=-1).values
            else:
                s3 = row[:, -1:].max(dim=-1).values
            pooled = torch.cat([s1, s2, s3], dim=-1)
            pooled_rows.append(pooled)
        return torch.stack(pooled_rows, dim=0)

    def forward(
        self,
        char_ids: torch.Tensor,
        pos_e1: torch.Tensor,
        pos_e2: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 logits (batch, num_classes)（逐句，不经过 bag attention）。"""
        pooled = self.forward_pooled(char_ids, pos_e1, pos_e2)
        return self.fc(pooled)


class MILAttention(nn.Module):
    """
    对同一 bag 内多条句向量做 selective attention（单组 u，对所有关系共享 bag 表示）。
    见：Neural Relation Extraction with Selective Attention over Instances (Lin et al., 2016).
    """

    def __init__(self, hidden_dim: int, att_dim: int = 128) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, att_dim)
        self.u = nn.Linear(att_dim, 1, bias=False)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        H: (n_inst, hidden_dim)
        返回 h_bag (hidden_dim,), attn (n_inst,) 且 attn 在实例维度和为 1
        """
        e = torch.tanh(self.W(H))
        scores = self.u(e).squeeze(-1)
        attn = F.softmax(scores, dim=0)
        h_bag = (attn.unsqueeze(-1) * H).sum(dim=0)
        return h_bag, attn


class PCNNMILAttention(nn.Module):
    """PCNN 编码句向量 + MIL-Attention 聚合 bag，再经共享 fc 得到 bag 级多标签 logits。"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        emb_dim: int = 128,
        num_filters: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.5,
        pad_id: int = 0,
        att_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pcnn = PCNN(
            vocab_size,
            num_classes,
            emb_dim=emb_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.mil_attn = MILAttention(self.pcnn.hidden_dim, att_dim=att_dim)

    def forward_bag(
        self,
        char_ids: torch.Tensor,
        pos_e1: torch.Tensor,
        pos_e2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单 bag 的多实例前向。
        返回 logits_bag (num_classes,), attn (n_inst,)
        """
        h = self.pcnn.forward_pooled(char_ids, pos_e1, pos_e2)
        h_bag, attn = self.mil_attn(h)
        logits = self.pcnn.fc(h_bag.unsqueeze(0)).squeeze(0)
        return logits, attn


def multihot_from_labels(labels_pos: list[str], space: list[str], device: torch.device) -> torch.Tensor:
    y = torch.zeros(len(space), device=device)
    idx = {p: i for i, p in enumerate(space)}
    for p in labels_pos:
        if p in idx:
            y[idx[p]] = 1.0
    return y


def bce_loss_bag(logits_bag: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits_bag, target, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits_bag, target)


class PCNNSelectiveAttention(nn.Module):
    """
    经典 DS bag-level 多分类：逐关系 selective attention。

    记句向量 H=(n_inst, hidden_dim)。对每个 class r：
      alpha_{i,r} = softmax_i( u_r^T tanh(W h_i) )
      h_{bag,r} = sum_i alpha_{i,r} h_i
      logit_r = w_r^T h_{bag,r} + b_r
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        emb_dim: int = 128,
        num_filters: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.5,
        pad_id: int = 0,
        att_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pcnn = PCNN(
            vocab_size,
            num_classes=1,  # 逐句分类头不使用；句向量由 forward_pooled 产出
            emb_dim=emb_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            pad_id=pad_id,
        )
        hidden_dim = self.pcnn.hidden_dim
        self.W = nn.Linear(hidden_dim, att_dim, bias=True)
        # 每个 class 一条 attention 向量 u_r
        self.u = nn.Parameter(torch.empty(num_classes, att_dim))
        nn.init.xavier_uniform_(self.u)
        # 每个 class 一套分类参数（等价于对 h_{bag,r} 做线性层）
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_bag(
        self, char_ids: torch.Tensor, pos_e1: torch.Tensor, pos_e2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
        - logits: (num_classes,)
        - attn: (num_classes, n_inst)  每个 class 在实例维度 softmax
        """
        H = self.pcnn.forward_pooled(char_ids, pos_e1, pos_e2)  # (n, hidden)
        E = torch.tanh(self.W(H))  # (n, att_dim)
        # scores: (n, C) = E @ u^T
        scores = E @ self.u.t()
        # 对实例维度做 softmax，得到每个 class 的 attention 分布
        attn = F.softmax(scores.transpose(0, 1), dim=1)  # (C, n)
        # 每个 class 的 bag 表示
        H_bag = attn @ H  # (C, hidden)
        logits = self.classifier(H_bag)  # (C, C)
        # 取对角：每个 class r 只取自己的 logit_r
        logits = torch.diagonal(logits, 0)  # (C,)
        return logits, attn
