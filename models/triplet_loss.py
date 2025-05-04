import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, distance="euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        if self.distance == "euclidean":
            d_pos = F.pairwise_distance(anchor, positive)
            d_neg = F.pairwise_distance(anchor, negative)
        elif self.distance == "cosine":
            d_pos = 1 - F.cosine_similarity(anchor, positive)
            d_neg = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance}")

        # 计算三元组损失：max(d_pos - d_neg + margin, 0)
        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()  # 返回批量的平均损失