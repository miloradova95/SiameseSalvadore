import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        """
        Compute triplet loss.

        Args:
            anchor_emb: Embeddings for anchor images [batch_size, emb_dim]
            positive_emb: Embeddings for positive images [batch_size, emb_dim]
            negative_emb: Embeddings for negative images [batch_size, emb_dim]

        Returns:
            loss: Scalar triplet loss
        """
        # Euclidean distances
        dist_ap = F.pairwise_distance(anchor_emb, positive_emb)
        dist_an = F.pairwise_distance(anchor_emb, negative_emb)

        # Triplet loss: max(0, dist_ap - dist_an + margin)
        loss = torch.mean(torch.clamp(dist_ap - dist_an + self.margin, min=0.0))
        return loss