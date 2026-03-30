import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    L = label * d²  +  (1 - label) * max(margin - d, 0)²

    where d = euclidean distance between embeddings.
    label = 1 → same class (pull together), label = 0 → different class (push apart).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        label = label.float()
        dist = F.pairwise_distance(emb1, emb2)  # Euclidean distance

        positive_loss = label * dist.pow(2)
        negative_loss = (1 - label) * F.relu(self.margin - dist).pow(2)

        return (positive_loss + negative_loss).mean()
