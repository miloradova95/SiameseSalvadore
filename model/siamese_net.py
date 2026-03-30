import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """DenseNet121 backbone with a projection head that outputs L2-normalized embeddings."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Drop the classifier — keep only the feature extractor
        self.backbone = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → (batch, 1024, 1, 1)

        # Projection head: 1024 → 512 → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1024, H, W)
        features = F.relu(features, inplace=True)
        features = self.pool(features)       # (B, 1024, 1, 1)
        features = torch.flatten(features, 1)  # (B, 1024)
        embeddings = self.projection(features)  # (B, embedding_dim)
        return F.normalize(embeddings, p=2, dim=1)  # L2 normalize


class SiameseNet(nn.Module):
    """Siamese network: shared EmbeddingNet applied to two inputs."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)

    def forward(self, anchor: torch.Tensor, pair: torch.Tensor):
        emb_anchor = self.embedding_net(anchor)
        emb_pair = self.embedding_net(pair)
        return emb_anchor, emb_pair

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)
