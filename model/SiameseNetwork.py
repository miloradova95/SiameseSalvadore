import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Use pretrained DenseNet121
        backbone = models.densenet121(pretrained=True)

        # Remove classifier, keep features
        self.feature_extractor = backbone.features  # This is all conv layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Number of features output by DenseNet before classifier
        num_features = backbone.classifier.in_features

        # Projection head → embedding
        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def extract_features(self, x):
        """Return convolutional feature maps before global pooling."""
        return self.feature_extractor(x)

    def forward_features(self, x):
        """Return normalized embedding and the last convolutional feature map."""
        features = self.extract_features(x)
        pooled = self.pool(features).view(features.size(0), -1)
        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb, features

    def forward_once(self, x):
        emb, _ = self.forward_features(x)
        return emb

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        return emb1, emb2

    def compute_sfam(self, feat1, feat2, normalize=True, eps=1e-8):
        """Compute a Similar Feature Activation Map between two feature maps.

        The outputs are normalized channel-wise cosine similarities at each spatial location.
        """
        if feat1.shape != feat2.shape:
            raise ValueError("Feature maps must have the same shape for SFAM computation")

        similarity = (feat1 * feat2).sum(dim=1, keepdim=True)
        norm1 = torch.norm(feat1, dim=1, keepdim=True)
        norm2 = torch.norm(feat2, dim=1, keepdim=True)
        sfam = similarity / (norm1 * norm2 + eps)

        if normalize:
            sfam = (sfam + 1.0) / 2.0
        return sfam

    def forward_with_sfam(self, img1, img2, output_size=None):
        """Forward two images and compute their Similar Feature Activation Map."""
        emb1, feat1 = self.forward_features(img1)
        emb2, feat2 = self.forward_features(img2)
        sfam = self.compute_sfam(feat1, feat2)

        if output_size is not None:
            sfam = F.interpolate(sfam, size=output_size, mode="bilinear", align_corners=False)

        return emb1, emb2, sfam
    
    def get_embedding(self, x):
        return self.forward_once(x)
    
