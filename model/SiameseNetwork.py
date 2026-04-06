import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

        # Number of features output by DenseNet before classifier
        num_features = backbone.classifier.in_features

        # Projection head → embedding
        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        # DenseNet features output: [B, C, H, W]
        x = self.feature_extractor(x)
        # Global average pooling (to flatten to [B, num_features])
        x = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        # Projection head
        x = self.fc(x)
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        return emb1, emb2
    
    def get_embedding(self, x):
        return self.forward_once(x)
    
