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

        # Use pretrained backbone
        backbone = models.densenet121(pretrained=True)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Projection head → embedding
        self.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)
        return emb1, emb2
    
