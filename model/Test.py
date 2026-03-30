import torch
from model.SiameseNetwork import SiameseNetwork
from model.ContrastiveLoss import ContrastiveLoss
from preprocessing.transforms import get_train_transforms
from preprocessing.helpers import get_dataloader

def test_successful():
    print("Test successful!")
    
test_successful()