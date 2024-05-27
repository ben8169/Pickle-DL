from django.db import models
import torch
import torch.nn as nn
import timm
import pickle


class EfficientNetEmbeddingModel(nn.Module):
    def __init__(self):
        super(EfficientNetEmbeddingModel, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = nn.Identity()  # Fully connected layer 제거

    def forward(self, x):
        return self.model(x)

def load_model():
    with open('efficientnet_embedding_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model