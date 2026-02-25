import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score,auc, precision_recall_fscore_support, confusion_matrix, roc_curve


# Enhanced EfficientFace model for deepfake detection - MATCHES TRAINED CHECKPOINT
class MyModel(nn.Module):
    def __init__(self, num_frames=16):
        super(MyModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        
        # Temporal feature extraction with 3D convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1280, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )
        
        # Channel and spatial attention
        self.channel_attention = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Sigmoid()
        )
        
        # Classification head - MATCHES CHECKPOINT EXACTLY
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),           # 0: weight/bias
            nn.LayerNorm(256),             # 1: weight/bias
            nn.ReLU(),                     # 2: no params
            nn.Dropout(0.5),               # 3: no params
            nn.Linear(256, 128),           # 4: weight/bias
            nn.LayerNorm(128),             # 5: weight/bias
            nn.ReLU(),                     # 6: no params
            nn.Dropout(0.3),               # 7: no params
            nn.Linear(128, 64),            # 8: weight/bias
            nn.LayerNorm(64),              # 9: weight/bias
            nn.ReLU(),                     # 10: no params
            nn.Dropout(0.2),               # 11: no params
            nn.Linear(64, 1)               # 12: weight/bias
        )
        
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        # Process frames independently
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)  # (batch_size * num_frames, 1280)
        
        # Reshape to 3D volume
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 1280)
        features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1280, num_frames, 1, 1)
        
        # Temporal convolutions
        temporal_features = self.temporal_conv(features)
        
        # Apply temporal attention
        temporal_weights = self.temporal_attention(temporal_features)
        temporal_features = temporal_features * temporal_weights
        
        # Apply channel and spatial attention
        channel_weights = self.channel_attention(temporal_features)
        spatial_weights = self.spatial_attention(temporal_features)
        attention_weights = channel_weights * spatial_weights
        attended_features = temporal_features * attention_weights
        
        # Global pooling
        pooled_features = torch.mean(attended_features, dim=(2, 3, 4))
        
        # Classification
        output = self.classifier(pooled_features)
        return output


from .face_extractor import FaceExtractor
