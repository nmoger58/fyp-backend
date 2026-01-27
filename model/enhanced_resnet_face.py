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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score,auc, precision_recall_fscore_support, confusion_matrix, roc_curve
from torchvision.models.efficientnet import EfficientNet_B0_Weights

# Enhanced EfficientFace model with refined improvements (Version 3 - Optimized for thesis)
class EnhancedEfficientFace(nn.Module):
    def __init__(self, num_frames=16):
        super(EnhancedEfficientFace, self).__init__()
        
        # Load pretrained model với weights thay vì pretrained
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        
        # Temporal feature extraction với lightweight temporal attention
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
        # Lightweight temporal attention (new)
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )
        
        # Attention mechanism (channel + lightweight spatial)
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
        
        # Classifier với layer normalization và điều chỉnh dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Thay BatchNorm bằng LayerNorm cho stability
            nn.ReLU(),
            nn.Dropout(0.6),  # Tăng dropout để chống overfit
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        # Xử lý từng frame độc lập
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)  # (batch_size * num_frames, 1280)
        
        # Reshape sang 3D volume
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 1280)
        features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1280, num_frames, 1, 1)
        
        # Temporal convolutions
        temporal_features = self.temporal_conv(features)
        
        # Áp dụng temporal attention (new)
        temporal_weights = self.temporal_attention(temporal_features)
        temporal_features = temporal_features * temporal_weights
        
        # Áp dụng channel + spatial attention
        channel_weights = self.channel_attention(temporal_features)
        spatial_weights = self.spatial_attention(temporal_features)
        attention_weights = channel_weights * spatial_weights
        attended_features = temporal_features * attention_weights
        
        # Global pooling
        pooled_features = torch.mean(attended_features, dim=(2, 3, 4))
        
        # Classification
        output = self.classifier(pooled_features)
        return output