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

class VideoAugmentation:
    def __init__(self, is_train=True):
        self.is_train = is_train
        
    def __call__(self, frames):
        frames = np.array(frames).copy()
        
        # Convert to float and normalize
        frames = frames.astype(np.float32) / 255.0
        
        if self.is_train:
            # Apply random brightness and contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-0.2, 0.2)  # Brightness
                frames = np.clip(alpha * frames + beta, 0, 1)
            
            # Random horizontal flip (same for all frames)
            if random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()
            
            # Color jitter
            if random.random() > 0.7:
                # Random color channel adjustment
                for c in range(3):
                    factor = random.uniform(0.8, 1.2)
                    frames[:, :, :, c] = np.clip(frames[:, :, :, c] * factor, 0, 1)
        
        # Convert to torch tensor
        frames = np.transpose(frames, (0, 3, 1, 2)).copy()  # (N, H, W, C) -> (N, C, H, W)
        return torch.FloatTensor(frames)