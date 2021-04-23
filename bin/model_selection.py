#!/usr/bin/env python3

import torch
import argparse
import torchvision
import os
import optuna
import joblib
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import seaborn as sns
import numpy as np
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import gc
import time

tensor = (3,256, 256)


# VGG16trained model Architecture

class VGG16Model(torch.nn.Module):
    """
    VGG16 pretrained model with additional projection head for transfer learning
        
    """
    def __init__(self, layer):
        super(VGG16Model, self).__init__()
        
        self.layer = layer
        self.body = torchvision.models.vgg16(pretrained=True).features
        
        for name,child in self.body.named_children():
            if name == self.layer:
               
                break
            for params in child.parameters():
                params.requires_grad = False
            
      
        self.in_feat = self.get_dim(tensor)

        self.head = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_features=self.in_feat, out_features=512, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(in_features= 512, out_features=64, bias=True), #not such a steep jump
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(64,5)
        )

    def get_dim(self, input_size):
        bs = 1
        ip = torch.rand(bs, *input_size)
        output = self.body(ip)
        op_view = output.view(bs,-1)
        return op_view.shape[-1]
        
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


###################################################################################################

# Early stopping implementation

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='early_stopping.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0   
    
    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), self.path)
        self.vall_loss_min = val_loss
        