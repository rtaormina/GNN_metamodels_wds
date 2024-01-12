# Learning 

# Libraries
import time
import torch.nn as nn
import numpy as np
import torch
import torch_geometric
import torch.optim as optim

from training.loss import *
from training.test import *
from utils.visualization import *


def train(model, loader, optimizer, alpha=0, normalization=None):
    '''
    Function that trains a model for one iteration
    It can work both for ANN and GNN models (which require different DataLoaders)
    ------
    model: nn.Model
        e.g., GNN model
    loader: DataLoader
        data loader for dataset
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})    
    '''
    model.train()
    losses = []
        
    for batch in loader:
        if isinstance(loader, torch_geometric.loader.dataloader.DataLoader):
            # Model prediction
            preds = model(batch)
			
            # loss function = MSE if alpha=0
            loss = smooth_loss(preds, batch, alpha=alpha)
            
        elif isinstance(loader, torch.utils.data.dataloader.DataLoader):
            x, y = batch[0], batch[1]
            preds = model(x)
            
            # MSE loss function
            loss = nn.MSELoss()(preds, y)
        
        # Normalization to have more representative loss values
        if normalization is not None:
            loss *= normalization['pressure']
            
        losses.append(loss.detach())
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return np.array(losses).mean()


def training(model, optimizer, train_loader, val_loader, test_loader, 
             n_epochs, patience=10, report_freq=10, alpha=0, lr_rate=10, lr_epoch=50, normalization=None):
    '''
    Training function which returns the training and validation losses over the epochs
    Learning rate scheduler and early stopping routines working correctly
    ------
    model: nn.Model
        e.g., GNN model
    optimizer: torch.optim
        model optimizer (e.g., Adam, SGD)
    *_loader: DataLoader
        data loader for training, validation and testing
    n_epochs: int
        maximum number of total epochs
    patience: int
        number of subsequent occurrences where the validation loss is increasing
    report_freq: int
        printing interval
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})
    '''
    #create vectors for the training and validation loss
    train_losses = []
    val_losses = []
    
    #initialize early stopping variable
    early_stop = 0
    
    #start measuring time
    start_time = time.time()
    
    # torch.autograd.set_detect_anomaly(True)
	
    for epoch in range(1, n_epochs+1):
        # Model training
        train_loss = train(model, train_loader, optimizer, alpha=alpha, normalization=normalization)

        # Model validation
        val_loss = test(model, val_loader, alpha=alpha, normalization=normalization)
            
        train_losses.append(train_loss)  
        val_losses.append(val_loss) 

        # Print R^2
        if report_freq == 0:
            continue
        elif epoch % report_freq == 0:
            train_R2 = plot_R2(model, train_loader, show=False)
            val_R2 = plot_R2(model, val_loader, show=False)
            test_R2 = plot_R2(model, test_loader, show=False)
            print("epoch:",epoch, "\t loss:", np.round(train_losses[-1],2),
                                  "m\t train R2:", np.round(train_R2,4),
                                  "\t val R2:", np.round(val_R2,4),
                                  "\t test R2:", np.round(test_R2,4))

        # learning rate scheduler
        if epoch%lr_epoch==0:
            learning_rate = optimizer.param_groups[0]['lr']/lr_rate
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print("Learning rate is divided by ", lr_rate, "to:", learning_rate)

        # Routine for early stopping
        if epoch>2 and val_losses[-1]>val_losses[-2]:
            early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                break
        else:
            early_stop = 0

    elapsed_time = time.time() - start_time
    
    return model, train_losses, val_losses, elapsed_time