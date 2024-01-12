# Learning 

# Libraries
import torch.nn as nn
import numpy as np
import torch
import torch_geometric
from training.loss import *

def testing(model, loader, alpha=0, normalization=None):
    '''
    Function that tests a model and returns either the average losses or the predicted and real pressure values
    It can work both for ANN and GNN models (which require different DataLoaders)
    ------
    model: nn.Model
        e.g., GNN model
    loader: DataLoader
        data loader for dataset
    plot: bool
        if True, returns predicted and real pressure values
        if False, returns average losses
    alpha: float
        smoothness parameter (see loss.py for more info)
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})    
    '''
    model.eval()
    losses = []
    pred = []
    real = []
    
    # retrieve model device (to correctly load data if GPU)
    device = next(model.parameters()).device
        
    with torch.no_grad():
        for batch in loader:
            # if loop is needed to separate pytorch and pyg dataloaders
            if isinstance(loader, torch_geometric.loader.dataloader.DataLoader):
                # Load data to device
                real.append(batch.y)
                batch = batch.to(device)
                
                # GNN model prediction
                out = model(batch)                
                pred.append(out.cpu())


                # loss function = MSE if alpha=0
                loss = smooth_loss(out, batch, alpha=alpha)
                
                
            elif isinstance(loader, torch.utils.data.dataloader.DataLoader):
                # Load data to device
                x, y = batch[0], batch[1]            
                real.append(y)
                x = x.to(device)
                y = y.to(device)
                
                # ANN model prediction
                out = model(x)
                pred.append(out.cpu())

                # MSE loss function
                loss = nn.MSELoss()(out, y)
                
            # Normalization to have more representative loss values
            if normalization is not None:
                loss *= normalization['pressure']
                
            losses.append(loss.cpu().detach())
        
        preds = np.concatenate(pred).reshape(-1,1)
        reals = np.concatenate(real).reshape(-1,1)
    
    return np.array(losses).mean(), preds, reals