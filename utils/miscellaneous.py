import numpy as np
import torch
import yaml
import os
from itertools import product

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''   
    
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    
    # TODO: thorough check with asserts needed 
    # ## I would just output a warning (this way I can still keep the hyperparameters of each algortihm)
    assert set(cfg['algorithms']) == set(cfg['hyperParams'].keys()), "Mismatch between algorithms and hyperparams!"        
        
    return cfg
    
def create_folder_structure(cfg, parent_folder='./results', max_trials=1000):
    '''
    Create folder for storing results. 
    Checks if folder already exists and iteratively tries to create one by adding a suffix. 
    
    Returns path of result folder
    '''
    folder_name = cfg['exp_name']    
    # retrieve here list of models
    algorithms = cfg['algorithms']    
    # add validation == training
    cfg['networks']['validation']=cfg['networks']['training']
    
    create_folder_flag = True
    counter = 0
    while create_folder_flag:
        if counter == 0:
            suffix = ''    
        else:
            suffix = f'{counter:04d}'        
        results_folder =  f'{parent_folder}/{folder_name}{suffix}'
        if not os.path.exists(results_folder):                
            # creating folders
            print(f'Creating folder: {results_folder}')
            for algorithm in algorithms:                    
                os.makedirs(f'{results_folder}/{algorithm}/')      
                os.makedirs(f'{results_folder}/{algorithm}/hist')
                os.makedirs(f'{results_folder}/{algorithm}/models')
                for split in ['training','validation','testing']:                    
                    os.makedirs(f'{results_folder}/{algorithm}/pred/{split}')                    
                    for wds in cfg['networks'][split]:                                        
                        os.makedirs(f'{results_folder}/{algorithm}/pred/{split}/{wds}')
            create_folder_flag = False
        else:
            counter += 1
            if counter > max_trials:
                raise OSError(f"Too many folders for experiment {folder_name}. Try changing it!")
    
    return results_folder
    
def create_folder_structure_MLPvsGNN(cfg, parent_folder='./results', max_trials=1000): 
    '''
    ad hoc solution that has to be removed (e.g., add a split arg to original function for shortcut? 
                                                  long term you must redo all this)
    '''
    folder_name = cfg['exp_name']    
    # retrieve here list of architectures
    algorithms = cfg['algorithms']
    
    create_folder_flag = True
    counter = 0
    while create_folder_flag == True:
        if counter == 0:
            suffix = ''    
        else:
            suffix = f'{counter:04d}'        
        results_folder =  f'{parent_folder}/{folder_name}{suffix}'
        if not os.path.exists(results_folder):                
            # creating folders
            print(f'Creating folder: {results_folder}')
            for wdn in cfg['networks']:
                os.makedirs(f'{results_folder}/{wdn}', exist_ok=True)
                for algorithm in algorithms:                    
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/hist')
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/models')    
                    os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/')
                    for split in ['training','validation','testing']:                    
                        os.makedirs(f'{results_folder}/{wdn}/{algorithm}/pred/{split}')                    
            create_folder_flag = False
        else:
            counter += 1
            if counter > max_trials:
                raise OSError(f"Too many folders for experiment {folder_name}. Try changing it!")
    
    return results_folder


def initalize_random_generators(cfg, count=0):
    '''
    This function initialites the random seeds specified in the config file.
    ------
    cfg: dict
        configuration file obtained with read_config
    count: int
        select seed used for testing
    '''
    # initialize random seeds for reproducibility
    np_seed=cfg['seeds']['np']
    torch_seed=cfg['seeds']['torch']
    
    # initialize random generators for numpy and pytorch
    np.random.seed(np_seed)       
    torch.manual_seed(torch_seed)
    
    return None    

def read_hyperparameters(cfg: dict, model: str):
    '''
    This function selects the hyperparameters specified in the config file.
    returns a list with all hyperparameters combinations
    ------
    cfg: dict
        configuration file obtained with read_config
    model: str
        'GNN' or 'ANN'. 
    '''
    if model == 'GNN':
        hid_channels = cfg['GNN_hyperp']['hid_channels']
        edge_channels = cfg['GNN_hyperp']['edge_channels']
        K = cfg['GNN_hyperp']['K']
        dropout_rate = cfg['GNN_hyperp']['dropout_rate']
        weight_decay = cfg['GNN_hyperp']['weight_decay']
        learning_rate = cfg['GNN_hyperp']['learning_rate']
        batch_size = cfg['GNN_hyperp']['batch_size']
        alpha = cfg['GNN_hyperp']['alpha']
        num_epochs = cfg['GNN_hyperp']['num_epochs']
        
        combinations = list(product(*[hid_channels, edge_channels, K, dropout_rate, weight_decay, learning_rate, batch_size, alpha, num_epochs]))
    
    elif model == 'ANN':
        hid_channels = cfg['ANN_hyperp']['hid_channels']
        hid_layers = cfg['ANN_hyperp']['hid_layers']
        dropout_rate = cfg['ANN_hyperp']['dropout_rate']
        weight_decay = cfg['ANN_hyperp']['weight_decay']
        learning_rate = cfg['ANN_hyperp']['learning_rate']
        batch_size = cfg['ANN_hyperp']['batch_size']
        num_epochs = cfg['ANN_hyperp']['num_epochs']
        
        combinations = list(product(*[hid_channels, hid_layers, dropout_rate, weight_decay, learning_rate, batch_size, num_epochs]))
        
    else:
        raise("model must be either 'GNN' or 'ANN'")

    return combinations