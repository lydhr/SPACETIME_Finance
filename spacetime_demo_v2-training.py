#!/usr/bin/env python
# coding: utf-8

# ### Setup

# In[4]:


# General Setup
import os
import sys

from os.path import join

# Local imports from spacetime
project_dir = './spacetime'
sys.path.insert(0, os.path.abspath(project_dir)) 


# In[5]:


# The data science trinity, we might not use all of them
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


from omegaconf import OmegaConf


# In[9]:


from utils.logging import print_config


# In[10]:


# Hacky args via an OmegaConf config
args = """
seed: 42
"""
args = OmegaConf.create(args)
print_config(args)


# ### Part 1.1 Data + Task
# 
# #### Load and setup data
# 
# We'll start out with predicting closing prices for the S&P 500.

# In[61]:


#!pip install yfinance #yahoo market data; finance api


# In[12]:


import yfinance as yf


# In[13]:


yf_data = yf.Ticker('^GSPC')  # Ticker for S&P 500


# In[14]:


data = yf_data.history(period='max', start='1993-01-01',  # wow 30 years of data
                       auto_adjust=True)  


# In[15]:


df = pd.DataFrame(data).reset_index()


# In[17]:


# Visualize closing prices
plt.plot(df['Date'], df['Close'])
plt.show()


# #### Set lag and horizon prediction task

# In[18]:


args.lag = 84          # We'll use the prior 12 calendar weeks as inputs
args.horizon = 20      # We'll then try to predict out the next 20 available days (4ish working weeks)
args.target = 'Close'  # Pick one feature to forecast 

# Windows of samples
samples = [w.to_numpy() for w in df[args.target].rolling(window=args.lag + args.horizon)][args.lag + args.horizon - 1:]
# Dates for each sample
dates = [w for w in df['Date'].rolling(window=args.lag + args.horizon)][args.lag + args.horizon - 1:]


# In[21]:


import datetime

test_year = 2022  
test_date = datetime.date(test_year, 1, min(args.horizon, 30)) 

## Convert 'Date' to datetime object
df['Date'] = pd.to_datetime(df['Date']).dt.date

## Find indices corresponding to test year dates
test_ix = len(dates) - df[df['Date'] >= test_date].shape[0]


# In[22]:


# Check that the horizon dates are roughly in 2022
dates[test_ix][args.lag:args.lag + args.horizon]


# #### Create training and validation splits

# In[23]:


def train_val_split(data_indices, val_ratio=0.1):
    train_ratio = 1 - val_ratio
    last_train_index = int(np.round(len(data_indices) * train_ratio))
    return data_indices[:last_train_index], data_indices[last_train_index:]


# In[24]:


# Split data indices for train and val sets
train_indices, val_indices = train_val_split(np.arange(test_ix))
train_samples = np.array(samples[:val_indices[0]])
val_samples = np.array(samples[val_indices[0]:val_indices[-1]])

# Sanity check the splits by plotting the last horizon term in each sample
ix = -1
plt.plot(train_samples[:, ix], label=f'{args.target} (train)', alpha=1)
plt.plot(np.arange(len(train_samples), len(train_samples) + len(val_samples)), 
         val_samples[:, ix], label=f'{args.target} (val)', alpha=1)
plt.legend()
plt.show()


# #### Capture the above in PyTorch datasets and dataloaders
# 
# 

# In[25]:


import copy
import torch
from torch.utils.data import Dataset, DataLoader


# In[26]:


class YahooStockPriceDataset(Dataset):
    def __init__(self, data: np.array, lag: int, horizon: int):
        super().__init__()
        self.data_x = torch.tensor(data).unsqueeze(-1).float()
        self.data_y = copy.deepcopy(self.data_x[:, -horizon:, :])
        
        self.lag = lag
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        x[-self.horizon:] = 0  # Mask input horizon terms
        return x, y, (self.lag, self.horizon)
    
    # For simplicity, we just keep these as identities.
    # But we could imagine some kind of data transformation / scaling for the inputs
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x
        


# In[27]:


# Function to load dataloaders for train, val, and test splits
def load_data(df: pd.DataFrame, 
              lag: int, 
              horizon: int, 
              target: str, 
              val_ratio: float,
              test_year_month_day: list[int], 
              **dataloader_kwargs: any):
    
    # Convert day-wise data into sequences of lag + horizon terms
    samples = [w.to_numpy() for w in df[target].rolling(window=lag + horizon)][lag + horizon - 1:]
    dates   = [w for w in df['Date'].rolling(window=lag + horizon)][lag + horizon - 1:]
    
    # Set aside test samples by date
    test_date = datetime.date(*test_year_month_day)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    test_ix = len(dates) - df[df['Date'] >= test_date].shape[0]
    test_samples = np.array(samples[test_ix:])
    
    # Get training + validation samples
    train_indices, val_indices = train_val_split(np.arange(len(dates[:test_ix])), val_ratio)
    train_samples = np.array(samples[:val_indices[0]])
    val_samples = np.array(samples[val_indices[0]:val_indices[-1]])
    
    # PyTorch datasets and dataloaders
    datasets = [YahooStockPriceDataset(_samples, lag, horizon)
                for _samples in [train_samples, val_samples, test_samples]]
    
    dataloaders = [DataLoader(dataset, shuffle=True if ix == 0 else False, **dataloader_kwargs)
                   for ix, dataset in enumerate(datasets)]
    return dataloaders


# In[28]:


# Function to visualize samples over time
def visualize_data(dataloaders, sample_idx, sample_dim=0,
                   splits=['train', 'val', 'test'], title=None):
    assert len(splits) == len(dataloaders)
    start_idx = 0
    for idx, split in enumerate(splits):
        y = dataloaders[idx].dataset.data_x[:, sample_idx, sample_dim]
        x = np.arange(len(y)) + start_idx
        plt.plot(x, y, label=split)
        start_idx += len(x)
    plt.title(title)
    plt.legend()
    plt.show()


# #### Data + task setup via training configs

# In[29]:


# Again we use OmegaConf bc it's great
dataset_configs = f"""
lag: {args.lag}
horizon: {args.horizon}
target: Close
val_ratio: 0.1
test_year_month_day:
- 2021
- 1
- 1
"""
dataset_configs = OmegaConf.create(dataset_configs)  


# In[30]:


dataloader_configs = """
batch_size: 32
num_workers: 2
pin_memory: true
"""
dataloader_configs = OmegaConf.create(dataloader_configs)


# In[31]:


# Load and visualize data
torch.manual_seed(args.seed)
dataloaders = load_data(df, **dataset_configs, **dataloader_configs)
train_loader, val_loader, test_loader = dataloaders

visualize_data(dataloaders, sample_idx=0)


# ### Part 1.2 SpaceTime Model
# 
# We'll now define a SpaceTime model. To do so, we specify individual config files that determine individual components such as the input layer (*i.e.,* the embedding layer), the encoder layers, the decoder layer, and the final output layer.
# 
# We explicitly write out the configs below, but they can also be found as `.yaml` files in `spacetime/configs/model/` (see the `spacetime/README.md` for more details).

# In[32]:


# We've got 4 main components to specify: 
# 1. The embedding / input projection (e.g., an MLP)
# 2. The encoder block ("open-loop" / convolutional SpaceTime SSMs go here)
# 3. The decoder block ("closed-loop" / recurrent SpaceTime SSMs go here)
# 4. The output projection (e.g., an MLP)

config_dir = 'spacetime/configs/'


# In[33]:


embedding_config = """
method: repeat
kwargs:
  input_dim: 1
  embedding_dim: null
  n_heads: 4
  n_kernels: 32
"""
embedding_config = OmegaConf.create(embedding_config)


# In[34]:


encoder_config = """
blocks:
- input_dim: 128
  pre_config: 'ssm/preprocess/residual'
  ssm_config: 'ssm/companion_preprocess'
  mlp_config: 'mlp/default'
  skip_connection: true
  skip_preprocess: false
"""
encoder_config = OmegaConf.create(encoder_config)


# In[35]:


decoder_config = """
blocks:
- input_dim: 128
  pre_config: 'ssm/preprocess/none'
  ssm_config: 'ssm/closed_loop/companion'
  mlp_config: 'mlp/identity'
  skip_connection: false
  skip_preprocess: false
"""
decoder_config = OmegaConf.create(decoder_config)


# In[36]:


output_config = """
input_dim: 128
output_dim: 1
method: mlp
kwargs:
  input_dim: 128
  output_dim: 1
  activation: gelu
  dropout: 0.2
  layernorm: false
  n_layers: 1
  n_activations: 1
  pre_activation: true
  input_shape: bld
  skip_connection: false
  average_pool: null
"""
output_config = OmegaConf.create(output_config)


# #### Use configs to make a SpaceTime neural net

# In[37]:


from model.network import SpaceTime
from setup import seed_everything


# In[38]:


# Initialize SpaceTime encoder and decoder preprocessing, SSM, and MLP components
# - These are referenced as paths in the above encoder and decoder configs
def init_encoder_decoder_config(config, config_dir):
    for ix, _config in enumerate(config['blocks']):
        # Load preprocess kernel configs
        c_path = join(config_dir, f"{_config['pre_config']}.yaml")
        _config['pre_config'] = OmegaConf.load(c_path)
        # Load SSM kernel configs
        c_path = join(config_dir, f"{_config['ssm_config']}.yaml")
        _config['ssm_config'] = OmegaConf.load(c_path)
        # Load MLP configs
        c_path = join(config_dir, f"{_config['mlp_config']}.yaml")
        _config['mlp_config'] = OmegaConf.load(c_path)
    return config


# In[39]:


encoder_config = init_encoder_decoder_config(encoder_config, join(config_dir, 'model'))
decoder_config = init_encoder_decoder_config(decoder_config, join(config_dir, 'model'))


# In[40]:


# Initialize SpaceTime model
model_configs = {
    'embedding_config': embedding_config,
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'output_config': output_config,
    'lag': dataset_configs.lag,
    'horizon': dataset_configs.horizon
}
seed_everything(args.seed)

model = SpaceTime(**model_configs)


# ##### View architecture
# We can either display the SpaceTime model as a PyTorch ``nn.Module`` object, or view the `OmegaConf` config behind it. 

# In[41]:


from utils.config import print_config  # View OmegaConf configs


# In[43]:


print_config(model_configs)  # OmegaConf config view 
# (This might render as black text, not that useful here rip)


# ### Part 1.3 SpaceTime Model Training
# 
# We'll now specify the training configs and train our model

# In[44]:


from loss import get_loss
from data_transforms import get_data_transforms
from optimizer import get_optimizer, get_scheduler
from setup.configs.optimizer import get_optimizer_config, get_scheduler_config

from train import train_model, evaluate_model, plot_forecasts


# #### Specify training args

# In[45]:


# Hacky training args, see the README or /spacetime/setup/args.py for details
arg_config = f"""
lag: {dataset_configs.lag}
horizon: {dataset_configs.horizon}
features: S
lr: 1e-3
weight_decay: 1e-4
dropout: 0.25
criterion_weights:
- 10
- 1
- 10
optimizer: adamw
scheduler: timm_cosine
max_epochs: 500
early_stopping_epochs: 20
data_transform: mean
loss: informer_rmse
val_metric: informer_rmse
seed: 42
dataset: sp500
variant: null
model: SpaceTime
"""
class Args():
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
            
args = Args(OmegaConf.create(arg_config))
# GPU
args.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

# These others are not super important
args.checkpoint_dir = './checkpoints'
args.log_dir = './log_dir'
args.variant = None
args.no_wandb = True
args.dataset_type = 'informer'  # for standard forecasting
args.log_epoch = 1000


# #### Initialize optimizer, scheduler, criterions, data transforms

# In[47]:


seed_everything(args.seed)
model = SpaceTime(**model_configs)  # Reset model from here

model.set_lag(args.lag)
model.set_horizon(args.horizon)
    
# Initialize optimizer and scheduler
optimizer = get_optimizer(model, get_optimizer_config(args, config_dir))
scheduler = get_scheduler(model, optimizer, get_scheduler_config(args, config_dir))
    
# Loss objectives
criterions = {name: get_loss(name) for name in ['rmse', 'mse', 'mae']}
eval_criterions = criterions
for name in ['rmse', 'mse', 'mae']:
    eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')
    
# Data transforms, e.g., normalization
input_transform, output_transform = get_data_transforms(args.data_transform, args.lag)


# #### Train model

# In[48]:


from setup import initialize_experiment


# In[49]:


initialize_experiment(args, experiment_name_id='',
                      best_train_metric=1e10, 
                      best_val_metric=1e10)


# In[50]:


# Actually train model
splits = ['train', 'val', 'test']
dataloaders_by_split = {split: dataloaders[ix] 
                        for ix, split in enumerate(splits)}

model = train_model(model, optimizer, scheduler, dataloaders_by_split, 
                    criterions, max_epochs=args.max_epochs, config=args, 
                    input_transform=input_transform,
                    output_transform=output_transform,
                    val_metric=args.val_metric, wandb=None, 
                    return_best=True, early_stopping_epochs=args.early_stopping_epochs) 


# ### Evaluate model

# In[51]:


from dataloaders import get_evaluation_loaders
from train.evaluate import plot_forecasts


# In[52]:


eval_splits = ['eval_train', 'val', 'test']
eval_loaders = get_evaluation_loaders(dataloaders, batch_size=dataloader_configs.batch_size)
eval_loaders_by_split = {split: eval_loaders[ix] for ix, split in
                         enumerate(eval_splits)}
model, log_metrics, total_y = evaluate_model(model, dataloaders=eval_loaders_by_split, 
                                             optimizer=optimizer, scheduler=scheduler, 
                                             criterions=eval_criterions, config=args,
                                             epoch=args.best_val_metric_epoch, 
                                             input_transform=input_transform, 
                                             output_transform=output_transform,
                                             val_metric=args.val_metric, wandb=None,
                                             train=False)
n_plots = len(splits) # train, val, test
fig, axes = plt.subplots(1, n_plots, figsize=(6.4 * n_plots, 4.8))

plot_forecasts(total_y, splits=eval_splits, axes=axes)


# In[53]:


eval_splits = ['eval_train', 'val', 'test']
eval_loaders = get_evaluation_loaders(dataloaders, batch_size=dataloader_configs.batch_size)
eval_loaders_by_split = {split: eval_loaders[ix] for ix, split in
                         enumerate(eval_splits)}
model, log_metrics, total_y = evaluate_model(model, dataloaders=eval_loaders_by_split, 
                                             optimizer=optimizer, scheduler=scheduler, 
                                             criterions=eval_criterions, config=args,
                                             epoch=args.best_val_metric_epoch, 
                                             input_transform=input_transform, 
                                             output_transform=output_transform,
                                             val_metric=args.val_metric, wandb=None,
                                             train=False)
n_plots = len(splits) # train, val, test
fig, axes = plt.subplots(1, n_plots, figsize=(6.4 * n_plots, 4.8))

plot_forecasts(total_y, splits=eval_splits, axes=axes)


# In[ ]:




