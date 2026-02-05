# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp

from scipy.stats import pearsonr

from models import *
from data import *

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter
import csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='japan', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='japan-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='cola_gnn', choices=['cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn','st_gat'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=1, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=3,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=200, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')

# Additional arguments for SpatiotemporalTransformerGAT
ap.add_argument('--d_model', type=int, default=32, help='transformer model dimension')
ap.add_argument('--nhead', type=int, default=4, help='number of transformer attention heads')
ap.add_argument('--num_transformer_layers', type=int, default=2, help='number of transformer layers')
ap.add_argument('--dim_feedforward', type=int, default=128, help='transformer feedforward dimension')
ap.add_argument('--hidden_dim_gnn', type=int, default=32, help='GNN hidden dimension')
ap.add_argument('--num_gnn_layers', type=int, default=2, help='number of GNN layers')
ap.add_argument('--gat_heads', type=int, default=1, help='number of GAT attention heads')
ap.add_argument('--beta', type=float, default=0.3, help='weight for dynamic adjacency')
ap.add_argument('--clamp_adj', action='store_true', default=True, help='clamp adjacency values to [0,1]')
ap.add_argument('--threshold_adj', type=float, default=None, help='threshold for adjacency values')

args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dcrnn_model import *
from spatiotemporal_transformer_gat import SpatiotemporalTransformerGAT


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = args.cuda and torch.cuda.is_available() 
args.cuda = args.gpu is not None and torch.cuda.is_available()

# Create device object for consistent usage throughout the code
if args.cuda:
    try:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    except (AttributeError, AssertionError):
        device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')

# args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    # Remove existing directory first to avoid conflicts
    if os.path.exists(tensorboard_log_dir):
        shutil.rmtree(tensorboard_log_dir, ignore_errors=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

data_loader = DataBasicLoader(args)


if args.model == 'CNNRNN_Res':
    model = CNNRNN_Res(args, data_loader)  
elif args.model == 'RNN':
    model = RNN(args, data_loader)
elif args.model == 'AR':
    model = AR(args, data_loader)
elif args.model == 'ARMA':
    model = ARMA(args, data_loader)
elif args.model == 'VAR':
    model = VAR(args, data_loader)
elif args.model == 'GAR':
    model = GAR(args, data_loader)
elif args.model == 'SelfAttnRNN':
    model = SelfAttnRNN(args, data_loader)
elif args.model == 'lstnet':
    model = LSTNet(args, data_loader)      
elif args.model == 'stgcn':
    model = STGCN(args, data_loader, data_loader.m, 1, args.window, 1)  
elif args.model == 'dcrnn':
    model = DCRNNModel(args, data_loader)   
elif args.model == 'cola_gnn':
    model = cola_gnn(args, data_loader)
elif args.model == 'st_gat':
    model = SpatiotemporalTransformerGAT(args, data_loader)
else: 
    raise LookupError('can not find the model')
 
logger.info('model %s', model)
import sys; sys.stdout.flush()
if args.cuda:
    model.cuda()
print('DEBUG: Creating optimizer...', flush=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params, flush=True)

def evaluate(data_loader, data, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output, _ = model(X)
        
        # Ensure output has proper shape [batch, horizon, nodes]
        if output.dim() == 2:  # If output is [batch, nodes]
            output = output.unsqueeze(1).expand(-1, Y.size(1), -1)
            
        # Compute loss across all prediction steps
        loss_train = F.l1_loss(output, Y)  # Both are [batch, horizon, nodes]
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m)

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx)  # [n_samples, horizon, nodes]
    y_true_mx = torch.cat(y_true_mx)  # [n_samples, horizon, nodes]
    
    # Reshape predictions to match targets
    y_pred_mx = y_pred_mx.reshape(-1, data_loader.m)  # [n_samples*horizon, nodes]
    y_true_mx = y_true_mx.reshape(-1, data_loader.m)  # [n_samples*horizon, nodes]
    
    # Convert to real values
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    pcc_tmp = []
    for k in range(data_loader.m):
        pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
    pcc_states = np.mean(np.array(pcc_tmp)) 
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae

def train(data_loader, data):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        optimizer.zero_grad()
        output, _ = model(X) 
        
        # Compute loss across all prediction steps
        loss_train = F.l1_loss(output, Y) # Both are [batch, horizon, nodes]
        total_loss += loss_train.item()
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples)
 
bad_counter = 0
best_epoch = 0
best_val = 1e+20
try:
    print('begin training')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss}, epoch)
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
       
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:',epoch, time.ctime())
            test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')
            print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')

# Save metrics to CSV
metrics = [mae, std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_filename = os.path.join(project_root, "result", "metrics.csv")
header = ['dataset', 'horizon', 'model', 'mae', 'std_mae', 'rmse', 'rmse_states', 'pcc', 'pcc_states', 'r2', 'r2_states', 'var', 'var_states', 'peak_mae']

# Create result directory if it doesn't exist
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

# Check if file exists or not to write header only once (optional)
try:
    with open(csv_filename, 'x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
except FileExistsError:
    pass

# After evaluation, write a new row with the metrics and parameters
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([args.dataset, args.horizon, args.model, mae, std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae])

print('Final evaluation')
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
