"""
Data Loading Module for MSAGAT-Net

Handles loading, preprocessing, and batching of spatio-temporal time series data
for graph-based forecasting.

Main Class:
    DataBasicLoader: Loads time series and adjacency data, handles normalization,
                     and provides batch iterators for training/validation/testing.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from typing import Optional, Dict, List, Tuple, Iterator


class DataBasicLoader:
    """
    Data loader for spatio-temporal forecasting datasets.
    
    Handles:
        - Loading raw time series data
        - Loading adjacency/similarity matrices
        - Min-max normalization based on training data
        - Train/validation/test splitting
        - Batch generation for training
    
    Args:
        args: Configuration object with attributes:
            - dataset (str): Dataset name (loads from data/{dataset}.txt)
            - sim_mat (str): Adjacency matrix name (loads from data/{sim_mat}.txt)
            - window (int): Input window size
            - horizon (int): Prediction horizon
            - train (float): Training data fraction
            - val (float): Validation data fraction
            - cuda (bool): Whether to use GPU
            - save_dir (str): Directory for saving/loading
            - extra (str, optional): External data directory
            - label (str, optional): Label file for external data
    
    Attributes:
        rawdat: Raw data array [timesteps, nodes]
        dat: Normalized data array [timesteps, nodes]
        adj: Adjacency matrix tensor [nodes, nodes]
        train: Tuple of (X, Y) tensors for training
        val: Tuple of (X, Y) tensors for validation  
        test: Tuple of (X, Y) tensors for testing
        max, min: Normalization parameters
        m: Number of nodes
        n: Number of timesteps
    """
    
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon
        self.d = 0
        self.add_his_day = False
        self.save_dir = args.save_dir
        
        # Load raw data
        data_path = os.path.join("data", f"{args.dataset}.txt")
        self.rawdat = np.loadtxt(data_path, delimiter=',')
        print(f'Data shape: {self.rawdat.shape}')
        
        # Load adjacency matrix if specified
        if args.sim_mat:
            self._load_adjacency(args)

        # Load external data if specified
        if args.extra:
            self._load_external(args)
 
        # Handle 1D data
        if len(self.rawdat.shape) == 1:
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        # Initialize normalized data array
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        print(f'Timesteps: {self.n}, Nodes: {self.m}')

        self.scale = np.ones(self.m)

        # Split indices
        train_end = int(args.train * self.n)
        val_end = int((args.train + args.val) * self.n)
        
        # Compute normalization parameters and normalize data
        self._compute_normalization(train_end, val_end)
        
        # Create train/val/test splits
        self._create_splits(train_end, val_end)
        
        print(f'Train/Val/Test sizes: {len(self.train[0])}, {len(self.val[0])}, {len(self.test[0])}')
    
    def _load_adjacency(self, args):
        """Load adjacency/similarity matrix."""
        adj_path = os.path.join("data", f"{args.sim_mat}.txt")
        self.adj = torch.Tensor(np.loadtxt(adj_path, delimiter=','))
        self.orig_adj = self.adj.clone()
        self.degree_adj = torch.sum(self.orig_adj, dim=-1)
        self.adj = Variable(self.adj)
        
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()
            self.degree_adj = self.degree_adj.cuda()
    
    def _load_label_file(self, filename: str) -> Tuple[Dict, int]:
        """Load label mapping from CSV file."""
        labelfile = pd.read_csv(os.path.join("data", f"{filename}.csv"), header=None)
        label = {labelfile.iloc[i, 0]: labelfile.iloc[i, 1] for i in range(len(labelfile))}
        return label, len(labelfile)

    def _load_external(self, args):
        """Load external adjacency information."""
        label, label_num = self._load_label_file(args.label)
        extra_dir = os.path.join("data", args.extra)
        files = os.listdir(extra_dir)
        
        extra_adj_list = []
        for fname in files:
            snapshot = pd.read_csv(os.path.join(extra_dir, fname), header=None)
            extra_adj = np.zeros((label_num, label_num))
            for j in range(len(snapshot)):
                src, dst, val = snapshot.iloc[j, 0], snapshot.iloc[j, 1], snapshot.iloc[j, 2]
                extra_adj[label[src], label[dst]] = val
            extra_adj_list.append(extra_adj)
            
        extra_adj = torch.Tensor(np.array(extra_adj_list))
        print(f'External information shape: {extra_adj.shape}')
        self.external = Variable(extra_adj)
        
        if args.cuda:
            self.external = extra_adj.cuda()

    def _compute_normalization(self, train_end: int, val_end: int):
        """Compute min-max normalization parameters from training data."""
        self.train_set = range(self.P + self.h - 1, train_end)
        self.valid_set = range(train_end, val_end)
        self.test_set = range(val_end, self.n)
        
        # Get training data for normalization
        tmp_train = self._batchify(self.train_set, self.h, useraw=True)
        train_mx = torch.cat((tmp_train[0][:, 0, :], tmp_train[1]), 0).numpy()
        
        # Compute min/max from training data
        self.max = np.max(train_mx, axis=0)
        self.min = np.min(train_mx, axis=0)
        
        # Compute peak threshold for evaluation
        self.peak_thold = np.mean(train_mx, axis=0)
        
        # Normalize all data
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)
        print(f'Normalized data shape: {self.dat.shape}')
         
    def _create_splits(self, train_end: int, val_end: int):
        """Create train/validation/test data splits."""
        self.train = self._batchify(self.train_set, self.h)
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        
        # Handle edge case where train == val
        if train_end == val_end:
            self.val = self.test
 
    def _batchify(self, idx_set, horizon: int, useraw: bool = False) -> List[torch.Tensor]:
        """
        Create input-output pairs for a set of indices.
        
        Args:
            idx_set: Indices to process
            horizon: Prediction horizon
            useraw: Whether to use raw (unnormalized) data
            
        Returns:
            List of [X, Y] tensors where:
                X: Input windows [samples, window, nodes]
                Y: Target values [samples, nodes]
        """
        n = len(idx_set)
        
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P + 1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        
        for i, idx in enumerate(idx_set):
            end = idx - self.h + 1
            start = end - self.P
            
            if useraw:
                X[i, :self.P, :] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i, :] = torch.from_numpy(self.rawdat[idx, :])
            else:
                his_window = self.dat[start:end, :]
                
                if self.add_his_day:
                    if idx > 51:
                        his_day = self.dat[idx - 52:idx - 51, :]
                    else:
                        his_day = np.zeros((1, self.m))
                    his_window = np.concatenate([his_day, his_window])
                    X[i, :self.P + 1, :] = torch.from_numpy(his_window)
                else:
                    X[i, :self.P, :] = torch.from_numpy(his_window)
                    
                Y[i, :] = torch.from_numpy(self.dat[idx, :])
                
        return [X, Y]

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Args:
            data: Normalized data array
            
        Returns:
            Data in original scale
        """
        return data * (self.max - self.min + 1e-12) + self.min

    def get_batches(self, data: List[torch.Tensor], batch_size: int, 
                    shuffle: bool = True) -> Iterator:
        """
        Generate batches from data.
        
        Args:
            data: List of [X, Y] tensors
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            List of [X, Y, indices] where X, Y are batch tensors
        """
        inputs, targets = data[0], data[1]
        length = len(inputs)
        
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
            
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            
            X = inputs[excerpt, :]
            Y = targets[excerpt, :]
            
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
                
            yield [Variable(X), Variable(Y), index]
            start_idx += batch_size


__all__ = ['DataBasicLoader']
