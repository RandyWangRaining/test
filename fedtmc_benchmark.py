"""
================================================================================
FedTMC: Federated Trusted Multi-View Classification (Optimized Version)
================================================================================

Federated version implementation based on TMC (ICLR 2021, TPAMI 2022)

Optimizations:
- Pre-load data to GPU to avoid repeated conversions in each epoch
- Use torch.randperm instead of np.random.permutation
- Reduce CPU-GPU data transfer
- Real-time saving of results + hyperparameters
- Supports classical multi-view datasets (PIE, HandWritten, Scene15, Caltech-101, CUB, Animal, ALOI)

Core Idea:
- Each Client outputs evidence (instead of normal features)
- Server fuses evidence using Dempster-Shafer rule
- Model uncertainty using Dirichlet distribution
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.io as sio
import h5py
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Datasets (Supports both random and stratified splitting)
# ============================================================================

class FedMVDataset:
    """Federated multi-view dataset - Supports both random and stratified splitting"""
    
    def __init__(self, name, views_data, labels, train_ratio=0.8, seed=42, stratified=False):
        self.name = name
        self.seed = seed
        
        # Label processing
        labels = labels.flatten().astype(np.int64)
        if labels.min() == 1:
            labels = labels - 1
        n_labels = len(labels)
        
        # ===== Critical Fix: Check and fix view dimensions =====
        fixed_views = []
        for i, v in enumerate(views_data):
            v = np.array(v)
            if v.shape[0] != n_labels:
                if v.shape[1] == n_labels:
                    print(f"  [Auto-fix] View {i}: transposing {v.shape} -> {v.T.shape}")
                    v = v.T
                else:
                    raise ValueError(
                        f"View {i} shape {v.shape} incompatible with {n_labels} labels."
                    )
            fixed_views.append(v)
        
        # Normalization
        self.views_data = []
        for v in fixed_views:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.views_data.append(scaler.fit_transform(v.astype(np.float64)))
        
        self.labels = labels
        self.num_views = len(self.views_data)
        self.num_samples = len(self.labels)
        self.num_classes = len(np.unique(self.labels))
        self.dims = [v.shape[1] for v in self.views_data]
        
        # Dataset splitting
        if stratified:
            print(f"  [Split Method] Stratified Sampling")
            self._stratified_split(train_ratio, seed)
        else:
            print(f"  [Split Method] Random Shuffle")
            self._random_split(train_ratio, seed)
        
        self._print_split_info()
    
    def _stratified_split(self, train_ratio, seed):
        indices = np.arange(self.num_samples)
        self.train_idx, self.test_idx = train_test_split(
            indices, train_size=train_ratio, stratify=self.labels, random_state=seed
        )
    
    def _random_split(self, train_ratio, seed):
        np.random.seed(seed)
        indices = np.random.permutation(self.num_samples)
        n_train = int(self.num_samples * train_ratio)
        self.train_idx = indices[:n_train]
        self.test_idx = indices[n_train:]
    
    def _print_split_info(self):
        print(f"  [Split] Train: {len(self.train_idx)}, Test: {len(self.test_idx)}")
    
    def __repr__(self):
        return (f"FedMVDataset({self.name}): {self.num_views} views, "
                f"{self.num_classes} classes, {self.num_samples} samples, "
                f"dims={self.dims}")


# ============================================================================
# TMC Client (Outputs Evidence)
# ============================================================================

class TMCClient:
    """TMC Client - Outputs evidence"""
    
    def __init__(self, client_id, input_dim, hidden_dim, num_classes,
                 lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus()
        ).to(device)
        
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-5)
        self._evidence = None
    
    def compute_evidence(self, X):
        self.encoder.train()
        evidence = self.encoder(X)
        evidence_to_send = evidence.detach().requires_grad_(True)
        self._evidence = evidence
        return evidence_to_send
    
    def receive_gradient_and_update(self, grad):
        if self._evidence is not None and grad is not None:
            self.optimizer.zero_grad()
            self._evidence.backward(grad)
            self.optimizer.step()
        self._evidence = None
    
    def compute_evidence_eval(self, X):
        self.encoder.eval()
        with torch.no_grad():
            evidence = self.encoder(X)
        return evidence


# ============================================================================
# TMC Server (DS Fusion)
# ============================================================================

class TMCServer:
    """TMC Server - Dempster-Shafer Fusion"""
    
    def __init__(self, num_classes, num_views, fusion_type='DS', 
                 annealing_epochs=10, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
        self.fusion_type = fusion_type
        self.annealing_epochs = annealing_epochs
        self.device = device
        self.current_epoch = 1
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def evidence_to_alpha(self, evidence):
        return evidence + 1
    
    def alpha_to_prob(self, alpha):
        S = alpha.sum(dim=-1, keepdim=True)
        return alpha / S
    
    def alpha_to_uncertainty(self, alpha):
        S = alpha.sum(dim=-1)
        return self.num_classes / S
    
    def ds_combine_two(self, alpha1, alpha2):
        K = self.num_classes
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        
        E1 = alpha1 - 1
        E2 = alpha2 - 1
        
        b1 = E1 / S1.expand(E1.shape)
        b2 = E2 / S2.expand(E2.shape)
        
        u1 = K / S1
        u2 = K / S2
        
        bb = torch.bmm(b1.view(-1, K, 1), b2.view(-1, 1, K))
        bu = torch.mul(b1, u2.expand(b1.shape))
        ub = torch.mul(b2, u1.expand(b1.shape))
        
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag
        
        b_a = (torch.mul(b1, b2) + bu + ub) / ((1 - C).view(-1, 1).expand(b1.shape))
        u_a = torch.mul(u1, u2) / ((1 - C).view(-1, 1).expand(u1.shape))
        
        S_a = K / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        
        return alpha_a
    
    def ds_combine_multi(self, alpha_list):
        alpha_combined = alpha_list[0]
        for i in range(1, len(alpha_list)):
            alpha_combined = self.ds_combine_two(alpha_combined, alpha_list[i])
        return alpha_combined
    
    def avg_combine(self, alpha_list):
        stacked = torch.stack(alpha_list, dim=0)
        return stacked.mean(dim=0)
    
    def compute_kl_loss(self, alpha):
        K = self.num_classes
        beta = torch.ones((1, K), device=self.device)
        
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl
    
    def compute_ce_loss(self, alpha, labels):
        K = self.num_classes
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        
        label = F.one_hot(labels, num_classes=K).float()
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        annealing_coef = min(1.0, self.current_epoch / self.annealing_epochs)
        alp = E * (1 - label) + 1
        B = annealing_coef * self.compute_kl_loss(alp)
        
        return A + B
    
    def fuse_and_classify(self, evidences, labels):
        alpha_dict = {}
        for v in sorted(evidences.keys()):
            alpha_dict[v] = self.evidence_to_alpha(evidences[v])
        
        loss = torch.zeros(1, device=self.device)
        for v in sorted(alpha_dict.keys()):
            loss = loss + self.compute_ce_loss(alpha_dict[v], labels)
        
        alpha_list = [alpha_dict[v] for v in sorted(alpha_dict.keys())]
        if self.fusion_type == 'DS':
            alpha_fused = self.ds_combine_multi(alpha_list)
        elif self.fusion_type == 'Avg':
            alpha_fused = self.avg_combine(alpha_list)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        loss = loss + self.compute_ce_loss(alpha_fused, labels)
        loss = torch.mean(loss)
        loss.backward()
        
        gradients = {}
        for v, evidence in evidences.items():
            if evidence.grad is not None:
                gradients[v] = evidence.grad.clone()
            else:
                gradients[v] = torch.zeros_like(evidence)
        
        return gradients, loss.item()
    
    def predict(self, evidences):
        with torch.no_grad():
            alpha_list = []
            for v in sorted(evidences.keys()):
                alpha_v = self.evidence_to_alpha(evidences[v])
                alpha_list.append(alpha_v)
            
            if self.fusion_type == 'DS':
                alpha_fused = self.ds_combine_multi(alpha_list)
            else:
                alpha_fused = self.avg_combine(alpha_list)
            
            probs = self.alpha_to_prob(alpha_fused)
            uncertainty = self.alpha_to_uncertainty(alpha_fused)
        
        return probs, uncertainty


# ============================================================================
# FedTMC Training Coordinator
# ============================================================================

class FedTMCTrainer:
    """FedTMC Training Coordinator"""
    
    def __init__(self, dataset, fusion_type='DS', hidden_dim=256,
                 lr=0.001, batch_size=200, annealing_epochs=50, 
                 device='cpu', verbose_init=True):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.fusion_type = fusion_type
        
        train_idx = dataset.train_idx
        test_idx = dataset.test_idx
        
        self.train_data = [
            torch.FloatTensor(dataset.views_data[v][train_idx]).to(device)
            for v in range(dataset.num_views)
        ]
        self.train_labels = torch.LongTensor(dataset.labels[train_idx]).to(device)
        self.n_train = len(self.train_labels)
        
        self.test_data = [
            torch.FloatTensor(dataset.views_data[v][test_idx]).to(device)
            for v in range(dataset.num_views)
        ]
        self.test_labels = torch.LongTensor(dataset.labels[test_idx]).to(device)
        
        if verbose_init:
            print(f"[Data Preloaded] Train: {self.n_train} | Test: {len(self.test_labels)} | Device: {device}")
        
        self.clients = []
        for v in range(dataset.num_views):
            client = TMCClient(
                client_id=v,
                input_dim=dataset.dims[v],
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                lr=lr,
                device=device
            )
            self.clients.append(client)
        
        self.server = TMCServer(
            num_classes=dataset.num_classes,
            num_views=dataset.num_views,
            fusion_type=fusion_type,
            annealing_epochs=annealing_epochs,
            device=device
        )
        
        if verbose_init:
            print(f"FedTMC Clients: {dataset.num_views} | Fusion: {fusion_type} | Hidden: {hidden_dim} | LR: {lr}")
    
    def train_one_batch(self, batch_idx):
        batch_labels = self.train_labels[batch_idx]
        
        evidences = {}
        for v, client in enumerate(self.clients):
            X_batch = self.train_data[v][batch_idx]
            evidence = client.compute_evidence(X_batch)
            evidences[v] = evidence
        
        gradients, loss = self.server.fuse_and_classify(evidences, batch_labels)
        
        for v, client in enumerate(self.clients):
            client.receive_gradient_and_update(gradients[v])
        
        return loss
    
    def train_epoch(self, epoch):
        self.server.set_epoch(epoch)
        perm = torch.randperm(self.n_train, device=self.device)
        
        total_loss = 0
        n_batches = 0
        
        for start in range(0, self.n_train, self.batch_size):
            end = min(start + self.batch_size, self.n_train)
            batch_idx = perm[start:end]
            
            loss = self.train_one_batch(batch_idx)
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self):
        evidences = {}
        for v, client in enumerate(self.clients):
            evidences[v] = client.compute_evidence_eval(self.test_data[v])
        
        probs, uncertainty = self.server.predict(evidences)
        return self._compute_metrics(probs, uncertainty, self.test_labels)
    
    def _compute_metrics(self, probs, uncertainty, labels):
        probs = probs.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()
        labels = labels.cpu().numpy()
        
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        acc = (pred == labels).mean()
        
        precision = precision_score(labels, pred, average='macro', zero_division=0)
        recall = recall_score(labels, pred, average='macro', zero_division=0)
        
        bins = np.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                ece += mask.sum() * np.abs((pred[mask] == labels[mask]).mean() - conf[mask].mean())
        ece /= len(conf)
        
        is_wrong = (pred != labels).astype(int)
        auroc = roc_auc_score(is_wrong, uncertainty) if len(np.unique(is_wrong)) > 1 else 0.5
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'ece': ece,
            'auroc': auroc,
            'f1_macro': f1_score(labels, pred, average='macro'),
            'avg_conf': conf.mean(),
            'avg_uncertainty': uncertainty.mean(),
            'overconf': conf.mean() - acc,
        }
    
    def train(self, epochs, eval_freq=10, verbose=True):
        history = []
        
        if verbose:
            print(f"\n{'='*110}")
            print(f"Dataset: {self.dataset.name} | Method: FedTMC | Fusion: {self.fusion_type}")
            print(f"Clients: {self.dataset.num_views} | Classes: {self.dataset.num_classes}")
            print(f"{'='*110}")
            print(f"{'Epoch':<8}{'Loss':<10}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'ECE':<10}{'Uncert':<10}")
            print(f"{'-'*110}")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            
            if epoch % eval_freq == 0 or epoch == epochs:
                m = self.evaluate()
                m['epoch'], m['loss'] = epoch, loss
                history.append(m)
                
                if verbose:
                    print(f"{epoch:<8}{loss:<10.4f}{m['accuracy']:<10.4f}"
                          f"{m['precision']:<10.4f}{m['recall']:<10.4f}"
                          f"{m['ece']:<10.4f}{m['avg_uncertainty']:<10.4f}")
        
        if verbose:
            print(f"{'-'*110}")
            final = history[-1]
            print(f"Final: Acc={final['accuracy']:.4f}, Prec={final['precision']:.4f}, "
                  f"Rec={final['recall']:.4f}, ECE={final['ece']:.4f}")
        
        final_result = history[-1] if history else {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'ece': 1, 'auroc': 0
        }
        return history, final_result


# ============================================================================
# Dataset Loader (Supports all datasets)
# ============================================================================

class DatasetLoader:
    """Dataset Loader - Supports complex and classical multi-view datasets"""
    
    # ======================== General loading helper functions ========================
    
    @staticmethod
    def _extract_views_from_cell_array(X):
        views = []
        if X.ndim == 2:
            if X.shape[0] == 1:
                for v in range(X.shape[1]):
                    views.append(np.array(X[0, v]))
            elif X.shape[1] == 1:
                for v in range(X.shape[0]):
                    views.append(np.array(X[v, 0]))
            else:
                if X.shape[0] < X.shape[1]:
                    for v in range(X.shape[1]):
                        views.append(np.array(X[0, v]))
                else:
                    for v in range(X.shape[0]):
                        views.append(np.array(X[v, 0]))
        elif X.ndim == 1:
            for v in range(len(X)):
                views.append(np.array(X[v]))
        return views
    
    @staticmethod
    def _is_2d_feature_matrix(arr):
        if not isinstance(arr, np.ndarray):
            return False
        if arr.dtype == object:
            return False
        if arr.ndim != 2:
            return False
        return arr.shape[0] > 1 and arr.shape[1] > 1
    
    @staticmethod
    def _extract_views_and_labels(data, mat_path, verbose=True):
        available_keys = [k for k in data.keys() if not k.startswith('__')]
        
        if verbose:
            print(f"  [Debug] Available keys: {available_keys}")
        
        views = None
        labels = None
        single_view_candidate = None
        
        # Strategy 1: Check for X or fea fields
        for feature_key in ['X', 'fea', 'data', 'feature', 'features']:
            if feature_key in data:
                X = data[feature_key]
                if verbose:
                    print(f"  [Debug] Found '{feature_key}': shape={X.shape}, dtype={X.dtype}")
                
                if isinstance(X, np.ndarray) and X.dtype == object:
                    views = DatasetLoader._extract_views_from_cell_array(X)
                    if verbose:
                        print(f"  [Debug] Extracted {len(views)} views from cell array")
                    break
                elif isinstance(X, np.ndarray) and X.ndim == 3:
                    views = [X[:, :, v] for v in range(X.shape[2])]
                    if verbose:
                        print(f"  [Debug] Extracted {len(views)} views from 3D array")
                    break
                elif DatasetLoader._is_2d_feature_matrix(X):
                    if verbose:
                        print(f"  [Debug] '{feature_key}' is a 2D matrix, checking for multi-view fields...")
                    single_view_candidate = X
        
        # Strategy 2: Check for multi-field formats
        if views is None:
            view_patterns = [
                ('X', lambda k: k.startswith('X') and k[1:].isdigit()),
                ('view', lambda k: k.startswith('view') and k[4:].isdigit()),
                ('fea', lambda k: k.startswith('fea') and k[3:].isdigit()),
                ('V', lambda k: k.startswith('V') and k[1:].isdigit()),
                ('x', lambda k: k.startswith('x') and k[1:].isdigit()),
            ]
            
            for prefix, match_fn in view_patterns:
                matched_keys = sorted([k for k in available_keys if match_fn(k)],
                                      key=lambda x: int(''.join(filter(str.isdigit, x))))
                if len(matched_keys) >= 2:
                    views = [np.array(data[k]) for k in matched_keys]
                    if verbose:
                        print(f"  [Debug] Found {len(views)} views with pattern '{prefix}*': {matched_keys}")
                    break
        
        # Strategy 3: If a single 2D matrix was found earlier
        if views is None and single_view_candidate is not None:
            views = [single_view_candidate]
            if verbose:
                print(f"  [Warning] Only found single view (2D matrix)")
        
        # Extract labels
        label_keys = ['Y', 'y', 'gt', 'gnd', 'labels', 'label', 'target', 'targets', 'class', 'classes']
        for label_key in label_keys:
            if label_key in data:
                labels = np.array(data[label_key]).flatten()
                if verbose:
                    print(f"  [Debug] Found labels in '{label_key}': {len(labels)} samples, {len(np.unique(labels))} classes")
                break
        
        if views is None:
            raise KeyError(f"Cannot find feature data in {mat_path}. Available keys: {available_keys}")
        if labels is None:
            raise KeyError(f"Cannot find labels in {mat_path}. Available keys: {available_keys}")
        
        return views, labels
    
    @staticmethod
    def _extract_views_and_labels_h5(f, mat_path, verbose=True):
        available_keys = list(f.keys())
        
        if verbose:
            print(f"  [Debug] H5 keys: {available_keys}")
        
        views = None
        labels = None
        
        for feature_key in ['X', 'fea', 'data', 'feature', 'features']:
            if feature_key in f:
                X = f[feature_key]
                if verbose:
                    print(f"  [Debug] Found '{feature_key}': shape={X.shape}, dtype={X.dtype}")
                
                if X.dtype == h5py.ref_dtype or (hasattr(X.dtype, 'metadata') and X.dtype.metadata):
                    views = []
                    if X.ndim == 2:
                        if X.shape[0] == 1:
                            for i in range(X.shape[1]):
                                ref = X[0, i]
                                data_arr = f[ref][:].T
                                views.append(data_arr)
                        elif X.shape[1] == 1:
                            for i in range(X.shape[0]):
                                ref = X[i, 0]
                                data_arr = f[ref][:].T
                                views.append(data_arr)
                        else:
                            if X.shape[0] < X.shape[1]:
                                for i in range(X.shape[1]):
                                    ref = X[0, i]
                                    data_arr = f[ref][:].T
                                    views.append(data_arr)
                            else:
                                for i in range(X.shape[0]):
                                    ref = X[i, 0]
                                    data_arr = f[ref][:].T
                                    views.append(data_arr)
                    elif X.ndim == 1:
                        for i in range(len(X)):
                            ref = X[i]
                            data_arr = f[ref][:].T
                            views.append(data_arr)
                    
                    if verbose:
                        print(f"  [Debug] Extracted {len(views)} views from H5 references")
                    break
                else:
                    X_data = X[:]
                    if X_data.ndim == 3:
                        views = [X_data[:, :, v].T for v in range(X_data.shape[2])]
                    else:
                        views = [X_data.T]
                    
                    if verbose:
                        print(f"  [Debug] Extracted {len(views)} views from H5 dataset")
                    break
        
        if views is None:
            view_patterns = [
                ('X', lambda k: k.startswith('X') and k[1:].isdigit()),
                ('view', lambda k: k.startswith('view') and k[4:].isdigit()),
                ('fea', lambda k: k.startswith('fea') and k[3:].isdigit()),
                ('V', lambda k: k.startswith('V') and k[1:].isdigit()),
            ]
            
            for prefix, match_fn in view_patterns:
                matched_keys = sorted([k for k in available_keys if match_fn(k)],
                                      key=lambda x: int(''.join(filter(str.isdigit, x))))
                if len(matched_keys) >= 2:
                    views = [f[k][:].T for k in matched_keys]
                    if verbose:
                        print(f"  [Debug] Found {len(views)} views with pattern '{prefix}*': {matched_keys}")
                    break
        
        label_keys = ['Y', 'y', 'gt', 'gnd', 'labels', 'label', 'target', 'targets']
        for label_key in label_keys:
            if label_key in f:
                labels = f[label_key][:].flatten()
                if verbose:
                    print(f"  [Debug] Found labels in '{label_key}': {len(labels)} samples")
                break
        
        if views is None:
            raise KeyError(f"Cannot find feature data in {mat_path}. Available keys: {available_keys}")
        if labels is None:
            raise KeyError(f"Cannot find labels in {mat_path}. Available keys: {available_keys}")
        
        return views, labels
    
    @staticmethod
    def _load_generic_mat(mat_path, dataset_name, verbose=True, **kwargs):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"{os.path.basename(mat_path)} not found at {mat_path}")
        
        if verbose:
            print(f"  [Loading] {mat_path}")
        
        try:
            data = sio.loadmat(mat_path)
            views, labels = DatasetLoader._extract_views_and_labels(data, mat_path, verbose)
        except NotImplementedError:
            if verbose:
                print(f"  [Info] Using h5py for MATLAB v7.3 format")
            with h5py.File(mat_path, 'r') as f:
                views, labels = DatasetLoader._extract_views_and_labels_h5(f, mat_path, verbose)
        
        if verbose:
            print(f"  [Loaded] {len(views)} views, {len(labels)} samples")
            for i, v in enumerate(views):
                print(f"    View {i}: shape={v.shape}")
        
        return FedMVDataset(dataset_name, views, labels, **kwargs)
    
    # ======================== Original complex datasets ========================
    
    @staticmethod
    def load_caltech101(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'Caltech101-all.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Caltech101-all.mat not found in {data_path}")
        data = sio.loadmat(mat_path)
        views = [data['X'][0, v] for v in range(data['X'].shape[1])]
        labels = data['Y'].flatten() if 'Y' in data else data['gt'].flatten()
        return FedMVDataset('Caltech101', views, labels, **kwargs)
    
    @staticmethod
    def load_nus_wide(data_path, split=1, **kwargs):
        nus_path = os.path.join(data_path, 'nus_wide')
        if not os.path.exists(nus_path):
            raise FileNotFoundError(f"nus_wide not found in {data_path}")
        
        exclude = ['y.npy', '2019PR.py']
        files = sorted([f for f in os.listdir(nus_path) 
                       if f.endswith('.npy') 
                       and not f.startswith(('train_split', 'test_split'))
                       and f not in exclude])
        
        views = [np.load(os.path.join(nus_path, f)) for f in files]
        labels = np.load(os.path.join(nus_path, 'y.npy')).flatten()
        
        dataset = FedMVDataset('NUS-WIDE', views, labels, **kwargs)
        
        train_path = os.path.join(nus_path, f'train_split_{split}.npy')
        test_path = os.path.join(nus_path, f'test_split_{split}.npy')
        if os.path.exists(train_path):
            dataset.train_idx = np.load(train_path).flatten().astype(int)
            dataset.test_idx = np.load(test_path).flatten().astype(int)
        
        return dataset
    
    @staticmethod
    def load_youtube_face(data_path, split=1, **kwargs):
        yt_path = os.path.join(data_path, 'YoutubeFace')
        if not os.path.exists(yt_path):
            raise FileNotFoundError(f"YoutubeFace not found in {data_path}")
        
        view_files = sorted([f for f in os.listdir(yt_path) 
                            if f.startswith('v') and f.endswith('.npy') and f[1].isdigit()])
        
        views = [np.load(os.path.join(yt_path, f)) for f in view_files]
        labels = np.load(os.path.join(yt_path, 'y.npy')).flatten()
        
        dataset = FedMVDataset('YoutubeFace', views, labels, **kwargs)
        
        train_path = os.path.join(yt_path, f'train_split_{split}.npy')
        test_path = os.path.join(yt_path, f'test_split_{split}.npy')
        if os.path.exists(train_path):
            dataset.train_idx = np.load(train_path).flatten().astype(int)
            dataset.test_idx = np.load(test_path).flatten().astype(int)
        
        return dataset

    @staticmethod
    def load_vggface2(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'VGGFace2-50.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f'VGGFace2-50.mat not found in {data_path}')
        
        data = sio.loadmat(mat_path)
        views = [data['X'][v, 0] for v in range(data['X'].shape[0])]
        labels = data['Y'].flatten()
        
        return FedMVDataset('VGGFace2-50', views, labels, **kwargs)
    
    @staticmethod
    def load_awa2(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'AWA2.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f'AWA2.mat not found in {data_path}')
        
        with h5py.File(mat_path, 'r') as f:
            X = f['X']
            views = []
            for i in range(X.shape[1]):
                ref = X[0, i]
                data_arr = f[ref][:].T
                views.append(data_arr)
            
            labels = f['Y'][:].flatten()
        
        return FedMVDataset('AWA2', views, labels, **kwargs)

    @staticmethod
    def load_reuters(data_path, name='Reuters5noisy', split=1, **kwargs):
        reuters_path = os.path.join(data_path, name)
        if not os.path.exists(reuters_path):
            raise FileNotFoundError(f'{name} not found in {data_path}')
        
        view_names = ['EN.npy', 'FR.npy', 'GR.npy', 'IT.npy', 'SP.npy']
        views = [np.load(os.path.join(reuters_path, v)) for v in view_names]
        labels = np.load(os.path.join(reuters_path, 'y.npy')).flatten()
        
        dataset = FedMVDataset(name, views, labels, **kwargs)
        
        train_path = os.path.join(reuters_path, f'train_split_{split}.npy')
        test_path = os.path.join(reuters_path, f'test_split_{split}.npy')
        if os.path.exists(train_path):
            dataset.train_idx = np.load(train_path).flatten().astype(int)
            dataset.test_idx = np.load(test_path).flatten().astype(int)
        
        return dataset
    
    # ======================== Newly added classical multi-view datasets ========================
    
    @staticmethod
    def load_pie(data_path, split=1, **kwargs):
        """PIE face dataset"""
        mat_path = os.path.join(data_path, 'PIE_face_10.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'PIE', **kwargs)
    
    @staticmethod
    def load_scene15(data_path, split=1, **kwargs):
        """Scene15 scene classification dataset"""
        mat_path = os.path.join(data_path, 'scene15_mtv.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Scene15', **kwargs)
    
    @staticmethod
    def load_animal(data_path, split=1, **kwargs):
        """Animal dataset"""
        mat_path = os.path.join(data_path, 'Animal.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Animal', **kwargs)
    
    @staticmethod
    def load_caltech101_mv(data_path, split=1, **kwargs):
        """Caltech-101 multi-view version"""
        mat_path = os.path.join(data_path, 'Caltech101-all.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Caltech101-all.mat not found in {data_path}")
        
        data = sio.loadmat(mat_path)
        views = [data['X'][0, v] for v in range(data['X'].shape[1])]
        labels = data['Y'].flatten() if 'Y' in data else data['gt'].flatten()
        return FedMVDataset('Caltech-101', views, labels, **kwargs)
    
    @staticmethod
    def load_handwritten(data_path, split=1, **kwargs):
        """HandWritten digit dataset"""
        possible_names = ['handwritten.mat', 'handwritten0.mat', 'HandWritten.mat', 'mfeat.mat', 'MSRC-v1.mat']
        mat_path = None
        for name in possible_names:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                mat_path = path
                break
        
        if mat_path is None:
            raise FileNotFoundError(
                f"HandWritten dataset not found in {data_path}. Tried: {possible_names}."
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'HandWritten', **kwargs)
    
    @staticmethod
    def load_cub(data_path, split=1, **kwargs):
        """CUB bird dataset"""
        possible_names = ['CUB.mat', 'cub.mat', 'CUB-200.mat', 'CUB-200-2011.mat']
        mat_path = None
        for name in possible_names:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                mat_path = path
                break
        
        if mat_path is None:
            raise FileNotFoundError(
                f"CUB dataset not found in {data_path}. Tried: {possible_names}."
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'CUB', **kwargs)
    
    @staticmethod
    def load_aloi(data_path, split=1, **kwargs):
        """ALOI dataset"""
        possible_names = ['ALOI.mat', 'aloi.mat', 'ALOI-100.mat']
        mat_path = None
        for name in possible_names:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                mat_path = path
                break
        
        if mat_path is None:
            raise FileNotFoundError(
                f"ALOI dataset not found in {data_path}. Tried: {possible_names}."
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'ALOI', **kwargs)
    
    @staticmethod
    def load_aloi_1k(data_path, split=1, **kwargs):
        """ALOI_1K dataset"""
        possible_names = ['ALOI_1K.mat', 'ALOI-1K.mat', 'aloi_1k.mat', 'ALOI1K.mat']
        mat_path = None
        for name in possible_names:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                mat_path = path
                break
        
        if mat_path is None:
            raise FileNotFoundError(
                f"ALOI_1K dataset not found in {data_path}. Tried: {possible_names}."
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'ALOI_1K', **kwargs)
    
    # ======================== Unified loading interface ========================
    
    @classmethod
    def load(cls, name, data_path, **kwargs):
        """Unified dataset loading interface"""
        loaders = {
            # Original complex datasets
            'Caltech101': cls.load_caltech101,
            'NUS-WIDE': cls.load_nus_wide,
            'YoutubeFace': cls.load_youtube_face,
            'VGGFace2-50': cls.load_vggface2,
            'AWA2': cls.load_awa2,
            'Reuters5noisy': lambda dp, **kw: cls.load_reuters(dp, 'Reuters5noisy', **kw),
            'Reuters3noisy': lambda dp, **kw: cls.load_reuters(dp, 'Reuters3noisy', **kw),
            
            # Newly added classical multi-view datasets
            'PIE': cls.load_pie,
            'Scene15': cls.load_scene15,
            'Animal': cls.load_animal,
            'Caltech-101': cls.load_caltech101_mv,
            'HandWritten': cls.load_handwritten,
            'CUB': cls.load_cub,
            'ALOI': cls.load_aloi,
            'ALOI_1K': cls.load_aloi_1k,
        }
        
        if name not in loaders:
            available = list(loaders.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        return loaders[name](data_path, **kwargs)


# ============================================================================
# FedTMC Benchmark Runner
# ============================================================================

class FedTMCBenchmark:
    """FedTMC Benchmark Runner"""
    
    FUSIONS = ['DS', 'Avg']
    
    # Complex datasets
    DATASETS_COMPLEX = ['Caltech101', 'NUS-WIDE', 'YoutubeFace', 'VGGFace2-50', 
                        'AWA2', 'Reuters3noisy', 'Reuters5noisy']
    
    # Classical multi-view datasets (consistent with fed_mv_bench.py)
    DATASETS_CLASSICAL = ['PIE', 'HandWritten', 'Scene15', 'Caltech-101', 
                          'CUB', 'Animal', 'ALOI']
    
    def __init__(self, data_path, device='cpu', save_path='results'):
        self.data_path = data_path
        self.device = device
        self.save_path = save_path
        self.results = {}
        self.config = {}
        
        os.makedirs(save_path, exist_ok=True)
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_file = os.path.join(save_path, f'fedtmc_benchmark_{self.timestamp}.json')
        print(f"[Save] Results will be saved to: {self.save_file}")
    
    def _save_realtime(self):
        save_data = {
            'config': self.config,
            'results': {},
            'timestamp': self.timestamp,
            'last_update': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        for ds, methods in self.results.items():
            save_data['results'][ds] = {}
            for method, metrics in methods.items():
                save_data['results'][ds][method] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v 
                    for k, v in metrics.items()
                }
        
        with open(self.save_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def run(self, datasets=None, fusions=None, epochs=100, eval_freq=10, runs=10,
            lr=0.001, batch_size=200, hidden_dim=256, annealing_epochs=50,
            train_ratio=0.8, stratified=False):
        
        if datasets is None:
            datasets = self.DATASETS_CLASSICAL  # Use classical datasets by default
        if fusions is None:
            fusions = self.FUSIONS
        
        self.config = {
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'annealing_epochs': annealing_epochs,
            'runs': runs,
            'eval_freq': eval_freq,
            'fusions': fusions,
            'datasets': datasets,
            'device': str(self.device),
            'data_path': self.data_path,
            'train_ratio': train_ratio,
            'stratified': stratified,
        }
        
        print("\n" + "=" * 110)
        print("FedTMC Benchmark: Federated Trusted Multi-View Classification")
        print("=" * 110)
        print(f"Datasets: {datasets}")
        print(f"Fusion Types: {fusions}")
        print(f"Epochs: {epochs} | Runs: {runs} | LR: {lr} | Batch: {batch_size}")
        print(f"Hidden: {hidden_dim} | Annealing: {annealing_epochs}")
        print(f"Split: Train {train_ratio:.0%} / Test {1-train_ratio:.0%}")
        print(f"Split Method: {'Stratified' if stratified else 'Random'}")
        print(f"Save to: {self.save_file}")
        print("=" * 110)
        
        self._save_realtime()
        
        for ds_name in datasets:
            print(f"\n>>> Dataset: {ds_name}")
            try:
                dataset = DatasetLoader.load(
                    ds_name, self.data_path,
                    train_ratio=train_ratio,
                    stratified=stratified
                )
                print(f"    {dataset}")
            except Exception as e:
                print(f"    [SKIP] {e}")
                import traceback
                traceback.print_exc()
                continue
            
            self.results[ds_name] = {}
            
            for fusion in fusions:
                method_name = f"FedTMC-{fusion}"
                print(f"\n  --- {method_name} ---")
                
                all_acc, all_prec, all_rec, all_ece = [], [], [], []
                all_auroc, all_uncert, all_f1 = [], [], []
                run_details = []
                
                for run in range(runs):
                    torch.manual_seed(42 + run)
                    np.random.seed(42 + run)
                    
                    trainer = FedTMCTrainer(
                        dataset, 
                        fusion_type=fusion,
                        lr=lr,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        annealing_epochs=annealing_epochs,
                        device=self.device,
                        verbose_init=(run == 0)
                    )
                    
                    _, final = trainer.train(epochs, eval_freq, verbose=(run == 0))
                    
                    all_acc.append(final['accuracy'])
                    all_prec.append(final['precision'])
                    all_rec.append(final['recall'])
                    all_ece.append(final['ece'])
                    all_auroc.append(final['auroc'])
                    all_uncert.append(final['avg_uncertainty'])
                    all_f1.append(final['f1_macro'])
                    
                    run_details.append({
                        'run': run,
                        'accuracy': float(final['accuracy']),
                        'precision': float(final['precision']),
                        'recall': float(final['recall']),
                        'f1_macro': float(final['f1_macro']),
                        'ece': float(final['ece']),
                        'auroc': float(final['auroc']),
                        'avg_uncertainty': float(final['avg_uncertainty']),
                    })
                    
                    self.results[ds_name][method_name] = {
                        'acc_mean': float(np.mean(all_acc)), 'acc_std': float(np.std(all_acc)),
                        'prec_mean': float(np.mean(all_prec)), 'prec_std': float(np.std(all_prec)),
                        'rec_mean': float(np.mean(all_rec)), 'rec_std': float(np.std(all_rec)),
                        'f1_mean': float(np.mean(all_f1)), 'f1_std': float(np.std(all_f1)),
                        'ece_mean': float(np.mean(all_ece)), 'ece_std': float(np.std(all_ece)),
                        'auroc_mean': float(np.mean(all_auroc)), 'auroc_std': float(np.std(all_auroc)),
                        'uncert_mean': float(np.mean(all_uncert)), 'uncert_std': float(np.std(all_uncert)),
                        'completed_runs': run + 1,
                        'run_details': run_details.copy(),
                    }
                    self._save_realtime()
                    
                    print(f"    Run {run+1}/{runs}: Acc={final['accuracy']:.4f}, ECE={final['ece']:.4f} [Saved]")
                    
                    del trainer
                    torch.cuda.empty_cache()
                
                print(f"  [Result] Acc: {np.mean(all_acc):.4f}±{np.std(all_acc):.4f} | "
                      f"ECE: {np.mean(all_ece):.4f}±{np.std(all_ece):.4f} | "
                      f"AUROC: {np.mean(all_auroc):.4f}±{np.std(all_auroc):.4f}")
        
        print(f"\n[Done] Final results saved to: {self.save_file}")
        return self.results
    
    def summary(self):
        if not self.results:
            print("No results yet.")
            return
        
        print("\n" + "=" * 120)
        print("FedTMC BENCHMARK SUMMARY")
        print("=" * 120)
        
        all_methods = set()
        for ds_results in self.results.values():
            all_methods.update(ds_results.keys())
        methods = sorted(list(all_methods))
        
        for metric, symbol in [('acc', '↑'), ('ece', '↓'), ('auroc', '↑')]:
            print(f"\n{metric.upper()} ({symbol}):")
            print(f"{'Dataset':<15}", end="")
            for m in methods:
                print(f"{m:>18}", end="")
            print()
            
            for ds, res in self.results.items():
                print(f"  {ds:<13}", end="")
                for m in methods:
                    if m in res and f'{metric}_mean' in res[m]:
                        val = res[m][f'{metric}_mean']
                        std = res[m][f'{metric}_std']
                        print(f"{val:.3f}±{std:.2f}".rjust(18), end="")
                    else:
                        print("N/A".rjust(18), end="")
                print()
        
        print("=" * 120)
        print(f"Results saved to: {self.save_file}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FedTMC Benchmark (Supports classical multi-view datasets)')
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets')
    parser.add_argument('--fusions', type=str, nargs='+', default=None,
                        help='Fusion method: DS (Dempster-Shafer) or Avg (Simple Average)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--annealing-epochs', type=int, default=50)
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--benchmark-type', type=str, choices=['complex', 'classical', 'all'], 
                        default='classical', help='Which group of datasets to run')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    bench = FedTMCBenchmark(args.data_path, device, save_path=args.save_path)
    
    # Select datasets based on benchmark_type
    if args.datasets is not None:
        datasets = args.datasets
    elif args.benchmark_type == 'complex':
        datasets = FedTMCBenchmark.DATASETS_COMPLEX
    elif args.benchmark_type == 'classical':
        datasets = FedTMCBenchmark.DATASETS_CLASSICAL
    else:  # all
        datasets = FedTMCBenchmark.DATASETS_COMPLEX + FedTMCBenchmark.DATASETS_CLASSICAL
    
    bench.run(
        datasets, 
        args.fusions, 
        args.epochs, 
        args.eval_freq, 
        args.runs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        annealing_epochs=args.annealing_epochs,
        train_ratio=args.train_ratio,
        stratified=args.stratified
    )
    bench.summary()