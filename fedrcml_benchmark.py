"""
================================================================================
FedRCML: Federated Reliable Conflictive Multi-view Learning
================================================================================

Federated version implementation based on RCML (Reliable Conflictive Multi-view Learning)

[Aligned with FedTMC architecture]

Supported datasets:
- Complex datasets: Caltech101, NUS-WIDE, YoutubeFace, VGGFace2-50, AWA2, Reuters5noisy, Reuters3noisy
- Classical datasets: PIE, HandWritten, Scene15, Caltech-101, CUB, Animal, ALOI, ALOI_1K
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
# Datasets (Identical to fed_mv_bench.py)
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
# EvidenceCollector (Aligned with FedTMC's TMCClient)
# ============================================================================

class EvidenceCollector(nn.Module):
    """
    EvidenceCollector - Aligned with the TMCClient network structure of FedTMC
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EvidenceCollector, self).__init__()
        
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
        )
    
    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# RCML Client
# ============================================================================

class RCMLClient:
    """RCML Client - Aligned with FedTMC's TMCClient"""
    
    def __init__(self, client_id, input_dim, hidden_dim, num_classes, lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        
        self.encoder = EvidenceCollector(input_dim, hidden_dim, num_classes).to(device)
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
# RCML Server
# ============================================================================

class RCMLServer:
    """RCML Server - Average Fusion + EDL Loss + DC Loss"""
    
    def __init__(self, num_classes, num_views, annealing_step=50, 
                 gamma=1.0, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
        self.annealing_step = annealing_step
        self.gamma = gamma
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
    
    def average_fusion(self, evidences):
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        evidence_a = evidence_list[0]
        for i in range(1, len(evidence_list)):
            evidence_a = (evidence_list[i] + evidence_a) / 2
        return evidence_a
    
    def kl_divergence(self, alpha):
        K = self.num_classes
        ones = torch.ones([1, K], dtype=torch.float32, device=self.device)
        
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        
        kl = first_term + second_term
        return kl
    
    def edl_digamma_loss(self, alpha, target):
        K = self.num_classes
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        y = F.one_hot(target, num_classes=K).float()
        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            torch.tensor(self.current_epoch / self.annealing_step, dtype=torch.float32, device=self.device),
        )
        
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha)
        
        return torch.mean(A + kl_div)
    
    def get_dc_loss(self, evidences):
        num_views = len(evidences)
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        
        batch_size = evidence_list[0].shape[0]
        num_classes = evidence_list[0].shape[1]
        
        p = torch.zeros((num_views, batch_size, num_classes), device=self.device)
        u = torch.zeros((num_views, batch_size), device=self.device)
        
        for v in range(num_views):
            alpha = evidence_list[v] + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            p[v] = alpha / S
            u[v] = torch.squeeze(num_classes / S)
        
        dc_sum = 0
        for i in range(num_views):
            pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)
            cc = (1 - u[i]) * (1 - u)
            dc = pd * cc
            dc_sum = dc_sum + torch.sum(dc, dim=0)
        
        dc_sum = torch.mean(dc_sum)
        return dc_sum
    
    def get_loss(self, evidences, evidence_a, labels):
        alpha_a = evidence_a + 1
        loss_acc = self.edl_digamma_loss(alpha_a, labels)
        
        for v in sorted(evidences.keys()):
            alpha = evidences[v] + 1
            loss_acc = loss_acc + self.edl_digamma_loss(alpha, labels)
        
        loss_acc = loss_acc / (len(evidences) + 1)
        dc_loss = self.get_dc_loss(evidences)
        loss = loss_acc + self.gamma * dc_loss
        
        return loss
    
    def fuse_and_classify(self, evidences, labels):
        evidence_a = self.average_fusion(evidences)
        loss = self.get_loss(evidences, evidence_a, labels)
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
            evidence_a = self.average_fusion(evidences)
            alpha_a = self.evidence_to_alpha(evidence_a)
            probs = self.alpha_to_prob(alpha_a)
            uncertainty = self.alpha_to_uncertainty(alpha_a)
        
        return probs, uncertainty


# ============================================================================
# FedRCML Training Coordinator
# ============================================================================

class FedRCMLTrainer:
    """FedRCML Training Coordinator"""
    
    def __init__(self, dataset, hidden_dim=256, lr=0.001, batch_size=512, 
                 annealing_step=50, gamma=1.0, device='cpu', verbose_init=True):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        self.train_data_gpu = [
            torch.FloatTensor(dataset.views_data[v][dataset.train_idx]).to(device)
            for v in range(dataset.num_views)
        ]
        self.train_labels_gpu = torch.LongTensor(dataset.labels[dataset.train_idx]).to(device)
        self.n_train = len(self.train_labels_gpu)
        
        self.test_data_gpu = [
            torch.FloatTensor(dataset.views_data[v][dataset.test_idx]).to(device)
            for v in range(dataset.num_views)
        ]
        self.test_labels_gpu = torch.LongTensor(dataset.labels[dataset.test_idx]).to(device)
        
        if verbose_init:
            print(f"[Data Preloaded] Train: {self.n_train} | Test: {len(self.test_labels_gpu)} | Device: {device}")
        
        self.clients = []
        for v in range(dataset.num_views):
            client = RCMLClient(
                client_id=v,
                input_dim=dataset.dims[v],
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                lr=lr,
                device=device
            )
            self.clients.append(client)
        
        self.server = RCMLServer(
            num_classes=dataset.num_classes,
            num_views=dataset.num_views,
            annealing_step=annealing_step,
            gamma=gamma,
            device=device
        )
        
        if verbose_init:
            print(f"FedRCML Clients: {dataset.num_views} | Hidden: {hidden_dim} | LR: {lr} | Gamma: {gamma}")
    
    def train_one_batch(self, batch_idx):
        batch_labels = self.train_labels_gpu[batch_idx]
        
        evidences = {}
        for v, client in enumerate(self.clients):
            X_batch = self.train_data_gpu[v][batch_idx]
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
            evidences[v] = client.compute_evidence_eval(self.test_data_gpu[v])
        
        probs, uncertainty = self.server.predict(evidences)
        return self._compute_metrics(probs, uncertainty, self.test_labels_gpu)
    
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
            print(f"Dataset: {self.dataset.name} | Method: FedRCML")
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
# Dataset Loader (Identical to fed_mv_bench.py)
# ============================================================================

class DatasetLoader:
    """Dataset Loader - Supports all datasets"""
    
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
        
        if views is None and single_view_candidate is not None:
            views = [single_view_candidate]
            if verbose:
                print(f"  [Warning] Only found single view (2D matrix)")
        
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
        mat_path = os.path.join(data_path, 'PIE_face_10.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'PIE', **kwargs)
    
    @staticmethod
    def load_scene15(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'scene15_mtv.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Scene15', **kwargs)
    
    @staticmethod
    def load_animal(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'Animal.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Animal', **kwargs)
    
    @staticmethod
    def load_caltech101_mv(data_path, split=1, **kwargs):
        mat_path = os.path.join(data_path, 'Caltech101-all.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Caltech101-all.mat not found in {data_path}")
        
        data = sio.loadmat(mat_path)
        views = [data['X'][0, v] for v in range(data['X'].shape[1])]
        labels = data['Y'].flatten() if 'Y' in data else data['gt'].flatten()
        return FedMVDataset('Caltech-101', views, labels, **kwargs)
    
    @staticmethod
    def load_handwritten(data_path, split=1, **kwargs):
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
# FedRCML Benchmark Runner
# ============================================================================

class FedRCMLBenchmark:
    """FedRCML Benchmark Runner"""
    
    # Complex datasets
    DATASETS_COMPLEX = ['Caltech101', 'NUS-WIDE', 'YoutubeFace', 'VGGFace2-50', 
                        'AWA2', 'Reuters3noisy', 'Reuters5noisy']
    
    # Classical multi-view datasets
    DATASETS_CLASSICAL = ['PIE', 'HandWritten', 'Scene15', 'Caltech-101', 
                          'CUB', 'Animal', 'ALOI', 'ALOI_1K']
    
    def __init__(self, data_path, device='cpu', save_path='results'):
        self.data_path = data_path
        self.device = device
        self.save_path = save_path
        self.results = {}
        self.config = {}
        
        os.makedirs(save_path, exist_ok=True)
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_file = os.path.join(save_path, f'fedrcml_benchmark_{self.timestamp}.json')
        print(f"[Save] Results will be saved to: {self.save_file}")
    
    def _save_realtime(self):
        save_data = {
            'config': self.config,
            'results': {},
            'timestamp': self.timestamp,
            'last_update': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        for ds, metrics in self.results.items():
            save_data['results'][ds] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v 
                for k, v in metrics.items()
            }
        
        with open(self.save_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def run(self, datasets=None, epochs=200, eval_freq=10, runs=10,
            hidden_dim=256, lr=0.001, batch_size=512, annealing_step=50, gamma=1.0,
            train_ratio=0.8, stratified=False):
        
        if datasets is None:
            datasets = self.DATASETS_CLASSICAL
        
        self.config = {
            'epochs': epochs,
            'runs': runs,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'batch_size': batch_size,
            'annealing_step': annealing_step,
            'gamma': gamma,
            'train_ratio': train_ratio,
            'stratified': stratified,
            'datasets': datasets,
            'device': str(self.device),
        }
        
        print("\n" + "=" * 110)
        print("FedRCML Benchmark: Federated Reliable Conflictive Multi-view Learning")
        print("=" * 110)
        print(f"Datasets: {datasets}")
        print(f"Epochs: {epochs} | Runs: {runs} | Hidden: {hidden_dim}")
        print(f"LR: {lr} | Batch: {batch_size} | Annealing: {annealing_step} | Gamma: {gamma}")
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
            
            print(f"\n  --- FedRCML ---")
            
            all_acc, all_prec, all_rec, all_ece = [], [], [], []
            all_auroc, all_uncert, all_f1 = [], [], []
            run_details = []
            
            for run in range(runs):
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                
                trainer = FedRCMLTrainer(
                    dataset,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    batch_size=batch_size,
                    annealing_step=annealing_step,
                    gamma=gamma,
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
                
                self.results[ds_name] = {
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
        
        print("\n" + "=" * 100)
        print("FedRCML BENCHMARK SUMMARY")
        print("=" * 100)
        
        print(f"\n{'Dataset':<15}{'Acc':<18}{'ECE':<18}{'AUROC':<18}")
        print("-" * 100)
        
        for ds, res in self.results.items():
            print(f"{ds:<15}"
                  f"{res['acc_mean']:.3f}±{res['acc_std']:.2f}".ljust(18) +
                  f"{res['ece_mean']:.3f}±{res['ece_std']:.2f}".ljust(18) +
                  f"{res['auroc_mean']:.3f}±{res['auroc_std']:.2f}".ljust(18))
        
        print("=" * 100)
        print(f"Results saved to: {self.save_file}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FedRCML Benchmark (Supports classical multi-view datasets)')
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--annealing-step', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--benchmark-type', type=str, choices=['complex', 'classical', 'all'], 
                        default='classical', help='Which group of datasets to run')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    bench = FedRCMLBenchmark(args.data_path, device, save_path=args.save_path)
    
    # Select datasets based on benchmark_type
    if args.datasets is not None:
        datasets = args.datasets
    elif args.benchmark_type == 'complex':
        datasets = FedRCMLBenchmark.DATASETS_COMPLEX
    elif args.benchmark_type == 'classical':
        datasets = FedRCMLBenchmark.DATASETS_CLASSICAL
    else:  # all
        datasets = FedRCMLBenchmark.DATASETS_COMPLEX + FedRCMLBenchmark.DATASETS_CLASSICAL
    
    bench.run(
        datasets,
        args.epochs,
        args.eval_freq,
        args.runs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        annealing_step=args.annealing_step,
        gamma=args.gamma,
        train_ratio=args.train_ratio,
        stratified=args.stratified
    )
    bench.summary()