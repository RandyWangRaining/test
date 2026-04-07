"""
================================================================================
FedMVBench: Vertical Federated Multi-View Learning (Flexible Split Version)
================================================================================

Features:
1. Random split used by default (consistent with the original framework for comparability)
2. Optional stratified sampling (via the --stratified parameter)
3. Auto-detects and prints class distribution differences
4. Auto-transposes view data dimensions

Use cases:
- Default mode: Compare with other frameworks maintaining consistent random splitting
- Stratified mode: Enable for small datasets or when stable results are required

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
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
        """
        Args:
            name: Dataset name
            views_data: List of view data
            labels: Labels
            train_ratio: Training set ratio (default 0.8, remaining is test set)
            seed: Random seed
            stratified: Whether to use stratified sampling (default False, uses random split)
        """
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
            v = np.array(v)  # Ensure it's a numpy array
            if v.shape[0] != n_labels:
                if v.shape[1] == n_labels:
                    print(f"  [Auto-fix] View {i}: transposing {v.shape} -> {v.T.shape}")
                    v = v.T
                else:
                    raise ValueError(
                        f"View {i} shape {v.shape} incompatible with {n_labels} labels. "
                        f"Neither dimension matches the number of samples."
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
        
        # ===== Dataset splitting: Supports both random and stratified methods =====
        if stratified:
            print(f"  [Split Method] Stratified Sampling (Ensures class balance)")
            self._stratified_split(train_ratio, seed)
        else:
            print(f"  [Split Method] Random Shuffle (Consistent with original framework)")
            self._random_split(train_ratio, seed)
        
        # Print split information
        self._print_split_info()
    
    def _stratified_split(self, train_ratio, seed):
        """Stratified sampling split to ensure consistent class proportions"""
        indices = np.arange(self.num_samples)
        
        self.train_idx, self.test_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=self.labels,
            random_state=seed
        )
    
    def _random_split(self, train_ratio, seed):
        """Random split (consistent with the original framework)"""
        np.random.seed(seed)
        indices = np.random.permutation(self.num_samples)
        
        n_train = int(self.num_samples * train_ratio)
        self.train_idx = indices[:n_train]
        self.test_idx = indices[n_train:]
    
    def _print_split_info(self):
        """Print split information and class distribution"""
        print(f"  [Split] Train: {len(self.train_idx)}, Test: {len(self.test_idx)}")
        
        # Check class distribution
        train_dist = np.bincount(self.labels[self.train_idx], minlength=self.num_classes)
        test_dist = np.bincount(self.labels[self.test_idx], minlength=self.num_classes)
        
        # Check for missing classes
        missing_in_train = np.where(train_dist == 0)[0]
        missing_in_test = np.where(test_dist == 0)[0]
        
        if len(missing_in_train) > 0:
            print(f"  [Warning] {len(missing_in_train)} classes missing in train set: {missing_in_train[:5]}...")
        if len(missing_in_test) > 0:
            print(f"  [Warning] {len(missing_in_test)} classes missing in test set: {missing_in_test[:5]}...")
        
        # Calculate distribution difference
        train_ratio = train_dist / train_dist.sum()
        test_ratio = test_dist / test_dist.sum()
        dist_diff = np.abs(train_ratio - test_ratio).mean()
        
        if dist_diff > 0.05:
            print(f"  [Info] Class distribution difference: {dist_diff:.4f} (Large difference, consider using --stratified)")
        else:
            print(f"  [Info] Class distribution difference: {dist_diff:.4f} (Relatively balanced distribution)")
    
    def __repr__(self):
        return (f"FedMVDataset({self.name}): {self.num_views} views, "
                f"{self.num_classes} classes, {self.num_samples} samples, "
                f"dims={self.dims}")


# ============================================================================
# Client (Optimized Version)
# ============================================================================

class Client:
    """Federated Client (Optimized version)"""
    
    def __init__(self, client_id, input_dim, hidden_dim, output_dim,
                 lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        ).to(device)
        
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-5)
        self._feature = None
    
    def compute_feature(self, X):
        self.encoder.train()
        feature = self.encoder(X)
        feature_to_send = feature.detach().requires_grad_(True)
        self._feature = feature
        return feature_to_send
    
    def receive_gradient_and_update(self, grad):
        if self._feature is not None and grad is not None:
            self.optimizer.zero_grad()
            self._feature.backward(grad)
            self.optimizer.step()
        self._feature = None
    
    def compute_feature_eval(self, X):
        self.encoder.eval()
        with torch.no_grad():
            feature = self.encoder(X)
        return feature


# ============================================================================
# Server (7 Fusion Operators)
# ============================================================================

class Server:
    """Federated Server"""
    
    def __init__(self, num_classes, num_views, feature_dim, fusion_type='FedAvg', lr=0.001, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.device = device
        
        if fusion_type == 'FedCat':
            classifier_input = feature_dim * num_views
        else:
            classifier_input = feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, num_classes)
        ).to(device)
        
        if fusion_type == 'FedAttention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            ).to(device)
        
        if fusion_type == 'FedWeighted':
            self.weights = nn.Parameter(torch.ones(num_views, device=device) / num_views)
        
        params = list(self.classifier.parameters())
        if fusion_type == 'FedAttention':
            params += list(self.attention.parameters())
        if fusion_type == 'FedWeighted':
            params += [self.weights]
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
    
    def fuse_and_classify(self, features, labels):
        self.classifier.train()
        feat_list = [features[v] for v in sorted(features.keys())]
        
        if self.fusion_type == 'FedAdd':
            fused = sum(feat_list)
        elif self.fusion_type == 'FedAvg':
            fused = sum(feat_list) / len(feat_list)
        elif self.fusion_type == 'FedMul':
            fused = feat_list[0]
            for f in feat_list[1:]:
                fused = fused * f
        elif self.fusion_type == 'FedMax':
            stacked = torch.stack(feat_list, dim=0)
            fused = stacked.max(dim=0)[0]
        elif self.fusion_type == 'FedCat':
            fused = torch.cat(feat_list, dim=1)
        elif self.fusion_type == 'FedAttention':
            self.attention.train()
            stacked = torch.stack(feat_list, dim=1)
            attn_scores = self.attention(stacked).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)
            fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_type == 'FedWeighted':
            weights = F.softmax(self.weights, dim=0)
            stacked = torch.stack(feat_list, dim=1)
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        logits = self.classifier(fused)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        gradients = {}
        for v, feat in features.items():
            if feat.grad is not None:
                gradients[v] = feat.grad.clone()
            else:
                gradients[v] = torch.zeros_like(feat)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return gradients, loss.item()
    
    def predict(self, features):
        self.classifier.eval()
        if hasattr(self, 'attention'):
            self.attention.eval()
        
        with torch.no_grad():
            feat_list = [features[v] for v in sorted(features.keys())]
            
            if self.fusion_type == 'FedAdd':
                fused = sum(feat_list)
            elif self.fusion_type == 'FedAvg':
                fused = sum(feat_list) / len(feat_list)
            elif self.fusion_type == 'FedMul':
                fused = feat_list[0]
                for f in feat_list[1:]:
                    fused = fused * f
            elif self.fusion_type == 'FedMax':
                stacked = torch.stack(feat_list, dim=0)
                fused = stacked.max(dim=0)[0]
            elif self.fusion_type == 'FedCat':
                fused = torch.cat(feat_list, dim=1)
            elif self.fusion_type == 'FedAttention':
                stacked = torch.stack(feat_list, dim=1)
                attn_scores = self.attention(stacked).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1)
                fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
            elif self.fusion_type == 'FedWeighted':
                weights = F.softmax(self.weights, dim=0)
                stacked = torch.stack(feat_list, dim=1)
                fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
            
            logits = self.classifier(fused)
            probs = F.softmax(logits, dim=1)
            uncertainty = 1 - probs.max(dim=1)[0]
        
        return probs, uncertainty


# ============================================================================
# Federated Trainer (Optimized Version)
# ============================================================================

class FederatedTrainer:
    """Federated Training Coordinator (Optimized version)"""
    
    def __init__(self, dataset, fusion_type='FedAvg', hidden_dim=256, feature_dim=128,
                 lr=0.001, batch_size=512, device='cpu'):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.fusion_type = fusion_type
        self.lr = lr
        
        print(f"  [Optimization] Preloading data to {device}...")
        
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
        
        print(f"  [Optimization] Train samples: {self.n_train}, Test samples: {len(self.test_labels_gpu)}")
        
        self.clients = []
        for v in range(dataset.num_views):
            client = Client(
                client_id=v,
                input_dim=dataset.dims[v],
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                lr=lr,
                device=device
            )
            self.clients.append(client)
        
        self.server = Server(
            num_classes=dataset.num_classes,
            num_views=dataset.num_views,
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            lr=lr,
            device=device
        )
        
        print(f"  Clients: {dataset.num_views} | Server fusion: {fusion_type} | LR: {lr}")
    
    def train_one_batch(self, batch_data, batch_labels):
        features = {}
        for v, client in enumerate(self.clients):
            feat = client.compute_feature(batch_data[v])
            features[v] = feat
        
        gradients, loss = self.server.fuse_and_classify(features, batch_labels)
        
        for v, client in enumerate(self.clients):
            client.receive_gradient_and_update(gradients[v])
        
        return loss
    
    def train_epoch(self, epoch):
        perm = torch.randperm(self.n_train, device=self.device)
        
        total_loss = 0
        n_batches = 0
        
        for start in range(0, self.n_train, self.batch_size):
            end = min(start + self.batch_size, self.n_train)
            batch_idx = perm[start:end]
            
            batch_data = [self.train_data_gpu[v][batch_idx] for v in range(self.dataset.num_views)]
            batch_labels = self.train_labels_gpu[batch_idx]
            
            loss = self.train_one_batch(batch_data, batch_labels)
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self):
        features = {}
        for v, client in enumerate(self.clients):
            features[v] = client.compute_feature_eval(self.test_data_gpu[v])
        
        probs, uncertainty = self.server.predict(features)
        
        return self._compute_metrics(probs, uncertainty, self.test_labels_gpu)
    
    def _compute_metrics(self, probs, uncertainty, labels):
        probs = probs.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()
        labels = labels.cpu().numpy()
        
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        acc = (pred == labels).mean()
        
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
            'ece': ece,
            'auroc': auroc,
            'f1_macro': f1_score(labels, pred, average='macro'),
            'avg_conf': conf.mean(),
            'overconf': conf.mean() - acc,
        }
    
    def train(self, epochs, eval_freq=10, verbose=True):
        history = []
        best = {'accuracy': 0, 'ece': 1, 'auroc': 0}
        
        if verbose:
            print(f"\n{'='*90}")
            print(f"Dataset: {self.dataset.name} | Fusion: {self.fusion_type} | LR: {self.lr}")
            print(f"Clients: {self.dataset.num_views} | Classes: {self.dataset.num_classes}")
            print(f"{'='*90}")
            print(f"{'Epoch':<8}{'Loss':<12}{'Acc':<10}{'ECE':<10}{'AUROC':<10}{'OverConf':<12}")
            print(f"{'-'*90}")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            
            if epoch % eval_freq == 0 or epoch == epochs:
                m = self.evaluate()
                m['epoch'], m['loss'] = epoch, loss
                history.append(m)
                
                if m['accuracy'] > best['accuracy']: best['accuracy'] = m['accuracy']
                if m['ece'] < best['ece']: best['ece'] = m['ece']
                if m['auroc'] > best['auroc']: best['auroc'] = m['auroc']
                
                if verbose:
                    print(f"{epoch:<8}{loss:<12.4f}{m['accuracy']:<10.4f}"
                          f"{m['ece']:<10.4f}{m['auroc']:<10.4f}{m['overconf']:<+12.4f}")
        
        if verbose:
            print(f"{'-'*90}")
            final = history[-1]
            print(f"Final (Epoch {epochs}): Acc={final['accuracy']:.4f}, ECE={final['ece']:.4f}, AUROC={final['auroc']:.4f}")
        
        final_result = history[-1] if history else {'accuracy': 0, 'ece': 1, 'auroc': 0}
        return history, final_result


# ============================================================================
# Dataset Loader (Modify according to actual filenames)
# ============================================================================

class DatasetLoader:
    """Dataset Loader"""
    
    # ======================== General loading helper functions ========================
    
    @staticmethod
    def _extract_views_and_labels(data, mat_path):
        """General function to extract views and labels from mat data"""
        # Extract view data
        if 'X' in data:
            X = data['X']
            if isinstance(X, np.ndarray) and X.dtype == object:
                if X.ndim == 2:
                    views = [X[0, v] for v in range(X.shape[1])]
                else:
                    views = [X[v] for v in range(len(X))]
            else:
                views = [X]
        elif 'fea' in data:
            fea = data['fea']
            if isinstance(fea, np.ndarray) and fea.dtype == object:
                if fea.ndim == 2:
                    views = [fea[0, v] for v in range(fea.shape[1])]
                else:
                    views = [fea[v] for v in range(len(fea))]
            else:
                views = [fea]
        else:
            available_keys = [k for k in data.keys() if not k.startswith('__')]
            raise KeyError(f"Cannot find feature data in {mat_path}. Available keys: {available_keys}")
        
        # Extract labels
        for label_key in ['Y', 'gt', 'gnd', 'labels', 'label']:
            if label_key in data:
                labels = data[label_key].flatten()
                return views, labels
        
        available_keys = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(f"Cannot find labels in {mat_path}. Available keys: {available_keys}")
    
    @staticmethod
    def _extract_views_and_labels_h5(f, mat_path):
        """General function to extract views and labels from h5py file"""
        # Extract view data
        if 'X' in f:
            X = f['X']
            views = []
            for i in range(X.shape[1]):
                ref = X[0, i]
                data_arr = f[ref][:].T
                views.append(data_arr)
        elif 'fea' in f:
            fea = f['fea']
            views = []
            for i in range(fea.shape[1]):
                ref = fea[0, i]
                data_arr = f[ref][:].T
                views.append(data_arr)
        else:
            available_keys = list(f.keys())
            raise KeyError(f"Cannot find feature data in {mat_path}. Available keys: {available_keys}")
        
        # Extract labels
        for label_key in ['Y', 'gt', 'gnd', 'labels', 'label']:
            if label_key in f:
                labels = f[label_key][:].flatten()
                return views, labels
        
        available_keys = list(f.keys())
        raise KeyError(f"Cannot find labels in {mat_path}. Available keys: {available_keys}")
    
    @staticmethod
    def _load_generic_mat(mat_path, dataset_name, **kwargs):
        """General .mat file loading function"""
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"{os.path.basename(mat_path)} not found at {mat_path}")
        
        try:
            data = sio.loadmat(mat_path)
            views, labels = DatasetLoader._extract_views_and_labels(data, mat_path)
        except NotImplementedError:
            # MATLAB v7.3 format, use h5py
            with h5py.File(mat_path, 'r') as f:
                views, labels = DatasetLoader._extract_views_and_labels_h5(f, mat_path)
        
        return FedMVDataset(dataset_name, views, labels, **kwargs)
    
    # ======================== Original datasets ========================
    
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
    
    # ======================== Newly added datasets (using general loading function) ========================
    
    @staticmethod
    def load_pie(data_path, split=1, **kwargs):
        """PIE face dataset - uses PIE_face_10.mat"""
        mat_path = os.path.join(data_path, 'PIE_face_10.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'PIE', **kwargs)
    
    @staticmethod
    def load_scene15(data_path, split=1, **kwargs):
        """Scene15 scene classification dataset - uses scene15_mtv.mat"""
        mat_path = os.path.join(data_path, 'scene15_mtv.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Scene15', **kwargs)
    
    @staticmethod
    def load_animal(data_path, split=1, **kwargs):
        """Animal dataset - uses Animal.mat"""
        mat_path = os.path.join(data_path, 'Animal.mat')
        return DatasetLoader._load_generic_mat(mat_path, 'Animal', **kwargs)
    
    @staticmethod
    def load_caltech101_mv(data_path, split=1, **kwargs):
        """Caltech-101 multi-view version (reuses Caltech101-all.mat)"""
        mat_path = os.path.join(data_path, 'Caltech101-all.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Caltech101-all.mat not found in {data_path}")
        
        data = sio.loadmat(mat_path)
        views = [data['X'][0, v] for v in range(data['X'].shape[1])]
        labels = data['Y'].flatten() if 'Y' in data else data['gt'].flatten()
        return FedMVDataset('Caltech-101', views, labels, **kwargs)
    
    # ======================== Datasets that need to be downloaded ========================
    
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
                f"HandWritten dataset not found in {data_path}. "
                f"Tried: {possible_names}. "
                f"Please download from: https://archive.ics.uci.edu/ml/datasets/Multiple+Features"
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'HandWritten', **kwargs)
    
    @staticmethod
    def load_cub(data_path, split=1, **kwargs):
        """CUB bird dataset"""
        possible_names = ['CUB.mat', 'cub.mat', 'CUB-200.mat']
        mat_path = None
        for name in possible_names:
            path = os.path.join(data_path, name)
            if os.path.exists(path):
                mat_path = path
                break
        
        if mat_path is None:
            raise FileNotFoundError(
                f"CUB dataset not found in {data_path}. "
                f"Tried: {possible_names}. "
                f"Please download multi-view CUB dataset."
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
                f"ALOI dataset not found in {data_path}. "
                f"Tried: {possible_names}. "
                f"Please download multi-view ALOI dataset."
            )
        
        return DatasetLoader._load_generic_mat(mat_path, 'ALOI', **kwargs)
    
    # ======================== Unified loading interface ========================
    
    @classmethod
    def load(cls, name, data_path, **kwargs):
        """Unified dataset loading interface"""
        loaders = {
            # Original datasets
            'Caltech101': cls.load_caltech101,
            'NUS-WIDE': cls.load_nus_wide,
            'YoutubeFace': cls.load_youtube_face,
            'VGGFace2-50': cls.load_vggface2,
            'AWA2': cls.load_awa2,
            'Reuters5noisy': lambda dp, **kw: cls.load_reuters(dp, 'Reuters5noisy', **kw),
            'Reuters3noisy': lambda dp, **kw: cls.load_reuters(dp, 'Reuters3noisy', **kw),
            
            # Newly added datasets
            'PIE': cls.load_pie,
            'Scene15': cls.load_scene15,
            'Animal': cls.load_animal,
            'Caltech-101': cls.load_caltech101_mv,
            
            # Needs to be downloaded
            'HandWritten': cls.load_handwritten,
            'CUB': cls.load_cub,
            'ALOI': cls.load_aloi,
        }
        
        if name not in loaders:
            available = list(loaders.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        return loaders[name](data_path, **kwargs)


# ============================================================================
# Benchmark
# ============================================================================

class Benchmark:
    """Benchmark Runner"""
    
    FUSIONS = ['FedAdd', 'FedAvg', 'FedMul', 'FedMax', 'FedCat', 'FedAttention', 'FedWeighted']
    
    # Original complex datasets
    DATASETS_COMPLEX = ['Caltech101', 'NUS-WIDE', 'YoutubeFace', 'VGGFace2-50', 
                        'AWA2', 'Reuters3noisy', 'Reuters5noisy']
    
    # Newly added classical multi-view datasets
    DATASETS_CLASSICAL = ['PIE', 'HandWritten', 'Scene15', 'Caltech-101', 
                          'CUB', 'Animal', 'ALOI']
    
    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.device = device
        self.results = {}
    
    def run(self, datasets=None, fusions=None, epochs=100, eval_freq=10, runs=10, 
            batch_size=512, lr=0.001, train_ratio=0.8, stratified=False):
        """
        Run Benchmark
        
        Args:
            train_ratio: Train set ratio (default 0.8, rest is test set)
            stratified: Whether to use stratified sampling (default False, uses random split)
            lr: Learning rate (default 0.001)
        """
        if datasets is None:
            datasets = self.DATASETS_CLASSICAL  # Run newly added classical datasets by default
        if fusions is None:
            fusions = self.FUSIONS
        
        print("\n" + "=" * 100)
        print("FedMVBench: Vertical Federated Multi-View Learning (Flexible Split Version)")
        print("=" * 100)
        print(f"Datasets: {datasets}")
        print(f"Fusions: {fusions}")
        print(f"Epochs: {epochs} | Runs: {runs} | Batch Size: {batch_size} | LR: {lr}")
        print(f"Split: Train {train_ratio:.0%} / Test {1-train_ratio:.0%}")
        print(f"Split Method: {'Stratified Sampling' if stratified else 'Random Shuffle (Default)'}")
        print(f"Optimization: Preload data to GPU to reduce CPU-GPU transfers")
        print("=" * 100)
        
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
                continue
            
            self.results[ds_name] = {}
            
            for fusion in fusions:
                print(f"\n  --- {fusion} ---")
                
                all_acc, all_ece, all_auroc = [], [], []
                
                for run in range(runs):
                    torch.manual_seed(42 + run)
                    np.random.seed(42 + run)
                    
                    trainer = FederatedTrainer(
                        dataset, fusion, 
                        batch_size=batch_size,
                        lr=lr,
                        device=self.device
                    )
                    
                    _, final = trainer.train(epochs, eval_freq, verbose=(run == 0))
                    
                    all_acc.append(final['accuracy'])
                    all_ece.append(final['ece'])
                    all_auroc.append(final['auroc'])
                    
                    del trainer
                    torch.cuda.empty_cache()
                
                self.results[ds_name][fusion] = {
                    'acc_mean': np.mean(all_acc), 'acc_std': np.std(all_acc),
                    'ece_mean': np.mean(all_ece), 'ece_std': np.std(all_ece),
                    'auroc_mean': np.mean(all_auroc), 'auroc_std': np.std(all_auroc),
                }
                
                print(f"  [Result @ Epoch {epochs}] Acc: {np.mean(all_acc):.4f}±{np.std(all_acc):.4f} | "
                      f"ECE: {np.mean(all_ece):.4f}±{np.std(all_ece):.4f} | "
                      f"AUROC: {np.mean(all_auroc):.4f}±{np.std(all_auroc):.4f}")
        
        return self.results
    
    def summary(self):
        """Print summary"""
        if not self.results:
            print("No results yet.")
            return
        
        print("\n" + "=" * 120)
        print("BENCHMARK SUMMARY (Final Epoch Results)")
        print("=" * 120)
        
        fusions = list(list(self.results.values())[0].keys())
        
        for metric, symbol in [('acc', '↑'), ('ece', '↓'), ('auroc', '↑')]:
            print(f"\n{metric.upper()} ({symbol}):")
            print(f"{'Dataset':<15}", end="")
            for f in fusions:
                print(f"{f:>14}", end="")
            print()
            
            for ds, res in self.results.items():
                print(f"  {ds:<13}", end="")
                for f in fusions:
                    val = res[f][f'{metric}_mean']
                    std = res[f][f'{metric}_std']
                    marker = ""
                    if metric == 'ece':
                        marker = "✓" if val < 0.15 else "✗"
                    print(f"{val:.3f}±{std:.2f}{marker}".rjust(14), end="")
                print()
        
        print("=" * 120)
    
    def save(self, path='results'):
        """Save results"""
        os.makedirs(path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(path, f'benchmark_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
        return filepath


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FedMVBench (Flexible Split Version)')
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets, defaults to running classical multi-view datasets')
    parser.add_argument('--fusions', type=str, nargs='+', default=None,
                        help='List of fusion methods')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs, default 100')
    parser.add_argument('--runs', type=int, default=10, help='Number of repeated runs, default 10')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size, default 512')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate, default 0.001')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval-freq', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--benchmark-type', type=str, choices=['complex', 'classical', 'all'], 
                        default='classical', help='Which group of datasets to run: complex/classical/all')
    
    # Flexible split parameters
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                        help='Train set ratio, default 0.8, the rest is test set')
    parser.add_argument('--stratified', action='store_true',
                        help='Enable stratified sampling (random split used by default)')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    bench = Benchmark(args.data_path, device)
    
    # Select datasets based on benchmark_type
    if args.datasets is not None:
        datasets = args.datasets
    elif args.benchmark_type == 'complex':
        datasets = Benchmark.DATASETS_COMPLEX
    elif args.benchmark_type == 'classical':
        datasets = Benchmark.DATASETS_CLASSICAL
    else:  # all
        datasets = Benchmark.DATASETS_COMPLEX + Benchmark.DATASETS_CLASSICAL
    
    bench.run(
        datasets, args.fusions, args.epochs, args.eval_freq, args.runs, 
        args.batch_size,
        lr=args.lr,
        train_ratio=args.train_ratio,
        stratified=args.stratified
    )
    bench.summary()
    bench.save()