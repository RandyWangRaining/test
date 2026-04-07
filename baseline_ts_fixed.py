"""
================================================================================
Baseline Fusion Operators + Temperature Scaling (Fixed Version)
================================================================================

Fixes (Aligned with the FedRCML+TS protocol):
  1. Split out a held-out validation set before training (does not participate in training).
  2. Search for the optimal T using both NLL and ECE criteria simultaneously.
  3. Report the results of the min-ECE criterion.

Usage:
  python baseline_ts_fixed.py \
    --data-path ./data \
    --datasets Scene15\
    --fusions FedAvg FedAttention \
    --gpu 0 --runs 3
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import h5py
import argparse
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset — 3-way split: train / val / test
# ============================================================================

class FedMVDataset3Split:
    """Split out a held-out validation set before training, which does not participate in training"""
    
    def __init__(self, name, views_data, labels, train_ratio=0.8, val_ratio=0.2, seed=42):
        self.name = name
        self.seed = seed
        
        labels = labels.flatten().astype(np.int64)
        if labels.min() == 1:
            labels = labels - 1
        n_labels = len(labels)
        
        fixed_views = []
        for i, v in enumerate(views_data):
            v = np.array(v)
            if v.shape[0] != n_labels:
                if v.shape[1] == n_labels:
                    v = v.T
            fixed_views.append(v)
        
        self.views_data = []
        for v in fixed_views:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.views_data.append(scaler.fit_transform(v.astype(np.float64)))
        
        self.labels = labels
        self.num_views = len(self.views_data)
        self.num_samples = len(self.labels)
        self.num_classes = len(np.unique(self.labels))
        self.dims = [v.shape[1] for v in self.views_data]
        
        # 3-way split
        np.random.seed(seed)
        indices = np.random.permutation(self.num_samples)
        n_test = int(self.num_samples * (1 - train_ratio))
        n_trainval = self.num_samples - n_test
        n_val = int(n_trainval * val_ratio)
        n_train = n_trainval - n_val
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        print(f"  {name}: {self.num_views} views, {self.num_classes} classes")
        print(f"  Split: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")


# ============================================================================
# Client & Server (Consistent with the original fedmv_benchmark)
# ============================================================================

class Client:
    def __init__(self, client_id, input_dim, hidden_dim, output_dim, lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        
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
            return self.encoder(X)


class Server:
    def __init__(self, num_classes, num_views, feature_dim, fusion_type='FedAvg', lr=0.001, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.device = device
        
        classifier_input = feature_dim * num_views if fusion_type == 'FedCat' else feature_dim
        self.classifier = nn.Sequential(nn.Linear(classifier_input, num_classes)).to(device)
        
        if fusion_type == 'FedAttention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, 1)
            ).to(device)
        
        if fusion_type == 'FedWeighted':
            self.weights = nn.Parameter(torch.ones(num_views, device=device) / num_views)
        
        params = list(self.classifier.parameters())
        if fusion_type == 'FedAttention':
            params += list(self.attention.parameters())
        if fusion_type == 'FedWeighted':
            params += [self.weights]
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
    
    def fuse(self, feat_list):
        if self.fusion_type == 'FedAvg':
            return sum(feat_list) / len(feat_list)
        elif self.fusion_type == 'FedAdd':
            return sum(feat_list)
        elif self.fusion_type == 'FedMul':
            fused = feat_list[0]
            for f in feat_list[1:]:
                fused = fused * f
            return fused
        elif self.fusion_type == 'FedMax':
            return torch.stack(feat_list, dim=0).max(dim=0)[0]
        elif self.fusion_type == 'FedCat':
            return torch.cat(feat_list, dim=1)
        elif self.fusion_type == 'FedAttention':
            stacked = torch.stack(feat_list, dim=1)
            attn_scores = self.attention(stacked).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)
            return (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_type == 'FedWeighted':
            weights = F.softmax(self.weights, dim=0)
            stacked = torch.stack(feat_list, dim=1)
            return (stacked * weights.view(1, -1, 1)).sum(dim=1)
    
    def fuse_and_classify(self, features, labels):
        self.classifier.train()
        if hasattr(self, 'attention'):
            self.attention.train()
        feat_list = [features[v] for v in sorted(features.keys())]
        fused = self.fuse(feat_list)
        logits = self.classifier(fused)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        gradients = {}
        for v, feat in features.items():
            gradients[v] = feat.grad.clone() if feat.grad is not None else torch.zeros_like(feat)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return gradients, loss.item()


# ============================================================================
# Temperature Scaling
# ============================================================================

def compute_ece(probs_np, labels_np, n_bins=15):
    conf = probs_np.max(axis=1)
    pred = probs_np.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs((pred[mask] == labels_np[mask]).mean() - conf[mask].mean())
    return ece / len(conf)


def find_optimal_temperature(logits, labels, criterion='ece'):
    """Search for the optimal T on the held-out validation set"""
    best_T = 1.0
    best_metric = float('inf')
    
    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    for T in np.arange(0.1, 20.01, 0.1):
        scaled = logits_np / T
        # numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(shifted)
        probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        
        if criterion == 'ece':
            metric = compute_ece(probs, labels_np)
        else:  # nll
            nll = 0.0
            for i in range(len(labels_np)):
                p = max(probs[i, labels_np[i]], 1e-10)
                nll -= np.log(p)
            metric = nll / len(labels_np)
        
        if metric < best_metric:
            best_metric = metric
            best_T = T
    
    return best_T


# ============================================================================
# Data Loading
# ============================================================================

def load_generic_mat(data_path, dataset_name):
    file_candidates = {
        'Scene15': ['Scene15.mat', 'scene15_mtv.mat'],
        'HandWritten': ['handwritten.mat', 'handwritten0.mat', 'HandWritten.mat'],
        'Caltech-101': ['Caltech101-all.mat'],
        'CUB': ['CUB.mat', 'cub.mat'],
        'Animal': ['Animal.mat'],
        'ALOI': ['ALOI.mat', 'aloi.mat'],
        'AWA2': ['AWA2.mat'],
    }
    
    candidates = file_candidates.get(dataset_name, [f'{dataset_name}.mat'])
    mat_path = None
    for fname in candidates:
        path = os.path.join(data_path, fname)
        if os.path.exists(path):
            mat_path = path
            break
    if mat_path is None:
        raise FileNotFoundError(f"{dataset_name} not found. Tried: {candidates}")
    
    try:
        data = sio.loadmat(mat_path)
        X = data.get('X', data.get('fea'))
        if isinstance(X, np.ndarray) and X.dtype == object:
            if X.shape[0] == 1:
                views = [np.array(X[0, v]) for v in range(X.shape[1])]
            else:
                views = [np.array(X[v, 0]) for v in range(X.shape[0])]
        else:
            views = [X]
        for key in ['Y', 'y', 'gt', 'gnd', 'labels']:
            if key in data:
                labels = np.array(data[key]).flatten()
                break
    except NotImplementedError:
        with h5py.File(mat_path, 'r') as f:
            X = f['X']
            views = []
            for i in range(X.shape[1]):
                views.append(f[X[0, i]][:].T)
            for key in ['Y', 'y', 'gt']:
                if key in f:
                    labels = f[key][:].flatten()
                    break
    
    # fix dimensions
    n = len(labels)
    views = [v.T if v.shape[0] != n and v.shape[1] == n else v for v in views]
    
    return views, labels


# ============================================================================
# Main Experiment
# ============================================================================

def run_baseline_ts_fixed(data_path, dataset_name, fusion_types,
                          gpu='0', epochs=300, runs=3, batch_size=256):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    views, labels = load_generic_mat(data_path, dataset_name)
    
    print(f"\n{'='*90}")
    print(f" BASELINE FUSION + TS (FIXED PROTOCOL): {dataset_name}")
    print(f" Protocol: held-out val (before training), ECE + NLL criteria")
    print(f"{'='*90}")
    
    all_results = {}
    
    for fusion in fusion_types:
        acc_list, ece_list = [], []
        ts_ece_nll_list, ts_ece_ece_list = [], []
        
        for run in range(runs):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            # 3-way split: train/val/test
            dataset = FedMVDataset3Split(dataset_name, views, labels,
                                         train_ratio=0.8, val_ratio=0.2, seed=42+run)
            
            # Prepare data
            train_data = [torch.FloatTensor(dataset.views_data[v][dataset.train_idx]).to(device)
                          for v in range(dataset.num_views)]
            train_labels = torch.LongTensor(dataset.labels[dataset.train_idx]).to(device)
            
            val_data = [torch.FloatTensor(dataset.views_data[v][dataset.val_idx]).to(device)
                        for v in range(dataset.num_views)]
            val_labels = torch.LongTensor(dataset.labels[dataset.val_idx]).to(device)
            
            test_data = [torch.FloatTensor(dataset.views_data[v][dataset.test_idx]).to(device)
                         for v in range(dataset.num_views)]
            test_labels = torch.LongTensor(dataset.labels[dataset.test_idx]).to(device)
            
            n_train = len(train_labels)
            
            # Create clients & server
            clients = [Client(v, dataset.dims[v], 256, 128, lr=0.001, device=device)
                       for v in range(dataset.num_views)]
            server = Server(dataset.num_classes, dataset.num_views, 128,
                           fusion_type=fusion, lr=0.001, device=device)
            
            # Train (only use train set, val is excluded)
            for epoch in range(1, epochs + 1):
                perm = torch.randperm(n_train, device=device)
                for start in range(0, n_train, batch_size):
                    end = min(start + batch_size, n_train)
                    if end - start < 2: continue
                    idx = perm[start:end]
                    
                    features = {}
                    for v, client in enumerate(clients):
                        features[v] = client.compute_feature(train_data[v][idx])
                    gradients, _ = server.fuse_and_classify(features, train_labels[idx])
                    for v, client in enumerate(clients):
                        client.receive_gradient_and_update(gradients[v])
            
            # Evaluate
            server.classifier.eval()
            if hasattr(server, 'attention'):
                server.attention.eval()
            
            with torch.no_grad():
                # Test logits
                test_feats = [clients[v].compute_feature_eval(test_data[v])
                              for v in range(dataset.num_views)]
                test_fused = server.fuse(test_feats)
                test_logits = server.classifier(test_fused)
                
                # Val logits
                val_feats = [clients[v].compute_feature_eval(val_data[v])
                             for v in range(dataset.num_views)]
                val_fused = server.fuse(val_feats)
                val_logits = server.classifier(val_fused)
            
            # Original performance
            test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
            
            conf = test_probs.max(axis=1)
            pred = test_probs.argmax(axis=1)
            acc = (pred == test_labels_np).mean()
            ece = compute_ece(test_probs, test_labels_np)
            
            # TS — NLL criterion
            T_nll = find_optimal_temperature(val_logits, val_labels, criterion='nll')
            ts_probs_nll = F.softmax(test_logits / T_nll, dim=1).cpu().numpy()
            ts_ece_nll = compute_ece(ts_probs_nll, test_labels_np)
            
            # TS — ECE criterion
            T_ece = find_optimal_temperature(val_logits, val_labels, criterion='ece')
            ts_probs_ece = F.softmax(test_logits / T_ece, dim=1).cpu().numpy()
            ts_ece_ece = compute_ece(ts_probs_ece, test_labels_np)
            
            acc_list.append(acc * 100)
            ece_list.append(ece * 100)
            ts_ece_nll_list.append(ts_ece_nll * 100)
            ts_ece_ece_list.append(ts_ece_ece * 100)
            
            print(f"  {fusion} Run {run+1}: Acc={acc*100:.2f}% ECE={ece*100:.2f}% "
                  f"+TS(NLL)={ts_ece_nll*100:.2f}%(T={T_nll:.1f}) "
                  f"+TS(ECE)={ts_ece_ece*100:.2f}%(T={T_ece:.1f})")
            
            del clients, server
            torch.cuda.empty_cache()
        
        all_results[fusion] = {
            'acc': np.mean(acc_list), 'acc_std': np.std(acc_list),
            'ece': np.mean(ece_list), 'ece_std': np.std(ece_list),
            'ts_ece_nll': np.mean(ts_ece_nll_list), 'ts_ece_nll_std': np.std(ts_ece_nll_list),
            'ts_ece_ece': np.mean(ts_ece_ece_list), 'ts_ece_ece_std': np.std(ts_ece_ece_list),
        }
        
        r = all_results[fusion]
        print(f"  {fusion} MEAN: Acc={r['acc']:.2f}% ECE={r['ece']:.2f}% "
              f"+TS(NLL)={r['ts_ece_nll']:.2f}% +TS(ECE)={r['ts_ece_ece']:.2f}%\n")
    
    # Summary
    print(f"\n{'='*100}")
    print(f" SUMMARY: Baseline Fusion + TS on {dataset_name} (FIXED PROTOCOL)")
    print(f"{'='*100}")
    print(f"{'Method':<18}{'Acc(%)':<14}{'ECE(%)':<14}{'+TS-NLL(%)':<14}{'+TS-ECE(%)':<14}{'ΔECE(ECE)':<12}")
    print(f"{'-'*80}")
    for fusion in fusion_types:
        r = all_results[fusion]
        delta = r['ece'] - r['ts_ece_ece']
        print(f"{fusion:<18}{r['acc']:.2f}±{r['acc_std']:.1f}  "
              f"{r['ece']:.2f}±{r['ece_std']:.1f}  "
              f"{r['ts_ece_nll']:.2f}±{r['ts_ece_nll_std']:.1f}  "
              f"{r['ts_ece_ece']:.2f}±{r['ts_ece_ece_std']:.1f}  "
              f"{delta:+.2f}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        default='./data')
    parser.add_argument('--datasets', nargs='+', default=['Scene15', 'Caltech-101', 'Animal'])
    parser.add_argument('--fusions', nargs='+', default=['FedAvg', 'FedAttention'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    for ds in args.datasets:
        try:
            run_baseline_ts_fixed(args.data_path, ds, args.fusions,
                                   gpu=args.gpu, epochs=args.epochs, runs=args.runs)
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}")
            traceback.print_exc()