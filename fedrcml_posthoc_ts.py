"""
================================================================================
FedRCML + Post-hoc Temperature Scaling (Post-hoc TS)

Core question:
  FedRCML version — Average fusion + DC Loss + Post-hoc TS

Experiment Design:
  1. Split 20% from the training set as a validation set (used for temperature scaling to avoid information leakage)
  2. Train standard FedRCML (Average fusion + DC Loss) using the remaining 80% training set
  3. After training, grid search for optimal temperature T* on the validation set
  4. Report Acc/ECE on the test set using T*
  5. Report the results of FedTMC (without TS) simultaneously as a control

Usage:
  python fedrcml_posthoc_ts.py \
    --data-path ./data \
    --datasets Scene15 --epochs 300 --runs 1 --gpu 0

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import h5py
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset — Supports train/val/test 3-way split
# ============================================================================

class FedMVDataset:
    def __init__(self, name, views_data, labels, train_ratio=0.8, val_ratio=0.2, seed=42):
        """
        val_ratio: Proportion of the validation set split from the training set (used for post-hoc TS tuning)
        Final split: train(64%) / val(16%) / test(20%)
        """
        self.name = name
        self.seed = seed
        self.views_data = []

        labels = labels.flatten().astype(np.int64)
        if labels.min() == 1:
            labels = labels - 1

        n_labels = len(labels)
        for i, v in enumerate(views_data):
            v = np.array(v)
            if v.shape[0] != n_labels:
                if v.shape[1] == n_labels:
                    v = v.T
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.views_data.append(scaler.fit_transform(v.astype(np.float64)))

        self.labels = labels
        self.num_views = len(self.views_data)
        self.num_samples = len(self.labels)
        self.num_classes = len(np.unique(self.labels))
        self.dims = [v.shape[1] for v in self.views_data]

        # 3-way split: train / val / test
        np.random.seed(seed)
        indices = np.random.permutation(self.num_samples)
        n_test = int(self.num_samples * (1 - train_ratio))
        n_trainval = self.num_samples - n_test
        n_val = int(n_trainval * val_ratio)
        n_train = n_trainval - n_val

        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]

        print(f"  {name}: {self.num_views} views, {self.num_classes} classes, {self.num_samples} samples")
        print(f"  Split: Train={len(self.train_idx)}, Val={len(self.val_idx)}, Test={len(self.test_idx)}")
        print(f"  Dims: {self.dims}")


# ============================================================================
# RCML Client
# ============================================================================

class RCMLClient:
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
# RCML Server
# ============================================================================

class RCMLServer:
    def __init__(self, num_classes, num_views, annealing_epochs=50, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
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
        return first_term + second_term

    def edl_digamma_loss(self, alpha, target):
        K = self.num_classes
        S = torch.sum(alpha, dim=1, keepdim=True)
        y = F.one_hot(target, num_classes=K).float()
        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
            torch.tensor(self.current_epoch / self.annealing_epochs, dtype=torch.float32, device=self.device),
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
        return torch.mean(dc_sum)

    def fuse_and_classify(self, evidences, labels):
        evidence_a = self.average_fusion(evidences)
        alpha_a = evidence_a + 1
        loss_acc = self.edl_digamma_loss(alpha_a, labels)
        for v in sorted(evidences.keys()):
            alpha = evidences[v] + 1
            loss_acc = loss_acc + self.edl_digamma_loss(alpha, labels)
        loss_acc = loss_acc / (len(evidences) + 1)
        dc_loss = self.get_dc_loss(evidences)
        loss = loss_acc + 1.0 * dc_loss
        loss.backward()
        gradients = {}
        for v, evidence in evidences.items():
            gradients[v] = evidence.grad.clone() if evidence.grad is not None else torch.zeros_like(evidence)
        return gradients, loss.item()

    def predict(self, evidences):
        with torch.no_grad():
            evidence_a = self.average_fusion(evidences)
            alpha_a = self.evidence_to_alpha(evidence_a)
            probs = self.alpha_to_prob(alpha_a)
        return probs, alpha_a


# ============================================================================
# Post-hoc Temperature Scaling
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


def compute_nll(probs_np, labels_np):
    """Negative log-likelihood — Used to find optimal temperature"""
    n = len(labels_np)
    nll = 0.0
    for i in range(n):
        p = max(probs_np[i, labels_np[i]], 1e-10)
        nll -= np.log(p)
    return nll / n


def find_optimal_temperature(alpha_fused, labels, device, method='nll'):
    """
    Search for the optimal temperature on the validation set.
    
    Apply temperature scaling to Dirichlet alpha: alpha_scaled = (alpha - 1) / T + 1
    This scales the evidence, preserving the Dirichlet structure.
    """
    labels_np = labels.cpu().numpy()
    alpha_np = alpha_fused.cpu().numpy()
    
    best_t = 1.0
    best_score = float('inf')
    
    # Grid search
    temperatures = np.concatenate([
        np.linspace(0.1, 1.0, 50),   # Fine search in low temp zone
        np.linspace(1.0, 5.0, 50),   # Search in high temp zone
        np.linspace(5.0, 20.0, 30),  # Search in super high temp zone
    ])
    
    for t in temperatures:
        # Scale evidence
        evidence = alpha_np - 1
        evidence_scaled = evidence / t
        alpha_scaled = evidence_scaled + 1
        
        # Calculate probability
        S = alpha_scaled.sum(axis=-1, keepdims=True)
        probs = alpha_scaled / S
        
        if method == 'nll':
            score = compute_nll(probs, labels_np)
        else:
            score = compute_ece(probs, labels_np)
        
        if score < best_score:
            best_score = score
            best_t = t
    
    return best_t


def apply_temperature(alpha_fused, temperature):
    """Apply temperature scaling to fused alpha"""
    evidence = alpha_fused - 1
    evidence_scaled = evidence / temperature
    alpha_scaled = evidence_scaled + 1
    S = alpha_scaled.sum(dim=-1, keepdim=True)
    probs = alpha_scaled / S
    return probs


def compute_all_metrics(probs_np, labels_np):
    conf = probs_np.max(axis=1)
    pred = probs_np.argmax(axis=1)
    acc = (pred == labels_np).mean()
    ece = compute_ece(probs_np, labels_np)
    overconf = conf.mean() - acc
    return {'accuracy': acc, 'ece': ece, 'avg_conf': conf.mean(), 'overconf': overconf}


# ============================================================================
# Dataset Loader — Supports multiple multi-view benchmark datasets
# ============================================================================

def load_generic_mat(data_path, dataset_name):
    """Generic .mat loader, supports multiple formats"""
    
    # Filename mapping
    file_candidates = {
        'Scene15': ['Scene15.mat', 'scene15_mtv.mat'],
        'HandWritten': ['handwritten.mat', 'handwritten0.mat', 'HandWritten.mat', 'mfeat.mat'],
        'Caltech-101': ['Caltech101-all.mat'],
        'CUB': ['CUB.mat', 'cub.mat', 'CUB-200.mat'],
        'Animal': ['Animal.mat', 'animal.mat'],
        'ALOI': ['ALOI.mat', 'aloi.mat', 'ALOI-100.mat'],
        # Complex datasets
        'VGGFace2-50': ['VGGFace2-50.mat'],
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
        raise FileNotFoundError(
            f"{dataset_name} not found in {data_path}. Tried: {candidates}")
    
    print(f"  Loading: {mat_path}")
    
    # Try scipy, if it fails, use h5py (MATLAB v7.3)
    try:
        data = sio.loadmat(mat_path)
        views, labels = _extract_scipy(data)
    except NotImplementedError:
        print(f"  Using h5py for MATLAB v7.3 format")
        with h5py.File(mat_path, 'r') as f:
            views, labels = _extract_h5py(f)
    
    # Fix dimensions (some datasets have transposed views)
    n_labels = len(labels)
    views_fixed = []
    for i, v in enumerate(views):
        v = np.array(v)
        if v.shape[0] != n_labels and v.shape[1] == n_labels:
            v = v.T
        views_fixed.append(v)
    
    print(f"  {dataset_name}: {len(views_fixed)} views, "
          f"shapes={[v.shape for v in views_fixed]}, "
          f"classes={len(np.unique(labels))}")
    
    return views_fixed, labels


def _extract_scipy(data):
    """Extract views and labels from scipy.io.loadmat results"""
    views = None
    labels = None
    
    # Extract views
    if 'X' in data:
        X = data['X']
        if isinstance(X, np.ndarray) and X.dtype == object:
            if X.shape[0] == 1:
                views = [np.array(X[0, v]) for v in range(X.shape[1])]
            elif X.shape[1] == 1:
                views = [np.array(X[v, 0]) for v in range(X.shape[0])]
            else:
                if X.shape[0] < X.shape[1]:
                    views = [np.array(X[0, v]) for v in range(X.shape[1])]
                else:
                    views = [np.array(X[v, 0]) for v in range(X.shape[0])]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            views = [X]
    
    # Try x1, x2, ... format
    if views is None:
        numbered_views = []
        for prefix in ['x', 'X', 'fea']:
            i = 1
            while f'{prefix}{i}' in data:
                numbered_views.append(np.array(data[f'{prefix}{i}']))
                i += 1
            if numbered_views:
                break
        if numbered_views:
            views = numbered_views
    
    # Extract labels
    for key in ['Y', 'y', 'gt', 'gnd', 'labels', 'label']:
        if key in data:
            labels = np.array(data[key]).flatten()
            break
    
    if views is None:
        keys = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(f"Cannot find views. Keys: {keys}")
    if labels is None:
        keys = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(f"Cannot find labels. Keys: {keys}")
    
    return views, labels


def _extract_h5py(f):
    """Extract views and labels from h5py File (MATLAB v7.3)"""
    views = None
    labels = None
    
    if 'X' in f:
        X = f['X']
        views = []
        if X.ndim == 2:
            dim0, dim1 = X.shape
            iterate_dim = dim1 if dim0 <= dim1 else dim0
            for i in range(iterate_dim):
                if dim0 <= dim1:
                    ref = X[0, i] if dim0 == 1 else X[i, 0]
                else:
                    ref = X[i, 0]
                data_arr = f[ref][:].T
                views.append(data_arr)
    
    for key in ['Y', 'y', 'gt', 'gnd', 'labels']:
        if key in f:
            labels = f[key][:].flatten()
            break
    
    if views is None or labels is None:
        raise KeyError(f"Cannot extract from h5py. Keys: {list(f.keys())}")
    
    return views, labels


# NUS-WIDE, YoutubeFace, Reuters and other npy format datasets
def load_npy_dataset(data_path, dataset_name):
    """Load npy format datasets"""
    
    if dataset_name.startswith('Reuters'):
        ds_path = os.path.join(data_path, dataset_name)
        view_names = ['EN.npy', 'FR.npy', 'GR.npy', 'IT.npy', 'SP.npy']
        if dataset_name == 'Reuters3noisy':
            view_names = ['EN.npy', 'FR.npy', 'GR.npy']
        views = [np.load(os.path.join(ds_path, v)) for v in view_names]
        labels = np.load(os.path.join(ds_path, 'y.npy')).flatten()
        return views, labels
    
    elif dataset_name == 'NUS-WIDE':
        ds_path = os.path.join(data_path, 'nus_wide')
        exclude = ['y.npy', '2019PR.py']
        files = sorted([f for f in os.listdir(ds_path)
                        if f.endswith('.npy')
                        and not f.startswith(('train_split', 'test_split'))
                        and f not in exclude])
        views = [np.load(os.path.join(ds_path, f)) for f in files]
        labels = np.load(os.path.join(ds_path, 'y.npy')).flatten()
        return views, labels
    
    elif dataset_name == 'YoutubeFace':
        ds_path = os.path.join(data_path, 'YoutubeFace')
        view_files = sorted([f for f in os.listdir(ds_path)
                             if f.startswith('v') and f.endswith('.npy') and f[1].isdigit()])
        views = [np.load(os.path.join(ds_path, f)) for f in view_files]
        labels = np.load(os.path.join(ds_path, 'y.npy')).flatten()
        return views, labels
    
    raise FileNotFoundError(f"Unknown npy dataset: {dataset_name}")


def load_dataset(data_path, dataset_name):
    """Unified dataset loading entry point"""
    npy_datasets = ['NUS-WIDE', 'YoutubeFace', 'Reuters3noisy', 'Reuters5noisy']
    
    if dataset_name in npy_datasets:
        return load_npy_dataset(data_path, dataset_name)
    else:
        return load_generic_mat(data_path, dataset_name)


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(data_path, dataset_name='Scene15', epochs=300, runs=1,
                   hidden_dim=256, lr=0.001, batch_size=200, gpu='0'):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name}...")
    views, labels = load_dataset(data_path, dataset_name)
    dataset = FedMVDataset(dataset_name, views, labels, train_ratio=0.8, val_ratio=0.2)

    all_results = []

    for run in range(runs):
        print(f"\n{'='*80}")
        print(f"Run {run+1}/{runs}")
        print(f"{'='*80}")

        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # ===== Prepare data =====
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
        print(f"  Train: {n_train}, Val: {len(val_labels)}, Test: {len(test_labels)}")

        # ===== Create FedTMC =====
        clients = []
        for v in range(dataset.num_views):
            client = RCMLClient(v, dataset.dims[v], hidden_dim, dataset.num_classes,
                               lr=lr, device=device)
            clients.append(client)

        server = RCMLServer(dataset.num_classes, dataset.num_views,
                           annealing_epochs=50, device=device)

        # ===== Train FedTMC =====
        print(f"\n--- Training FedRCML (Average fusion + DC loss, no calibration) ---")
        print(f"{'Epoch':<8}{'Loss':<12}{'Acc':<10}{'ECE':<10}{'OverConf':<12}")
        print(f"{'-'*60}")

        for epoch in range(1, epochs + 1):
            server.set_epoch(epoch)
            perm = torch.randperm(n_train, device=device)
            total_loss = 0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                if end - start < 2:
                    continue  # skip batch_size=1 (crashes BatchNorm)
                batch_idx = perm[start:end]
                batch_labels = train_labels[batch_idx]

                evidences = {}
                for v, client in enumerate(clients):
                    evidences[v] = client.compute_evidence(train_data[v][batch_idx])

                gradients, loss = server.fuse_and_classify(evidences, batch_labels)

                for v, client in enumerate(clients):
                    client.receive_gradient_and_update(gradients[v])

                total_loss += loss
                n_batches += 1

            # Evaluate every 50 epochs
            if epoch % 50 == 0 or epoch == epochs:
                evidences = {v: clients[v].compute_evidence_eval(test_data[v])
                             for v in range(dataset.num_views)}
                probs, alpha_fused = server.predict(evidences)
                m = compute_all_metrics(probs.cpu().numpy(), test_labels.cpu().numpy())
                print(f"{epoch:<8}{total_loss/n_batches:<12.4f}{m['accuracy']:<10.4f}"
                      f"{m['ece']:<10.4f}{m['overconf']:<+12.4f}")

        # ===== Evaluate FedTMC (No TS) =====
        print(f"\n--- Evaluating on Test Set ---")

        # Test set
        evidences_test = {v: clients[v].compute_evidence_eval(test_data[v])
                          for v in range(dataset.num_views)}
        probs_test, alpha_test = server.predict(evidences_test)
        metrics_no_ts = compute_all_metrics(probs_test.cpu().numpy(), test_labels.cpu().numpy())

        print(f"\n  FedRCML (no TS):  Acc={metrics_no_ts['accuracy']*100:.2f}%  "
              f"ECE={metrics_no_ts['ece']*100:.2f}%  "
              f"OverConf={metrics_no_ts['overconf']:+.4f}")

        # ===== Post-hoc Temperature Scaling =====
        print(f"\n--- Post-hoc Temperature Scaling ---")

        # Search optimal temperature on validation set
        evidences_val = {v: clients[v].compute_evidence_eval(val_data[v])
                         for v in range(dataset.num_views)}
        _, alpha_val = server.predict(evidences_val)

        # Method 1: Minimize NLL
        best_t_nll = find_optimal_temperature(alpha_val, val_labels, device, method='nll')
        # Method 2: Minimize ECE
        best_t_ece = find_optimal_temperature(alpha_val, val_labels, device, method='ece')

        print(f"  Optimal T (min NLL on val): {best_t_nll:.3f}")
        print(f"  Optimal T (min ECE on val): {best_t_ece:.3f}")

        # Apply optimal temperature on test set
        probs_ts_nll = apply_temperature(alpha_test, best_t_nll)
        metrics_ts_nll = compute_all_metrics(probs_ts_nll.cpu().numpy(), test_labels.cpu().numpy())

        probs_ts_ece = apply_temperature(alpha_test, best_t_ece)
        metrics_ts_ece = compute_all_metrics(probs_ts_ece.cpu().numpy(), test_labels.cpu().numpy())

        print(f"\n  FedRCML + TS (T={best_t_nll:.2f}, min-NLL): "
              f"Acc={metrics_ts_nll['accuracy']*100:.2f}%  "
              f"ECE={metrics_ts_nll['ece']*100:.2f}%  "
              f"OverConf={metrics_ts_nll['overconf']:+.4f}")

        print(f"  FedRCML + TS (T={best_t_ece:.2f}, min-ECE): "
              f"Acc={metrics_ts_ece['accuracy']*100:.2f}%  "
              f"ECE={metrics_ts_ece['ece']*100:.2f}%  "
              f"OverConf={metrics_ts_ece['overconf']:+.4f}")

        all_results.append({
            'no_ts': metrics_no_ts,
            'ts_nll': {**metrics_ts_nll, 'temperature': best_t_nll},
            'ts_ece': {**metrics_ts_ece, 'temperature': best_t_ece},
        })

        del clients, server
        torch.cuda.empty_cache()

    # ===== Summary =====
    print(f"\n{'='*80}")
    print(f"SUMMARY: FedRCML vs FedRCML+TS on {dataset_name}")
    print(f"{'='*80}")

    print(f"\n{'Method':<30}{'Acc (%)':<15}{'ECE (%)':<15}{'OverConf':<15}")
    print(f"{'-'*75}")

    # Average results
    avg_no_ts = {k: np.mean([r['no_ts'][k] for r in all_results]) for k in ['accuracy', 'ece', 'overconf']}
    avg_ts_nll = {k: np.mean([r['ts_nll'][k] for r in all_results]) for k in ['accuracy', 'ece', 'overconf']}
    avg_ts_ece = {k: np.mean([r['ts_ece'][k] for r in all_results]) for k in ['accuracy', 'ece', 'overconf']}
    avg_t_nll = np.mean([r['ts_nll']['temperature'] for r in all_results])
    avg_t_ece = np.mean([r['ts_ece']['temperature'] for r in all_results])

    print(f"{'FedRCML (no TS)':<30}{avg_no_ts['accuracy']*100:<15.2f}{avg_no_ts['ece']*100:<15.2f}{avg_no_ts['overconf']:<+15.4f}")
    print(f"{'FedRCML + TS (min-NLL)':<30}{avg_ts_nll['accuracy']*100:<15.2f}{avg_ts_nll['ece']*100:<15.2f}{avg_ts_nll['overconf']:<+15.4f}")
    print(f"{'FedRCML + TS (min-ECE)':<30}{avg_ts_ece['accuracy']*100:<15.2f}{avg_ts_ece['ece']*100:<15.2f}{avg_ts_ece['overconf']:<+15.4f}")
    print(f"\n  Avg optimal T (NLL): {avg_t_nll:.3f}")
    print(f"  Avg optimal T (ECE): {avg_t_ece:.3f}")

    print(f"\n  Note: TS does NOT change accuracy (same argmax).")
    print(f"  Note: Temperature was optimized on held-out validation set (20% of training data).")
    print(f"  Note: In VFL settings, constructing such validation set requires label sharing,")
    print(f"        which may not be feasible in practice.")

    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FedRCML + Post-hoc Temperature Scaling')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['Scene15'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    for ds in args.datasets:
        results = run_experiment(
            data_path=args.data_path,
            dataset_name=ds,
            epochs=args.epochs,
            runs=args.runs,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            batch_size=args.batch_size,
            gpu=args.gpu
        )