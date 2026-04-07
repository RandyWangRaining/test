"""
================================================================================
FedTUNED: Federated Trusted Unified Feature-Neighborhood Dynamics
================================================================================

Based on TUNED (AAAI 2025): "Trusted Unified Feature-Neighborhood Dynamics
for Multi-View Classification" by Huang et al.

Adapted for Vertical Federated Learning (VFL) setting.

Key components from TUNED:
1. Local F-N structure: DNN features + GCN with CAN adaptive adjacency
2. F-N aggregation: Fuse feature and neighborhood structures → evidence
3. Global consensus evidence: Shared Dirichlet sampling + local conditioning
4. S-MRF fusion: Selective Markov Random Field (replaces DS combination rule)
5. Loss: EDL loss + consistency loss

VFL adaptation:
- Each client holds one view, builds local GCN adjacency, extracts evidence
- Server receives evidence from all clients, performs S-MRF fusion
- Raw features never leave clients

Compatible with FedRCML's DatasetLoader and FedMVDataset classes.
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
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import h5py
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Import shared components from FedRCML (DatasetLoader, FedMVDataset)
# If running standalone, these classes are duplicated below.
# ============================================================================

try:
    from fed_rcml import FedMVDataset, DatasetLoader
    print("[FedTUNED] Using shared DatasetLoader from fed_rcml.py")
except ImportError:
    # Fallback: define minimal versions here
    # (Copy FedMVDataset and DatasetLoader from fed_rcml.py if needed)
    print("[FedTUNED] fed_rcml.py not found, using built-in DatasetLoader")
    
    class FedMVDataset:
        """Federated Multi-View Dataset"""
        def __init__(self, name, views_data, labels, train_ratio=0.8, seed=42, stratified=False):
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
                    else:
                        raise ValueError(f"View {i} shape {v.shape} incompatible with {n_labels} labels.")
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
            
            if stratified:
                indices = np.arange(self.num_samples)
                self.train_idx, self.test_idx = train_test_split(
                    indices, train_size=train_ratio, stratify=self.labels, random_state=seed
                )
            else:
                np.random.seed(seed)
                indices = np.random.permutation(self.num_samples)
                n_train = int(self.num_samples * train_ratio)
                self.train_idx = indices[:n_train]
                self.test_idx = indices[n_train:]
        
        def __repr__(self):
            return (f"FedMVDataset({self.name}): {self.num_views} views, "
                    f"{self.num_classes} classes, {self.num_samples} samples, "
                    f"dims={self.dims}")
    
    from data_loader import load_dataset
    
    class DatasetLoader:
        @staticmethod
        def load(name, data_path, train_ratio=0.8, stratified=False, seed=42):
            views_data, labels = load_dataset(data_path, name)
            return FedMVDataset(name, views_data, labels, train_ratio=train_ratio,
                                seed=seed, stratified=stratified)


# ============================================================================
# CAN Adaptive Adjacency Matrix Construction
# ============================================================================

def build_can_adjacency(X, k=10):
    """
    Build adaptive adjacency matrix using CAN (Clustering with Adaptive Neighbors).
    Following TUNED Eq. 2-3.
    
    Args:
        X: Feature matrix (n_samples, n_features), numpy array
        k: Number of nearest neighbors
    Returns:
        A: Symmetric adjacency matrix (n_samples, n_samples), sparse torch tensor
    """
    n = X.shape[0]
    k = min(k, n - 1)
    
    # Find k nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)
    
    # Remove self-loop (first neighbor is self)
    distances = distances[:, 1:]  # (n, k)
    indices = indices[:, 1:]      # (n, k)
    
    # CAN adaptive weights: M_ij = max(0, (phi(x_i) - D_ij) / (k*phi(x_i) - sum(D_ij)))
    # phi(x_i) = distance to k-th neighbor
    phi = distances[:, -1]  # (n,)
    sum_d = distances.sum(axis=1)  # (n,)
    
    # Build sparse weight matrix
    rows, cols, vals = [], [], []
    for i in range(n):
        denom = k * phi[i] - sum_d[i]
        if abs(denom) < 1e-10:
            denom = 1e-10
        for j_idx in range(k):
            j = indices[i, j_idx]
            w = max(0.0, (phi[i] - distances[i, j_idx]) / denom)
            if w > 0:
                rows.append(i)
                cols.append(j)
                vals.append(w)
    
    # Symmetrize: A = (M + M^T) / 2
    rows_sym = rows + cols
    cols_sym = cols + rows
    vals_sym = vals + vals
    
    indices_t = torch.LongTensor([rows_sym, cols_sym])
    values_t = torch.FloatTensor(vals_sym) / 2.0
    A = torch.sparse_coo_tensor(indices_t, values_t, (n, n)).coalesce()
    
    return A


def normalize_adjacency(A, n):
    """
    Normalize adjacency matrix: D^{-1/2} (A + I) D^{-1/2}
    Input: sparse A, output: dense normalized matrix
    """
    A_dense = A.to_dense()
    A_hat = A_dense + torch.eye(n)
    D = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-10)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


# ============================================================================
# GCN Layer
# ============================================================================

class GCNLayer(nn.Module):
    """Single GCN layer: sigma(A_hat * X * W)"""
    
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, X, A_norm):
        """
        X: (n, in_features)
        A_norm: (n, n) normalized adjacency
        """
        H = self.linear(X)         # (n, out_features)
        H = A_norm @ H             # Graph convolution
        return F.relu(H)


# ============================================================================
# TUNED Client: Local F-N Structure Extraction + Evidence Generation
# ============================================================================

class TUNEDClient:
    """
    TUNED Client for VFL.
    
    Each client:
    1. Builds CAN adjacency matrix on local features (offline, once)
    2. Extracts feature structure via DNN (f^v)
    3. Extracts neighborhood structure via GCN (g^v)
    4. Fuses F-N structures to produce evidence via Psi(·)
    """
    
    def __init__(self, client_id, input_dim, hidden_dim, num_classes,
                 gcn_hidden=64, k_neighbors=10, lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        
        # Feature structure DNN: f^v(x; theta_v)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(device)
        
        # Neighborhood structure GCN: g^v(x; A_v)
        self.gcn1 = GCNLayer(input_dim, gcn_hidden).to(device)
        self.gcn2 = GCNLayer(gcn_hidden, hidden_dim).to(device)
        
        # F-N aggregation Psi(·): fuse feature + neighborhood → evidence
        self.fn_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),  # Ensure non-negative evidence
        ).to(device)
        
        # Collect all parameters
        params = (list(self.feature_net.parameters()) +
                  list(self.gcn1.parameters()) +
                  list(self.gcn2.parameters()) +
                  list(self.fn_aggregator.parameters()))
        
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
        
        # Adjacency matrix (built offline)
        self.A_norm_train = None
        self.A_norm_test = None
        
        self._evidence = None
    
    def build_adjacency(self, X_train_np, X_test_np):
        """Build CAN adjacency matrices offline (once)."""
        n_train = X_train_np.shape[0]
        n_test = X_test_np.shape[0]
        
        # Training adjacency
        A_train = build_can_adjacency(X_train_np, k=self.k_neighbors)
        self.A_norm_train = normalize_adjacency(A_train, n_train).to(self.device)
        
        # Test adjacency
        A_test = build_can_adjacency(X_test_np, k=self.k_neighbors)
        self.A_norm_test = normalize_adjacency(A_test, n_test).to(self.device)
    
    def _forward(self, X, A_norm):
        """Extract evidence from features + neighborhood structure."""
        # Feature structure: h^v = f^v(x)
        h_feat = self.feature_net(X)  # (n, hidden_dim)
        
        # Neighborhood structure: q^v = g^v(x, A)
        q1 = self.gcn1(X, A_norm)     # (n, gcn_hidden)
        q_neigh = self.gcn2(q1, A_norm)  # (n, hidden_dim)
        
        # F-N aggregation: e^v = Psi(h^v, q^v)
        fn_concat = torch.cat([h_feat, q_neigh], dim=1)  # (n, 2*hidden_dim)
        evidence = self.fn_aggregator(fn_concat)  # (n, num_classes)
        
        return evidence
    
    def compute_evidence(self, X, is_train=True):
        """Forward pass for training (with gradient tracking)."""
        self.feature_net.train()
        self.gcn1.train()
        self.gcn2.train()
        self.fn_aggregator.train()
        
        A_norm = self.A_norm_train if is_train else self.A_norm_test
        evidence = self._forward(X, A_norm)
        
        evidence_to_send = evidence.detach().requires_grad_(True)
        self._evidence = evidence
        return evidence_to_send
    
    def receive_gradient_and_update(self, grad):
        """Backprop server gradient through local network."""
        if self._evidence is not None and grad is not None:
            self.optimizer.zero_grad()
            self._evidence.backward(grad)
            self.optimizer.step()
        self._evidence = None
    
    def compute_evidence_eval(self, X):
        """Forward pass for evaluation (no gradient)."""
        self.feature_net.eval()
        self.gcn1.eval()
        self.gcn2.eval()
        self.fn_aggregator.eval()
        
        with torch.no_grad():
            evidence = self._forward(X, self.A_norm_test)
        return evidence


# ============================================================================
# TUNED Server: S-MRF Fusion + Global Consensus + Loss
# ============================================================================

class TUNEDServer:
    """
    TUNED Server for VFL.
    
    Performs:
    1. Global consensus evidence extraction (shared Dirichlet sampling)
    2. Selective Markov Random Field (S-MRF) fusion
    3. Loss computation (EDL + consistency)
    """
    
    def __init__(self, num_classes, num_views, annealing_step=50,
                 tau=0.5, lambda_con=0.1, device='cpu'):
        self.num_classes = num_classes
        self.num_views = num_views
        self.annealing_step = annealing_step
        self.tau = tau  # S-MRF edge threshold
        self.lambda_con = lambda_con  # Consistency loss weight
        self.device = device
        self.current_epoch = 1
        
        # Shared consensus evidence extractor (parameterized Dirichlet)
        # Learns global consensus conditioned on all views
        self.consensus_net = nn.Sequential(
            nn.Linear(num_classes * num_views, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softplus(),
        ).to(device)
        
        # Fusion function Phi(·): condition consensus on local evidence
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
        ).to(device)
        
        # S-MRF view weight network
        self.smrf_weight_net = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.consensus_net.parameters()) +
            list(self.fusion_mlp.parameters()) +
            list(self.smrf_weight_net.parameters()),
            lr=0.001, weight_decay=1e-5
        )
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def compute_smrf_weights(self, evidences):
        """
        Compute S-MRF edge weights based on pairwise cosine similarity.
        Following TUNED Eq. 10-12.
        """
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        V = len(evidence_list)
        batch_size = evidence_list[0].shape[0]
        
        # Compute pairwise cosine similarity between view evidences
        # Average over batch to get view-level similarity
        W = torch.zeros(V, V, device=self.device)
        for i in range(V):
            for j in range(i + 1, V):
                sim = F.cosine_similarity(evidence_list[i], evidence_list[j], dim=1)
                avg_sim = sim.mean()
                W[i, j] = avg_sim
                W[j, i] = avg_sim
        
        # Threshold: only keep edges >= tau * w_max (Eq. 11)
        w_max = W.max()
        if w_max > 0:
            mask = (W >= self.tau * w_max).float()
            W = W * mask
        
        # Normalize weights per view
        W_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-10)
        W_norm = W / W_sum
        
        return W_norm
    
    def smrf_fusion(self, evidences):
        """
        S-MRF selective fusion (Eq. 12).
        Aggregates evidence from selectively connected views.
        """
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        V = len(evidence_list)
        
        W = self.compute_smrf_weights(evidences)  # (V, V)
        
        # Stack evidences: (V, batch, K)
        E_stack = torch.stack(evidence_list, dim=0)
        
        # Weighted aggregation per view, then average
        # E_agg = sum_j w_{i,j} * E_j for each i, then average over i
        E_agg = torch.zeros_like(evidence_list[0])
        for i in range(V):
            for j in range(V):
                if i != j:
                    E_agg = E_agg + W[i, j] * evidence_list[j]
        
        E_agg = E_agg / max(V - 1, 1)
        
        # Also add direct average as residual
        E_mean = torch.stack(evidence_list, dim=0).mean(dim=0)
        E_fused = E_agg + E_mean
        
        return E_fused
    
    def global_consensus_fusion(self, evidences):
        """
        Global consensus evidence extraction (Eq. 9).
        Samples consensus from shared Dirichlet, conditions on local evidence.
        """
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        
        # Concatenate all view evidences as input to consensus net
        e_concat = torch.cat(evidence_list, dim=1)  # (batch, K*V)
        e_consensus = self.consensus_net(e_concat)   # (batch, K)
        
        # Condition consensus on each view: e_tilde_v = e_v + Phi(e_consensus, e_v)
        enhanced_evidences = {}
        for v in sorted(evidences.keys()):
            e_pair = torch.cat([e_consensus, evidences[v]], dim=1)  # (batch, 2K)
            delta = self.fusion_mlp(e_pair)  # (batch, K)
            enhanced_evidences[v] = F.softplus(evidences[v] + delta)  # Ensure non-negative
        
        return enhanced_evidences
    
    def kl_divergence(self, alpha):
        """KL divergence between Dirichlet(alpha) and uniform Dirichlet(1)."""
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
    
    def edl_loss(self, alpha, target):
        """EDL digamma loss with KL annealing (Eq. 14-16)."""
        K = self.num_classes
        S = torch.sum(alpha, dim=1, keepdim=True)
        y = F.one_hot(target, num_classes=K).float()
        
        # Cross-entropy term
        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # KL divergence with annealing
        annealing_coef = min(1.0, self.current_epoch / self.annealing_step)
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha)
        
        return torch.mean(A + kl_div)
    
    def consistency_loss(self, evidences):
        """
        Consistency loss (Eq. 17): encourage inter-view agreement.
        Combines cosine similarity + deviation from mean + regularization.
        """
        evidence_list = [evidences[v] for v in sorted(evidences.keys())]
        V = len(evidence_list)
        
        if V < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Pairwise cosine similarity (encourage agreement)
        cos_sim = 0.0
        count = 0
        for i in range(V):
            for j in range(i + 1, V):
                sim = F.cosine_similarity(evidence_list[i], evidence_list[j], dim=1).mean()
                cos_sim += sim
                count += 1
        cos_loss = -cos_sim / max(count, 1)
        
        # Deviation from mean
        e_stack = torch.stack(evidence_list, dim=0)  # (V, batch, K)
        e_mean = e_stack.mean(dim=0, keepdim=True)   # (1, batch, K)
        dev_loss = ((e_stack - e_mean) ** 2).mean()
        
        return cos_loss + 0.1 * dev_loss
    
    def get_loss(self, evidences, evidence_fused, labels):
        """Total loss: EDL (per-view + fused) + consistency."""
        # Fused evidence loss
        alpha_fused = evidence_fused + 1
        loss = self.edl_loss(alpha_fused, labels)
        
        # Per-view EDL loss
        for v in sorted(evidences.keys()):
            alpha_v = evidences[v] + 1
            loss = loss + self.edl_loss(alpha_v, labels)
        
        loss = loss / (len(evidences) + 1)
        
        # Consistency loss
        loss_con = self.consistency_loss(evidences)
        loss = loss + self.lambda_con * loss_con
        
        return loss
    
    def fuse_and_classify(self, evidences, labels):
        """Full forward: consensus → S-MRF fusion → loss → gradients."""
        self.optimizer.zero_grad()
        
        # Step 1: Global consensus enhancement
        enhanced_evidences = self.global_consensus_fusion(evidences)
        
        # Step 2: S-MRF selective fusion
        evidence_fused = self.smrf_fusion(enhanced_evidences)
        
        # Step 3: Loss
        loss = self.get_loss(enhanced_evidences, evidence_fused, labels)
        loss.backward()
        
        # Step 4: Update server parameters
        self.optimizer.step()
        
        # Step 5: Collect gradients for clients
        gradients = {}
        for v, evidence in evidences.items():
            if evidence.grad is not None:
                gradients[v] = evidence.grad.clone()
            else:
                gradients[v] = torch.zeros_like(evidence)
        
        return gradients, loss.item()
    
    def predict(self, evidences):
        """Inference: consensus → S-MRF fusion → predict."""
        self.consensus_net.eval()
        self.fusion_mlp.eval()
        self.smrf_weight_net.eval()
        
        with torch.no_grad():
            enhanced_evidences = self.global_consensus_fusion(evidences)
            evidence_fused = self.smrf_fusion(enhanced_evidences)
            
            alpha = evidence_fused + 1
            S = alpha.sum(dim=-1, keepdim=True)
            probs = alpha / S
            uncertainty = self.num_classes / S.squeeze(-1)
        
        return probs, uncertainty


# ============================================================================
# FedTUNED Trainer
# ============================================================================

class FedTUNEDTrainer:
    """FedTUNED Training Coordinator."""
    
    def __init__(self, dataset, hidden_dim=256, gcn_hidden=64, k_neighbors=10,
                 lr=0.001, batch_size=512, annealing_step=50,
                 tau=0.5, lambda_con=0.1, device='cpu', verbose_init=True):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        # Preload data to GPU
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
        
        # Create TUNED clients
        self.clients = []
        for v in range(dataset.num_views):
            client = TUNEDClient(
                client_id=v,
                input_dim=dataset.dims[v],
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                gcn_hidden=gcn_hidden,
                k_neighbors=k_neighbors,
                lr=lr,
                device=device
            )
            # Build adjacency matrices offline
            if verbose_init:
                print(f"  Building CAN adjacency for Client {v} (dim={dataset.dims[v]})...")
            client.build_adjacency(
                dataset.views_data[v][dataset.train_idx],
                dataset.views_data[v][dataset.test_idx]
            )
            self.clients.append(client)
        
        # Create TUNED server
        self.server = TUNEDServer(
            num_classes=dataset.num_classes,
            num_views=dataset.num_views,
            annealing_step=annealing_step,
            tau=tau,
            lambda_con=lambda_con,
            device=device
        )
        
        if verbose_init:
            print(f"FedTUNED Clients: {dataset.num_views} | Hidden: {hidden_dim} | "
                  f"GCN Hidden: {gcn_hidden} | k={k_neighbors} | tau={tau}")
    
    def train_epoch(self, epoch):
        """Train one epoch (full-batch for GCN compatibility)."""
        self.server.set_epoch(epoch)
        
        # GCN requires full adjacency, so we use full training set
        # (Standard practice for transductive GCN-based methods)
        evidences = {}
        for v, client in enumerate(self.clients):
            evidence = client.compute_evidence(self.train_data_gpu[v], is_train=True)
            evidences[v] = evidence
        
        gradients, loss = self.server.fuse_and_classify(evidences, self.train_labels_gpu)
        
        for v, client in enumerate(self.clients):
            client.receive_gradient_and_update(gradients[v])
        
        return loss
    
    def evaluate(self):
        """Evaluate on test set."""
        evidences = {}
        for v, client in enumerate(self.clients):
            evidences[v] = client.compute_evidence_eval(self.test_data_gpu[v])
        
        probs, uncertainty = self.server.predict(evidences)
        return self._compute_metrics(probs, uncertainty, self.test_labels_gpu)
    
    def _compute_metrics(self, probs, uncertainty, labels):
        """Compute accuracy, ECE, AUROC, etc."""
        probs = probs.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()
        labels = labels.cpu().numpy()
        
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        acc = (pred == labels).mean()
        
        precision = precision_score(labels, pred, average='macro', zero_division=0)
        recall = recall_score(labels, pred, average='macro', zero_division=0)
        
        # ECE with 15 equal-width bins
        bins = np.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if mask.sum() > 0:
                ece += mask.sum() * np.abs(
                    (pred[mask] == labels[mask]).mean() - conf[mask].mean()
                )
        ece /= len(conf)
        
        # AUROC for uncertainty
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
        """Full training loop."""
        history = []
        
        if verbose:
            print(f"\n{'=' * 110}")
            print(f"Dataset: {self.dataset.name} | Method: FedTUNED")
            print(f"Clients: {self.dataset.num_views} | Classes: {self.dataset.num_classes}")
            print(f"{'=' * 110}")
            print(f"{'Epoch':<8}{'Loss':<10}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'ECE':<10}{'Uncert':<10}")
            print(f"{'-' * 110}")
        
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
            print(f"{'-' * 110}")
            final = history[-1]
            print(f"Final: Acc={final['accuracy']:.4f}, Prec={final['precision']:.4f}, "
                  f"Rec={final['recall']:.4f}, ECE={final['ece']:.4f}")
        
        final_result = history[-1] if history else {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'ece': 1, 'auroc': 0
        }
        return history, final_result


# ============================================================================
# FedTUNED Benchmark Runner
# ============================================================================

class FedTUNEDBenchmark:
    """FedTUNED Benchmark Runner - compatible with FedRCML benchmark format."""
    
    DATASETS_CLASSICAL = ['HandWritten', 'Scene15', 'Caltech-101',
                          'CUB', 'Animal', 'ALOI']
    
    DATASETS_COMPLEX = ['NUS-WIDE', 'YoutubeFace', 'VGGFace2-50',
                        'AWA2', 'Reuters3noisy', 'Reuters5noisy']
    
    def __init__(self, data_path, device='cpu', save_path='results'):
        self.data_path = data_path
        self.device = device
        self.save_path = save_path
        self.results = {}
        self.config = {}
        
        os.makedirs(save_path, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_file = os.path.join(save_path, f'fedtuned_benchmark_{self.timestamp}.json')
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
            hidden_dim=256, gcn_hidden=64, k_neighbors=10,
            lr=0.001, batch_size=512, annealing_step=50,
            tau=0.5, lambda_con=0.1, train_ratio=0.8, stratified=False):
        
        if datasets is None:
            datasets = self.DATASETS_CLASSICAL
        
        self.config = {
            'method': 'FedTUNED',
            'epochs': epochs, 'runs': runs,
            'hidden_dim': hidden_dim, 'gcn_hidden': gcn_hidden,
            'k_neighbors': k_neighbors, 'lr': lr,
            'batch_size': batch_size, 'annealing_step': annealing_step,
            'tau': tau, 'lambda_con': lambda_con,
            'train_ratio': train_ratio, 'stratified': stratified,
            'datasets': datasets, 'device': str(self.device),
        }
        
        print("\n" + "=" * 110)
        print("FedTUNED Benchmark: Federated Trusted Unified Feature-Neighborhood Dynamics")
        print("=" * 110)
        print(f"Datasets: {datasets}")
        print(f"Epochs: {epochs} | Runs: {runs} | Hidden: {hidden_dim} | GCN: {gcn_hidden}")
        print(f"k={k_neighbors} | tau={tau} | lambda_con={lambda_con}")
        print("=" * 110)
        
        self._save_realtime()
        
        for ds_name in datasets:
            print(f"\n>>> Dataset: {ds_name}")
            try:
                dataset = DatasetLoader.load(
                    ds_name, self.data_path,
                    train_ratio=train_ratio, stratified=stratified
                )
                print(f"    {dataset}")
            except Exception as e:
                print(f"    [SKIP] {e}")
                import traceback
                traceback.print_exc()
                continue
            
            print(f"\n  --- FedTUNED ---")
            
            all_acc, all_prec, all_rec, all_ece = [], [], [], []
            all_auroc, all_uncert, all_f1 = [], [], []
            run_details = []
            
            for run in range(runs):
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                
                trainer = FedTUNEDTrainer(
                    dataset,
                    hidden_dim=hidden_dim,
                    gcn_hidden=gcn_hidden,
                    k_neighbors=k_neighbors,
                    lr=lr,
                    batch_size=batch_size,
                    annealing_step=annealing_step,
                    tau=tau,
                    lambda_con=lambda_con,
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
                    'ece': float(final['ece']),
                    'auroc': float(final['auroc']),
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
                
                print(f"    Run {run + 1}/{runs}: Acc={final['accuracy']:.4f}, ECE={final['ece']:.4f} [Saved]")
                
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
        print("FedTUNED BENCHMARK SUMMARY")
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
    
    parser = argparse.ArgumentParser(description='FedTUNED Benchmark')
    parser.add_argument('--data-path', type=str, required=True, help='Dataset path')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--gcn-hidden', type=int, default=64)
    parser.add_argument('--k-neighbors', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--annealing-step', type=int, default=50)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lambda-con', type=float, default=0.1)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--benchmark-type', type=str,
                        choices=['complex', 'classical', 'all'],
                        default='classical')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    bench = FedTUNEDBenchmark(args.data_path, device, save_path=args.save_path)
    
    if args.datasets is not None:
        datasets = args.datasets
    elif args.benchmark_type == 'complex':
        datasets = FedTUNEDBenchmark.DATASETS_COMPLEX
    elif args.benchmark_type == 'classical':
        datasets = FedTUNEDBenchmark.DATASETS_CLASSICAL
    else:
        datasets = FedTUNEDBenchmark.DATASETS_CLASSICAL + FedTUNEDBenchmark.DATASETS_COMPLEX
    
    bench.run(
        datasets, args.epochs, args.eval_freq, args.runs,
        hidden_dim=args.hidden_dim, gcn_hidden=args.gcn_hidden,
        k_neighbors=args.k_neighbors, lr=args.lr,
        batch_size=args.batch_size, annealing_step=args.annealing_step,
        tau=args.tau, lambda_con=args.lambda_con,
        train_ratio=args.train_ratio, stratified=args.stratified
    )
    bench.summary()