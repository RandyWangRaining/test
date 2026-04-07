#!/usr/bin/env python3
"""
Experiment: Baseline Fusion Operators + Temperature Scaling Comparison
Validate Key Question: "Is baseline fusion + TS sufficient?"
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from fedmv_benchmark import FederatedTrainer, DatasetLoader, FedMVDataset
from data_loader import load_dataset
import argparse


def temperature_scale(logits, T):
    return logits / T


def find_optimal_temperature(logits_val, labels_val, criterion='ece'):
    """Search for the optimal temperature on the validation set"""
    best_T = 1.0
    best_metric = float('inf')
    
    for T in np.arange(0.1, 20.01, 0.01):
        scaled = temperature_scale(logits_val, T)
        probs = F.softmax(scaled, dim=1)
        probs_np = probs.cpu().numpy()
        labels_np = labels_val.cpu().numpy()
        
        if criterion == 'nll':
            metric = F.cross_entropy(scaled, labels_val).item()
        else:  # ece
            conf = probs_np.max(axis=1)
            pred = probs_np.argmax(axis=1)
            bins = np.linspace(0, 1, 16)
            ece = 0.0
            for i in range(15):
                mask = (conf > bins[i]) & (conf <= bins[i+1])
                if mask.sum() > 0:
                    ece += mask.sum() * np.abs((pred[mask] == labels_np[mask]).mean() - conf[mask].mean())
            metric = ece / len(conf)
        
        if metric < best_metric:
            best_metric = metric
            best_T = T
    
    return best_T


def compute_metrics(probs_np, labels_np):
    conf = probs_np.max(axis=1)
    pred = probs_np.argmax(axis=1)
    acc = (pred == labels_np).mean()
    
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs((pred[mask] == labels_np[mask]).mean() - conf[mask].mean())
    ece /= len(conf)
    
    overconf = conf.mean() - acc
    return acc, ece, overconf


def run_baseline_ts(data_path, dataset_name, fusion_types, gpu='0', epochs=300, runs=5):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    views, labels = load_dataset(data_path, dataset_name)
    
    print(f"\n{'='*100}")
    print(f" BASELINE FUSION + TS: {dataset_name}")
    print(f"{'='*100}")
    
    all_results = {}
    
    for fusion in fusion_types:
        acc_list, ece_list, ts_ece_list = [], [], []
        
        for run in range(runs):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            dataset = FedMVDataset(dataset_name, views, labels, seed=42+run)
            
            trainer = FederatedTrainer(
                dataset, fusion_type=fusion,
                hidden_dim=256, feature_dim=128,
                lr=0.001, batch_size=256, device=device
            )
            
            history, final = trainer.train(epochs, eval_freq=10, verbose=False)
            
            # Get logits (without applying softmax)
            with torch.no_grad():
                features = {}
                for v, client in enumerate(trainer.clients):
                    features[v] = client.compute_feature_eval(trainer.test_data_gpu[v])
                
                feat_list = [features[v] for v in sorted(features.keys())]
                if fusion == 'FedAvg':
                    fused = sum(feat_list) / len(feat_list)
                elif fusion == 'FedAdd':
                    fused = sum(feat_list)
                elif fusion == 'FedMul':
                    fused = feat_list[0]
                    for f in feat_list[1:]:
                        fused = fused * f
                elif fusion == 'FedMax':
                    stacked = torch.stack(feat_list, dim=0)
                    fused = stacked.max(dim=0)[0]
                elif fusion == 'FedCat':
                    fused = torch.cat(feat_list, dim=1)
                elif fusion == 'FedAttention':
                    trainer.server.attention.eval()
                    stacked = torch.stack(feat_list, dim=1)
                    attn_scores = trainer.server.attention(stacked).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1)
                    fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
                elif fusion == 'FedWeighted':
                    weights = F.softmax(trainer.server.weights, dim=0)
                    stacked = torch.stack(feat_list, dim=1)
                    fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
                
                test_logits = trainer.server.classifier(fused)
                test_labels = trainer.test_labels_gpu
                
                # Split out validation set (20% from the training set)
                val_features = {}
                n_train = len(trainer.train_labels_gpu)
                val_size = int(n_train * 0.2)
                val_idx = torch.arange(n_train - val_size, n_train)
                
                for v, client in enumerate(trainer.clients):
                    val_features[v] = client.compute_feature_eval(trainer.train_data_gpu[v][val_idx])
                
                val_feat_list = [val_features[v] for v in sorted(val_features.keys())]
                if fusion == 'FedAvg':
                    val_fused = sum(val_feat_list) / len(val_feat_list)
                elif fusion == 'FedAdd':
                    val_fused = sum(val_feat_list)
                elif fusion == 'FedMul':
                    val_fused = val_feat_list[0]
                    for f in val_feat_list[1:]:
                        val_fused = val_fused * f
                elif fusion == 'FedMax':
                    stacked = torch.stack(val_feat_list, dim=0)
                    val_fused = stacked.max(dim=0)[0]
                elif fusion == 'FedCat':
                    val_fused = torch.cat(val_feat_list, dim=1)
                elif fusion == 'FedAttention':
                    stacked = torch.stack(val_feat_list, dim=1)
                    attn_scores = trainer.server.attention(stacked).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1)
                    val_fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
                elif fusion == 'FedWeighted':
                    weights = F.softmax(trainer.server.weights, dim=0)
                    stacked = torch.stack(val_feat_list, dim=1)
                    val_fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
                
                val_logits = trainer.server.classifier(val_fused)
                val_labels = trainer.train_labels_gpu[val_idx]
            
            # Original performance
            probs_np = F.softmax(test_logits, dim=1).cpu().numpy()
            labels_np = test_labels.cpu().numpy()
            acc, ece, overconf = compute_metrics(probs_np, labels_np)
            
            # TS optimization
            T_opt = find_optimal_temperature(val_logits, val_labels, criterion='nll')
            ts_probs = F.softmax(test_logits / T_opt, dim=1).cpu().numpy()
            _, ts_ece, _ = compute_metrics(ts_probs, labels_np)
            
            acc_list.append(acc * 100)
            ece_list.append(ece * 100)
            ts_ece_list.append(ts_ece * 100)
            
            print(f"  {fusion} Run {run+1}: Acc={acc*100:.2f}% ECE={ece*100:.2f}% +TS={ts_ece*100:.2f}% (T={T_opt:.2f})")
        
        mean_acc = np.mean(acc_list)
        mean_ece = np.mean(ece_list)
        mean_ts_ece = np.mean(ts_ece_list)
        all_results[fusion] = {
            'acc': mean_acc, 'ece': mean_ece, 'ts_ece': mean_ts_ece,
            'acc_std': np.std(acc_list), 'ece_std': np.std(ece_list), 'ts_ece_std': np.std(ts_ece_list)
        }
        print(f"  {fusion} MEAN: Acc={mean_acc:.2f}% ECE={mean_ece:.2f}% +TS={mean_ts_ece:.2f}%\n")
    
    # Summary
    print(f"\n{'='*100}")
    print(f" SUMMARY: Baseline Fusion + TS on {dataset_name}")
    print(f"{'='*100}")
    print(f"{'Method':<20}{'Acc (%)':<15}{'ECE (%)':<15}{'+TS ECE (%)':<15}{'ΔECE':<10}")
    print(f"{'-'*75}")
    for fusion in fusion_types:
        r = all_results[fusion]
        delta = r['ece'] - r['ts_ece']
        print(f"{fusion:<20}{r['acc']:.2f}±{r['acc_std']:.2f}    "
              f"{r['ece']:.2f}±{r['ece_std']:.2f}    "
              f"{r['ts_ece']:.2f}±{r['ts_ece_std']:.2f}    "
              f"{delta:+.2f}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--datasets', nargs='+', default=['Scene15', 'Caltech-101'])
    parser.add_argument('--fusions', nargs='+', default=['FedAvg', 'FedCat', 'FedAttention', 'FedMax'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    all_summary = {}
    for ds in args.datasets:
        try:
            results = run_baseline_ts(args.data_path, ds, args.fusions,
                                       gpu=args.gpu, epochs=args.epochs, runs=args.runs)
            all_summary[ds] = results
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}")
            traceback.print_exc()
    
    # Final overall summary
    print(f"\n\n{'='*120}")
    print(f" FINAL SUMMARY: All Baseline Fusion + TS Results")
    print(f"{'='*120}")
    for ds in args.datasets:
        if ds in all_summary:
            print(f"\n  {ds}:")
            for fusion, r in all_summary[ds].items():
                print(f"    {fusion:<20} Acc={r['acc']:.2f}%  ECE={r['ece']:.2f}%  +TS={r['ts_ece']:.2f}%")