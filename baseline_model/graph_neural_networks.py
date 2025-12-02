"""
Graph Neural Networks for Urban Renewal Prediction
===================================================

This script implements single-layer Graph Neural Networks for building-level urban renewal prediction:
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)

Models use building spatial relationships to improve prediction accuracy.
Features include 5-fold cross-validation and comprehensive evaluation metrics.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATv2Conv, GCNConv
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, average_precision_score,
                            confusion_matrix, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Batch
from datetime import timedelta
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set global font to Arial
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20
rcParams['font.weight'] = 'bold'

# Check GPU availability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class SingleLayerGCNNetwork(torch.nn.Module):
    """Single-layer GCN for building networks"""

    def __init__(self, num_building_features):
        super(SingleLayerGCNNetwork, self).__init__()

        self.building_conv1 = GCNConv(num_building_features, 256)
        self.building_conv2 = GCNConv(256, 128)
        self.building_conv3 = GCNConv(128, 2)

        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()

    def forward(self, building_x, building_edge_indices):
        batch_data_list = []
        global_indices = []

        for comm_id, data in building_edge_indices.items():
            if len(data['local_to_global']) > 0:
                local_building_x = building_x[data['local_to_global']]

                graph_data = Data(
                    x=local_building_x,
                    edge_index=data['edge_index']
                )

                batch_data_list.append(graph_data)
                global_indices.extend(data['local_to_global'].tolist())

        if batch_data_list:
            batch = Batch.from_data_list(batch_data_list).to(device)

            x = self.building_conv1(batch.x, batch.edge_index)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.building_conv2(x, batch.edge_index)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.building_conv3(x, batch.edge_index)

            combined_output = torch.zeros((building_x.size(0), 2), device=device)
            combined_output[global_indices] = x

            return F.log_softmax(combined_output, dim=1)
        else:
            return F.log_softmax(torch.zeros((building_x.size(0), 2), device=device), dim=1)


class SingleLayerGATNetwork(torch.nn.Module):
    """Single-layer GAT for building networks"""

    def __init__(self, num_building_features):
        super(SingleLayerGATNetwork, self).__init__()

        self.building_conv1 = GATv2Conv(num_building_features, 64, heads=8, dropout=0.3)
        self.building_conv2 = GATv2Conv(64*8, 32, heads=8, dropout=0.3)
        self.building_conv3 = GATv2Conv(32*8, 2, heads=1, dropout=0.3)

        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()

    def forward(self, building_x, building_edge_indices):
        batch_data_list = []
        global_indices = []

        for comm_id, data in building_edge_indices.items():
            if len(data['local_to_global']) > 0:
                local_building_x = building_x[data['local_to_global']]

                graph_data = Data(
                    x=local_building_x,
                    edge_index=data['edge_index']
                )

                batch_data_list.append(graph_data)
                global_indices.extend(data['local_to_global'].tolist())

        if batch_data_list:
            batch = Batch.from_data_list(batch_data_list).to(device)

            x = self.building_conv1(batch.x, batch.edge_index)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.building_conv2(x, batch.edge_index)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.building_conv3(x, batch.edge_index)

            combined_output = torch.zeros((building_x.size(0), 2), device=device)
            combined_output[global_indices] = x

            return F.log_softmax(combined_output, dim=1)
        else:
            return F.log_softmax(torch.zeros((building_x.size(0), 2), device=device), dim=1)


def construct_building_graphs_single_layer(df, default_k=5):
    """Construct building graphs for single-layer networks"""
    start_time = time.time()
    edge_indices = {}
    edge_weights = {}

    building_counts = []
    k_values = []
    edge_counts = []
    comm_without_buildings = 0
    total_communities = len(df['comm_id'].unique())

    for comm_id, group in df.groupby('comm_id'):
        if len(group) < 2:
            comm_without_buildings += 1
            continue

        # Get building coordinates
        coords = np.column_stack([
            group.geometry.centroid.x.astype(np.float32),
            group.geometry.centroid.y.astype(np.float32)
        ])

        k = min(default_k, len(group) - 1)

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        edge_list = []
        edge_weights_list = []

        local_to_global = group.index.tolist()

        for i in range(len(group)):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                edge_list.append([i, j])
                edge_weights_list.append(1.0 / (dist + 1e-6))

        if edge_list:
            edge_indices[comm_id] = {
                'edge_index': torch.tensor(edge_list, dtype=torch.long).t().to(device),
                'local_to_global': torch.tensor(local_to_global, dtype=torch.long).to(device)
            }
            edge_weights[comm_id] = torch.tensor(edge_weights_list, dtype=torch.float32).to(device)

            building_counts.append(len(group))
            k_values.append(k)
            edge_counts.append(len(edge_list))

    elapsed_time = time.time() - start_time

    if building_counts:
        print(f"Building graphs constructed in {timedelta(seconds=elapsed_time)}")
        print(f"Total communities: {total_communities}, successful graphs: {len(edge_indices)}")
    else:
        print("Warning: No building graphs were successfully constructed!")

    return edge_indices, edge_weights


def prepare_building_features(df):
    """Prepare building features for GNN models"""
    building_features = ['perimeter', 'floor', 'height', 'area']
    function_cols = [col for col in df.columns if 'function_' in col]
    roof_cols = [col for col in df.columns if 'roof_' in col]
    rs_features = [col for col in df.columns if col.startswith('hsr_')]

    features = df[building_features + function_cols + roof_cols + rs_features].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return torch.tensor(features, dtype=torch.float32).to(device)


def create_save_dir(save_path):
    """Create directory if it doesn't exist"""
    os.makedirs(save_path, exist_ok=True)


def interpolate_roc_curve(fpr, tpr, mean_fpr):
    """Interpolate TPR values at mean FPR points"""
    return np.interp(mean_fpr, fpr, tpr)


def interpolate_pr_curve(recall, precision, mean_recall):
    """Interpolate precision values at mean recall points"""
    return np.interp(mean_recall, recall[::-1], precision[::-1])


def plot_roc_curves(all_results, save_path='./results/'):
    """Plot ROC curves with all folds and mean curve"""
    create_save_dir(save_path)

    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.ravel()

    mean_fpr = np.linspace(0, 1, 100)

    for idx, model_name in enumerate(models):
        if idx >= len(axes):
            break

        results = all_results[model_name]

        tprs = []
        aucs = []

        # Plot individual fold curves
        for fold_idx, fold_result in enumerate(results):
            y_true = fold_result['y_true']
            y_pred_proba = fold_result['y_pred_proba']

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            # Interpolate this fold's curve
            interp_tpr = interpolate_roc_curve(fpr, tpr, mean_fpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            # Plot individual fold with transparency
            axes[idx].plot(fpr, tpr, alpha=0.5, linewidth=2.5,
                          label=f'Fold {fold_idx+1} (AUC = {roc_auc:.3f})')

        # Calculate mean and std
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Plot mean curve
        axes[idx].plot(mean_fpr, mean_tpr, color='darkblue', linewidth=4,
                      label=f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

        # Plot diagonal reference line
        axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

        axes[idx].set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        axes[idx].set_title(model_name, fontsize=16, fontweight='bold')
        axes[idx].legend(loc='lower right', fontsize=12)
        axes[idx].grid(False)

    fig.tight_layout()
    fig.savefig(f'{save_path}gnn_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curves saved to {save_path}gnn_roc_curves.png")


def plot_pr_curves(all_results, save_path='./results/'):
    """Plot PR curves with all folds and mean curve"""
    create_save_dir(save_path)

    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.ravel()

    mean_recall = np.linspace(0, 1, 100)

    for idx, model_name in enumerate(models):
        if idx >= len(axes):
            break

        results = all_results[model_name]

        precisions = []
        aps = []

        # Plot individual fold curves
        for fold_idx, fold_result in enumerate(results):
            y_true = fold_result['y_true']
            y_pred_proba = fold_result['y_pred_proba']

            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            aps.append(pr_auc)

            # Interpolate this fold's curve
            interp_precision = interpolate_pr_curve(recall, precision, mean_recall)
            precisions.append(interp_precision)

            # Plot individual fold with transparency
            axes[idx].plot(recall, precision, alpha=0.5, linewidth=2.5,
                          label=f'Fold {fold_idx+1} (AP = {pr_auc:.3f})')

        # Calculate mean and std
        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)

        # Plot mean curve
        axes[idx].plot(mean_recall, mean_precision, color='darkred', linewidth=4,
                      label=f'Mean (AP = {mean_ap:.3f} ± {std_ap:.3f})')

        axes[idx].set_xlabel('Recall', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Precision', fontsize=14, fontweight='bold')
        axes[idx].set_title(model_name, fontsize=16, fontweight='bold')
        axes[idx].legend(loc='lower left', fontsize=12)
        axes[idx].grid(False)

    fig.tight_layout()
    fig.savefig(f'{save_path}gnn_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PR curves saved to {save_path}gnn_pr_curves.png")


def train_gnn_model(model_class, model_name, building_x, building_edge_indices,
                    y_tensor, mask, train_idx, val_idx):
    """Train GNN model and return prediction results"""
    torch.cuda.empty_cache()

    # Initialize model
    model = model_class(num_building_features=building_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Setup weighted loss function
    class_counts = torch.bincount(y_tensor)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()
    criterion = torch.nn.NLLLoss(weight=weights)

    # Initialize early stopping
    best_val_f1 = 0
    best_model_state = None
    patience = 20
    counter = 0

    # Training loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(building_x, building_edge_indices)
        loss = criterion(out[mask][train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(building_x, building_edge_indices)
            pred = out[mask][val_idx].max(1)[1]
            val_f1 = f1_score(y_tensor[val_idx].cpu().numpy(), pred.cpu().numpy())

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1

        if epoch % 50 == 0:
            print(f"{model_name} - Epoch {epoch}: Loss = {loss.item():.4f}, Val F1 = {val_f1:.4f}")

        if counter >= patience:
            print(f"{model_name} - Early stopping at epoch {epoch}")
            break

    # Final evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(building_x, building_edge_indices)
        pred = out[mask][val_idx].max(1)[1].cpu().numpy()
        pred_proba = torch.exp(out[mask][val_idx][:, 1]).cpu().numpy()
        y_true = y_tensor[val_idx].cpu().numpy()

    return pred, pred_proba, y_true


def load_and_preprocess_data(data_path):
    """Load and preprocess the urban renewal dataset"""
    # Load dataset
    buildings_gdf = gpd.read_file(data_path)
    print(f"Dataset loaded successfully with {len(buildings_gdf)} buildings")

    # Process labels (convert to binary classification)
    y = buildings_gdf['renewal'].values
    mask = (y == 1) | (y == 2)
    y_train = y[mask].copy()
    y_train = np.where(y_train == 2, 0, y_train)
    print("Label distribution:", Counter(y_train))

    return buildings_gdf, mask, y_train


def main(data_path, save_path='./results/'):
    """Main function to train GNN models with comprehensive evaluation"""
    total_start_time = time.time()

    # Load and preprocess data
    print("Loading data...")
    buildings_gdf, mask, y_train = load_and_preprocess_data(data_path)

    # Build building graphs
    print("\nConstructing building graphs...")
    building_edge_indices, building_edge_weights = construct_building_graphs_single_layer(buildings_gdf)

    # Prepare building features
    print("\nPreparing building features...")
    building_x = prepare_building_features(buildings_gdf)
    print(f"Building features prepared with dimension: {building_x.shape}")

    # Prepare labels
    y_tensor = torch.LongTensor(y_train).to(device)

    # Define models
    models = {
        'GCN': SingleLayerGCNNetwork,
        'GAT': SingleLayerGATNetwork
    }

    # Store all results
    all_results = {model_name: [] for model_name in models.keys()}

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\nStarting 5-fold cross-validation training...")
    print("="*100)

    for fold, (train_idx, val_idx) in enumerate(kf.split(building_x[mask].cpu().numpy())):
        print(f"\n{'='*100}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*100}")

        for model_name, model_class in models.items():
            print(f"\nTraining {model_name} model...")
            fold_start_time = time.time()

            pred, pred_proba, y_true = train_gnn_model(
                model_class, model_name, building_x, building_edge_indices,
                y_tensor, mask, train_idx, val_idx
            )

            # Calculate evaluation metrics
            acc = accuracy_score(y_true, pred)
            prec = precision_score(y_true, pred)
            rec = recall_score(y_true, pred)
            f1 = f1_score(y_true, pred)
            roc_auc = roc_auc_score(y_true, pred_proba)
            pr_auc = average_precision_score(y_true, pred_proba)

            fold_time = time.time() - fold_start_time

            # Store results
            all_results[model_name].append({
                'fold': fold + 1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'y_true': y_true,
                'y_pred': pred,
                'y_pred_proba': pred_proba,
                'training_time': fold_time
            })

            print(f"{model_name} - Fold {fold+1} Results:")
            print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
            print(f"  F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
            print(f"  Training time: {timedelta(seconds=fold_time)}")

    # Print average results
    print("\n" + "="*100)
    print("Average Results Across All Folds")
    print("="*100)

    summary_data = []
    for model_name in models.keys():
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'y_pred_proba']}
            for r in all_results[model_name]
        ])

        avg_results = results_df.mean()
        std_results = results_df.std()

        print(f"\n{model_name}:")
        print(f"  Accuracy:   {avg_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}")
        print(f"  Precision:  {avg_results['precision']:.4f} ± {std_results['precision']:.4f}")
        print(f"  Recall:     {avg_results['recall']:.4f} ± {std_results['recall']:.4f}")
        print(f"  F1 Score:   {avg_results['f1']:.4f} ± {std_results['f1']:.4f}")
        print(f"  ROC-AUC:    {avg_results['roc_auc']:.4f} ± {std_results['roc_auc']:.4f}")
        print(f"  PR-AUC:     {avg_results['pr_auc']:.4f} ± {std_results['pr_auc']:.4f}")
        print(f"  Avg Training Time: {timedelta(seconds=avg_results['training_time'])}")

        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{avg_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}",
            'Precision': f"{avg_results['precision']:.4f} ± {std_results['precision']:.4f}",
            'Recall': f"{avg_results['recall']:.4f} ± {std_results['recall']:.4f}",
            'F1': f"{avg_results['f1']:.4f} ± {std_results['f1']:.4f}",
            'ROC-AUC': f"{avg_results['roc_auc']:.4f} ± {std_results['roc_auc']:.4f}",
            'PR-AUC': f"{avg_results['pr_auc']:.4f} ± {std_results['pr_auc']:.4f}"
        })

    # Save summary table
    create_save_dir(save_path)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{save_path}gnn_model_comparison_summary.csv', index=False)
    print(f"\nSummary table saved to {save_path}gnn_model_comparison_summary.csv")

    # Generate all visualizations
    print("\n" + "="*100)
    print("Generating visualizations...")
    print("="*100)

    plot_roc_curves(all_results, save_path)
    plot_pr_curves(all_results, save_path)

    total_time = time.time() - total_start_time
    print(f"\nTotal runtime: {timedelta(seconds=total_time)}")
    print("\nAll visualizations and evaluations completed successfully!")


if __name__ == "__main__":
    # Update this path to your actual data file
    data_path = "path/to/your/buildings_data.shp"  # Replace with actual path
    main(data_path)