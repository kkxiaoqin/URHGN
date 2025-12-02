"""
Training script for URHGN model.

This script provides cross-validation training with comprehensive evaluation
and visualization capabilities.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, average_precision_score,
                            confusion_matrix, roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Batch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

# Import the URHGN model
from models.urhgn import URHGNModel

# Configure matplotlib
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 12


def get_device(device_id=None):
    """
    Get appropriate device for training.

    Args:
        device_id: Specific GPU ID (e.g., 0, 1). If None, auto-select.

    Returns:
        torch.device object
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if device_id is not None:
        device_id = min(device_id, torch.cuda.device_count() - 1)
        return torch.device(f'cuda:{device_id}')

    # Auto-select GPU with most available memory
    best_gpu = 0
    max_memory = 0
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        if free_memory > max_memory:
            max_memory = free_memory
            best_gpu = i

    return torch.device(f'cuda:{best_gpu}')


def construct_community_graph(community_gdf):
    """Construct community adjacency graph based on spatial contiguity."""
    edge_list = []

    for i, comm1 in community_gdf.iterrows():
        for j, comm2 in community_gdf.iterrows():
            if i < j:
                try:
                    if comm1.geometry.touches(comm2.geometry):
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                except:
                    continue

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)

    return edge_index


def construct_building_graphs(buildings_gdf, k_neighbors=5):
    """Construct k-NN building graphs for each community."""
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in buildings_gdf.geometry])

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)), algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    community_groups = buildings_gdf.groupby('community_id') if 'community_id' in buildings_gdf.columns else [(0, buildings_gdf)]

    building_graphs = {}

    for comm_id, group in community_groups:
        group_indices = group.index.tolist()
        global_to_local = {global_idx: i for i, global_idx in enumerate(group_indices)}

        if len(group_indices) <= 1:
            building_graphs[comm_id] = {
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'local_to_global': torch.tensor(group_indices, dtype=torch.long)
            }
            continue

        local_edges = []
        for i, global_idx in enumerate(group_indices):
            neighbors = indices[global_idx][1:]
            for neighbor_idx in neighbors:
                if neighbor_idx in global_to_local:
                    local_edges.append([i, global_to_local[neighbor_idx]])

        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        building_graphs[comm_id] = {
            'edge_index': edge_index,
            'local_to_global': torch.tensor(group_indices, dtype=torch.long)
        }

    return building_graphs


def prepare_data(buildings_path, community_path, k_neighbors=5):
    """Prepare training data from shapefiles."""
    # Load data
    buildings_gdf = gpd.read_file(buildings_path)
    community_gdf = gpd.read_file(community_path)

    # Features
    building_features = ['perimeter', 'floor', 'height', 'area',
                        'function_commercial', 'function_residential', 'function_industrial',
                        'function_public', 'function_others', 'roof_concrete', 'roof_tile',
                        'roof_metal', 'roof_others']

    community_features = ['POI_1', 'POI_2', 'POI_3', 'POI_4', 'POI_5', 'POI_6', 'POI_7',
                         'POI_8', 'POI_9', 'POI_10', 'POI_11', 'POI_12', 'POI_13', 'POI_14',
                         'POI_15', 'POI_16', 'population', 'green_ratio', 'price']

    # Process building data
    building_scaler = StandardScaler()
    available_building_features = [f for f in building_features if f in buildings_gdf.columns]
    buildings_gdf[available_building_features] = building_scaler.fit_transform(buildings_gdf[available_building_features])

    # Process community data
    community_scaler = StandardScaler()
    available_community_features = [f for f in community_features if f in community_gdf.columns]
    community_gdf[available_community_features] = community_scaler.fit_transform(community_gdf[available_community_features])

    # Create building to community mapping
    buildings_gdf = buildings_gdf.to_crs(community_gdf.crs)
    buildings_with_comm = gpd.sjoin(buildings_gdf, community_gdf, how='left', predicate='intersects')

    if 'index_right' not in buildings_with_comm.columns:
        buildings_with_comm['community_id'] = 0
    else:
        buildings_with_comm['community_id'] = buildings_with_comm['index_right'].fillna(0).astype(int)

    # Prepare labels
    buildings_with_comm = buildings_with_comm[buildings_with_comm['renewal'].isin([1, 2])]
    buildings_with_comm['label'] = (buildings_with_comm['renewal'] == 1).astype(int)

    # Construct graphs
    building_graphs = construct_building_graphs(buildings_with_comm, k_neighbors)
    community_graph = construct_community_graph(community_gdf)

    # Prepare tensors
    building_features_tensor = torch.tensor(buildings_with_comm[available_building_features].values, dtype=torch.float32)
    community_features_tensor = torch.tensor(community_gdf[available_community_features].values, dtype=torch.float32)
    labels = torch.tensor(buildings_with_comm['label'].values, dtype=torch.float32)
    building_to_comm = torch.tensor(buildings_with_comm['community_id'].values, dtype=torch.long)

    return {
        'building_features': building_features_tensor,
        'community_features': community_features_tensor,
        'building_graphs': building_graphs,
        'community_graph': community_graph,
        'building_to_comm': building_to_comm,
        'labels': labels,
        'building_feature_names': available_building_features,
        'community_feature_names': available_community_features
    }


def evaluate_model(model, data, device):
    """Evaluate model performance."""
    model.eval()
    with torch.no_grad():
        building_features = data['building_features'].to(device)
        community_features = data['community_features'].to(device)
        building_to_comm = data['building_to_comm'].to(device)
        labels = data['labels'].to(device)

        outputs = model(building_features, data['building_graphs'],
                       community_features, data['community_graph'], building_to_comm)

        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        predictions = (probs > 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(labels_np, predictions)
        precision = precision_score(labels_np, predictions, zero_division=0)
        recall = recall_score(labels_np, predictions, zero_division=0)
        f1 = f1_score(labels_np, predictions, zero_division=0)
        roc_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'probabilities': probs,
            'labels': labels_np
        }


def plot_results(metrics_list, save_dir=None):
    """Plot training and evaluation results."""
    if not metrics_list:
        return

    # Aggregate metrics
    accuracies = [m['accuracy'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    roc_aucs = [m['roc_auc'] for m in metrics_list]

    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cross-Validation Results', fontsize=16, fontweight='bold')

    axes[0, 0].boxplot([accuracies, precisions, f1_scores], labels=['Accuracy', 'Precision', 'F1-Score'])
    axes[0, 0].set_title('Performance Metrics Distribution')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].boxplot([roc_aucs], labels=['ROC-AUC'])
    axes[0, 1].set_title('ROC-AUC Distribution')
    axes[0, 1].set_ylabel('AUC Score')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot average ROC curve
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i, metrics in enumerate(metrics_list):
        if len(np.unique(metrics['labels'])) > 1:
            fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            axes[1, 0].plot(fpr, tpr, alpha=0.3, label=f'Fold {i+1}')

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        axes[1, 0].plot(mean_fpr, mean_tpr, 'r-', linewidth=2, label='Mean ROC')
        axes[1, 0].fill_between(mean_fpr, np.mean(tprs, axis=0) - np.std(tprs, axis=0),
                               np.mean(tprs, axis=0) + np.std(tprs, axis=0), alpha=0.2)

    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Performance summary table
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Mean': [np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                np.mean(f1_scores), np.mean(roc_aucs)],
        'Std': [np.std(accuracies), np.std(precisions), np.std(recalls),
                np.std(f1_scores), np.std(roc_aucs)]
    })

    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=metrics_summary.round(4).values,
                             colLabels=metrics_summary.columns,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Performance Summary', pad=20)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'cv_results.png'), dpi=300, bbox_inches='tight')

    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print("="*60)


def train_model(device_id=None):
    """Main training function with cross-validation."""
    # Setup device
    device = get_device(device_id)
    print(f"Using device: {device}")

    # Data paths
    buildings_path = "data/shp/buildings.shp"
    community_path = "data/shp/community.shp"

    # Prepare data
    print("Loading and preprocessing data...")
    data = prepare_data(buildings_path, community_path, k_neighbors=5)

    # Cross-validation setup
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_metrics = []
    fold = 1

    print(f"\nStarting {k_folds}-fold cross-validation...")
    print("-" * 50)

    for train_idx, val_idx in kf.split(data['labels']):
        print(f"\nFold {fold}/{k_folds}")
        print("-" * 20)

        # Split data
        train_data = {}
        val_data = {}

        # Copy non-indexed data
        for key in ['community_features', 'community_graph', 'building_feature_names', 'community_feature_names']:
            if key in data:
                train_data[key] = data[key]
                val_data[key] = data[key]

        # Split indexed data
        train_data['building_features'] = data['building_features'][train_idx]
        train_data['labels'] = data['labels'][train_idx]
        train_data['building_to_comm'] = data['building_to_comm'][train_idx]

        val_data['building_features'] = data['building_features'][val_idx]
        val_data['labels'] = data['labels'][val_idx]
        val_data['building_to_comm'] = data['building_to_comm'][val_idx]

        # For simplicity, we'll use the same building graphs (in practice, you'd want to reconstruct them)
        train_data['building_graphs'] = data['building_graphs']
        val_data['building_graphs'] = data['building_graphs']

        # Initialize model
        model = URHGNModel(
            num_building_features=len(data['building_feature_names']),
            num_community_features=len(data['community_feature_names']),
            device=device
        ).to(device)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        num_epochs = 200
        patience = 20
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            train_outputs = model(
                train_data['building_features'].to(device),
                train_data['building_graphs'],
                train_data['community_features'].to(device),
                train_data['community_graph'],
                train_data['building_to_comm'].to(device)
            )

            train_loss = criterion(train_outputs, train_data['labels'].to(device).long())
            train_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(
                    val_data['building_features'].to(device),
                    val_data['building_graphs'],
                    val_data['community_features'].to(device),
                    val_data['community_graph'],
                    val_data['building_to_comm'].to(device)
                )

                val_loss = criterion(val_outputs, val_data['labels'].to(device).long())

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate
        metrics = evaluate_model(model, val_data, device)
        all_metrics.append(metrics)

        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        fold += 1

    # Plot and save results
    plot_results(all_metrics, save_dir="results")

    return all_metrics


if __name__ == "__main__":
    # Set device ID if you want to use a specific GPU (e.g., device_id=0 or device_id=1)
    # device_id=None  # Auto-select GPU
    device_id = 0   # Use first GPU

    metrics = train_model(device_id)