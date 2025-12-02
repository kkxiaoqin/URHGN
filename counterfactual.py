"""
Counterfactual analysis for URHGN model component contribution assessment.

This script performs ablation studies to understand the contribution of different
model components (edges, attributes, layers) to prediction performance.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F

# Import the URHGN model
from models.urhgn import URHGNModel


def get_device(device_id=None):
    """Get appropriate device for analysis."""
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if device_id is not None:
        device_id = min(device_id, torch.cuda.device_count() - 1)
        return torch.device(f'cuda:{device_id}')

    return torch.device('cuda:0')


def construct_community_graph(community_gdf):
    """Construct community adjacency graph."""
    edge_list = []
    for idx, comm in community_gdf.iterrows():
        neighbors = community_gdf[community_gdf.geometry.touches(comm.geometry)].index
        for neighbor in neighbors:
            edge_list.append([idx, neighbor])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Prepare community features
    poi_cols = [f'POI_{i}' for i in range(1, 17)]
    other_cols = ['population', 'green_ratio', 'price']
    available_features = [col for col in poi_cols + other_cols if col in community_gdf.columns]

    if not available_features:
        # Fallback to common column names
        available_features = [col for col in community_gdf.columns if community_gdf[col].dtype in ['int64', 'float64']][:19]

    features = community_gdf[available_features].values

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    x = torch.tensor(features, dtype=torch.float32)

    return edge_index, x, available_features


def construct_building_graphs(buildings_gdf, k_neighbors=5):
    """Construct building graphs within each community."""
    edge_indices = {}

    if 'community_id' not in buildings_gdf.columns:
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf['community_id'] = 0

    for comm_id, group in buildings_gdf.groupby('community_id'):
        if len(group) <= 1:
            edge_indices[comm_id] = {
                'edge_index': torch.tensor([[], []], dtype=torch.long),
                'local_to_global': torch.tensor(group.index.tolist(), dtype=torch.long)
            }
            continue

        coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in group.geometry])
        actual_k = min(k_neighbors, len(coords) - 1)

        if actual_k > 0:
            nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='ball_tree').fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            local_edges = []
            for i in range(len(indices)):
                for j in indices[i][1:]:
                    local_edges.append([i, j])

            if local_edges:
                edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        edge_indices[comm_id] = {
            'edge_index': edge_index,
            'local_to_global': torch.tensor(group.index.tolist(), dtype=torch.long)
        }

    return edge_indices


def prepare_counterfactual_data(buildings_path, community_path):
    """Prepare data for counterfactual analysis."""
    # Load data
    buildings_gdf = gpd.read_file(buildings_path)
    community_gdf = gpd.read_file(community_path)

    # Create building to community mapping
    buildings_gdf = buildings_gdf.to_crs(community_gdf.crs)
    buildings_with_comm = gpd.sjoin(buildings_gdf, community_gdf, how='left', predicate='intersects')

    if 'index_right' not in buildings_with_comm.columns:
        buildings_with_comm['community_id'] = 0
    else:
        buildings_with_comm['community_id'] = buildings_with_comm['index_right'].fillna(0).astype(int)

    # Filter and prepare labels
    buildings_with_comm = buildings_with_comm[buildings_with_comm['renewal'].isin([1, 2])]
    buildings_with_comm['label'] = (buildings_with_comm['renewal'] == 1).astype(int)

    # Construct graphs
    building_graphs = construct_building_graphs(buildings_with_comm, k_neighbors=5)
    community_edge_index, community_features, community_feature_names = construct_community_graph(community_gdf)

    # Prepare building features
    building_features = ['perimeter', 'floor', 'height', 'area',
                        'function_commercial', 'function_residential', 'function_industrial',
                        'function_public', 'function_others', 'roof_concrete', 'roof_tile',
                        'roof_metal', 'roof_others']

    available_building_features = [f for f in building_features if f in buildings_with_comm.columns]
    if not available_building_features:
        available_building_features = [col for col in buildings_with_comm.columns
                                     if buildings_with_comm[col].dtype in ['int64', 'float64']][:13]

    building_scaler = StandardScaler()
    building_features_array = building_scaler.fit_transform(buildings_with_comm[available_building_features])
    building_features_tensor = torch.tensor(building_features_array, dtype=torch.float32)

    # Create mapping tensor
    building_to_comm = torch.tensor(buildings_with_comm['community_id'].values, dtype=torch.long)
    labels = torch.tensor(buildings_with_comm['label'].values, dtype=torch.float32)

    return {
        'building_features': building_features_tensor,
        'community_features': community_features,
        'building_graphs': building_graphs,
        'community_graph': community_edge_index,
        'building_to_comm': building_to_comm,
        'labels': labels,
        'building_feature_names': available_building_features,
        'community_feature_names': community_feature_names
    }


class CounterfactualURHGNModel(nn.Module):
    """Counterfactual version of URHGN model with configurable components."""

    def __init__(self, original_model, remove_building_edges=False, remove_community_edges=False,
                 remove_building_attrs=False, remove_community_attrs=False):
        super(CounterfactualURHGNModel, self).__init__()

        self.remove_building_edges = remove_building_edges
        self.remove_community_edges = remove_community_edges
        self.remove_building_attrs = remove_building_attrs
        self.remove_community_attrs = remove_community_attrs

        # Copy model architecture
        self.comm_conv1 = original_model.comm_conv1
        self.comm_conv2 = original_model.comm_conv2
        self.feature_attention = original_model.feature_attention
        self.building_conv1 = original_model.building_conv1
        self.building_conv2 = original_model.building_conv2
        self.building_conv3 = original_model.building_conv3
        self.dropout = original_model.dropout

    def forward(self, building_features, building_edge_indices, community_features,
                community_edge_index, building_to_comm_mapping):
        # Modify inputs based on configuration
        if self.remove_community_attrs:
            community_features = torch.zeros_like(community_features)

        modified_community_edge_index = community_edge_index
        if self.remove_community_edges:
            modified_community_edge_index = torch.tensor([[], []], dtype=torch.long)

        # Process community layer
        comm_x = self.comm_conv1(community_features, modified_community_edge_index)
        comm_x = F.relu(comm_x)
        comm_x = self.dropout(comm_x)

        comm_x = self.comm_conv2(comm_x, modified_community_edge_index)
        comm_x = F.relu(comm_x)
        comm_x = self.dropout(comm_x)

        # Map community features to buildings
        building_comm_features = comm_x[building_to_comm_mapping]

        if self.remove_building_attrs:
            building_features = torch.zeros_like(building_features)

        # Apply attention mechanism
        weighted_building, weighted_comm = self.feature_attention(
            building_features, building_comm_features
        )

        fused_building_features = torch.cat([weighted_building, weighted_comm], dim=1)

        # Process building layer
        batch_data_list = []
        global_indices = []

        for comm_id, data in building_edge_indices.items():
            if len(data['local_to_global']) > 0:
                local_building_x = fused_building_features[data['local_to_global']]

                modified_edge_index = data['edge_index']
                if self.remove_building_edges:
                    modified_edge_index = torch.tensor([[], []], dtype=torch.long)

                graph_data = {
                    'x': local_building_x,
                    'edge_index': modified_edge_index
                }

                from torch_geometric.data import Data, Batch
                data_obj = Data(x=graph_data['x'], edge_index=graph_data['edge_index'])
                batch_data_list.append(data_obj)
                global_indices.extend(data['local_to_global'].tolist())

        if batch_data_list:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(batch_data_list)

            x = self.building_conv1(batch.x, batch.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.building_conv2(x, batch.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.building_conv3(x, batch.edge_index)

            device = next(self.parameters()).device
            combined_output = torch.zeros((building_features.size(0), 2), device=device)
            combined_output[global_indices] = x

            return F.log_softmax(combined_output, dim=1)
        else:
            device = next(self.parameters()).device
            return F.log_softmax(torch.zeros((building_features.size(0), 2), device=device), dim=1)


def evaluate_experiment(model, data, experiment_name, device):
    """Evaluate a single counterfactual experiment."""
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

        metrics = {
            'experiment': experiment_name,
            'accuracy': accuracy_score(labels_np, predictions),
            'precision': precision_score(labels_np, predictions, zero_division=0),
            'recall': recall_score(labels_np, predictions, zero_division=0),
            'f1': f1_score(labels_np, predictions, zero_division=0),
            'roc_auc': roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0,
            'probabilities': probs,
            'predictions': predictions,
            'labels': labels_np
        }

        return metrics


def run_counterfactual_analysis(model_path, buildings_path, community_path, device_id=None):
    """Run complete counterfactual analysis."""
    # Setup device
    device = get_device(device_id)
    print(f"Using device: {device}")

    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)

    num_building_features = 13  # Adjust based on your data
    num_community_features = 19  # Adjust based on your data

    original_model = URHGNModel(num_building_features, num_community_features, device=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    original_model.load_state_dict(state_dict, strict=False)
    original_model.to(device)
    original_model.eval()

    # Prepare data
    print("Preparing data...")
    data = prepare_counterfactual_data(buildings_path, community_path)

    # Define experiments
    experiments = {
        'E0_Full': {
            'remove_building_edges': False,
            'remove_community_edges': False,
            'remove_building_attrs': False,
            'remove_community_attrs': False
        },
        'E1_No_Edges': {
            'remove_building_edges': True,
            'remove_community_edges': True,
            'remove_building_attrs': False,
            'remove_community_attrs': False
        },
        'E2_No_Building_Edges': {
            'remove_building_edges': True,
            'remove_community_edges': False,
            'remove_building_attrs': False,
            'remove_community_attrs': False
        },
        'E3_No_Community_Edges': {
            'remove_building_edges': False,
            'remove_community_edges': True,
            'remove_building_attrs': False,
            'remove_community_attrs': False
        },
        'E4_No_Building_Attrs': {
            'remove_building_edges': False,
            'remove_community_edges': False,
            'remove_building_attrs': True,
            'remove_community_attrs': False
        },
        'E5_No_Community_Attrs': {
            'remove_building_edges': False,
            'remove_community_edges': False,
            'remove_building_attrs': False,
            'remove_community_attrs': True
        }
    }

    print("Running counterfactual experiments...")
    all_results = []

    for exp_name, config in experiments.items():
        print(f"\nRunning {exp_name}...")
        print("-" * 30)

        # Create counterfactual model
        cf_model = CounterfactualURHGNModel(original_model, **config)
        cf_model.to(device)

        # Evaluate
        metrics = evaluate_experiment(cf_model, data, exp_name, device)
        all_results.append(metrics)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Calculate prediction changes
    baseline_probs = all_results[0]['probabilities']
    baseline_labels = all_results[0]['labels']

    detailed_results = []

    for i, result in enumerate(all_results):
        exp_name = result['experiment']
        probs = result['probabilities']

        # Calculate absolute prediction probability change
        abs_change = np.mean(np.abs(probs - baseline_probs))

        # Calculate flip rate
        baseline_preds = (baseline_probs > 0.5).astype(int)
        exp_preds = (probs > 0.5).astype(int)
        flip_rate = np.mean(baseline_preds != exp_preds)

        detailed_result = result.copy()
        detailed_result['abs_pred_change'] = abs_change
        detailed_result['prediction_flip'] = flip_rate
        detailed_results.append(detailed_result)

    # Save results
    print("\nSaving results...")
    os.makedirs("counterfactual_results", exist_ok=True)

    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv("counterfactual_results/experiment_results.csv", index=False)

    # Save predictions for detailed analysis
    predictions_df = pd.DataFrame({
        'baseline_prob': baseline_probs,
        'baseline_label': baseline_labels
    })

    for i, result in enumerate(all_results[1:], 1):
        exp_name = result['experiment']
        predictions_df[f'{exp_name}_prob'] = result['probabilities']
        predictions_df[f'{exp_name}_label'] = result['predictions']

    predictions_df.to_csv("counterfactual_results/predictions_comparison.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("COUNTERFACTUAL ANALYSIS SUMMARY")
    print("="*60)

    baseline_auc = all_results[0]['roc_auc']

    print(f"\nBaseline Model Performance (E0_Full):")
    print(f"  ROC-AUC: {baseline_auc:.4f}")
    print(f"  Accuracy: {all_results[0]['accuracy']:.4f}")

    print(f"\nComponent Contributions:")
    for result in detailed_results[1:]:
        exp_name = result['experiment']
        auc_drop = baseline_auc - result['roc_auc']
        flip_rate = result['prediction_flip']

        component_names = {
            'E1_No_Edges': 'All Graph Edges',
            'E2_No_Building_Edges': 'Building Graph Edges',
            'E3_No_Community_Edges': 'Community Graph Edges',
            'E4_No_Building_Attrs': 'Building Attributes',
            'E5_No_Community_Attrs': 'Community Attributes'
        }

        component = component_names.get(exp_name, exp_name)
        contribution = (auc_drop / baseline_auc) * 100 if baseline_auc > 0 else 0

        print(f"  {component}:")
        print(f"    AUC Drop: {auc_drop:.4f} ({contribution:.1f}%)")
        print(f"    Flip Rate: {flip_rate:.3f}")

    print("="*60)
    print("Results saved to 'counterfactual_results/' directory")

    return detailed_results


if __name__ == "__main__":
    # Configuration
    model_path = "results/best_model.pth"  # Path to your trained model
    buildings_path = "data/shp/buildings.shp"
    community_path = "data/shp/community.shp"

    # Set device ID if you want to use a specific GPU
    device_id = None  # Auto-select

    # Run counterfactual analysis
    results = run_counterfactual_analysis(
        model_path, buildings_path, community_path,
        device_id=device_id
    )