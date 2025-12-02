"""
GNNExplainer analysis for URHGN model interpretability.

This script provides model explainability using GNNExplainer to understand
feature importance and neighborhood influences in urban renewal predictions.
"""

import os
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode, ModelTaskLevel, ModelReturnType
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the URHGN model
from models.urhgn import URHGNModel


def get_device(device_id=None):
    """Get appropriate device for explanation."""
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if device_id is not None:
        device_id = min(device_id, torch.cuda.device_count() - 1)
        return torch.device(f'cuda:{device_id}')

    return torch.device('cuda:0')


class CommunityGNNWrapper(torch.nn.Module):
    """Community layer GNN wrapper for GNNExplainer."""

    def __init__(self, original_model):
        super(CommunityGNNWrapper, self).__init__()
        self.comm_conv1 = original_model.comm_conv1
        self.comm_conv2 = original_model.comm_conv2
        self.dropout = original_model.dropout

    def forward(self, x, edge_index):
        x = self.comm_conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.comm_conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = torch.nn.Linear(x.shape[1], 2).to(x.device)(x)
        return F.log_softmax(x, dim=1)


class BuildingGNNWrapper(torch.nn.Module):
    """Building layer GNN wrapper for GNNExplainer."""

    def __init__(self, original_model, input_dim):
        super(BuildingGNNWrapper, self).__init__()
        self.building_conv1 = original_model.building_conv1
        self.building_conv2 = original_model.building_conv2
        self.building_conv3 = original_model.building_conv3
        self.dropout = original_model.dropout

    def forward(self, x, edge_index):
        x = self.building_conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.building_conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.building_conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def load_data_for_explanation(buildings_path, community_path):
    """Load and prepare data for explanation."""
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

    return (buildings_with_comm, community_gdf, available_building_features,
            available_community_features, building_scaler, community_scaler)


def construct_graphs_for_explanation(buildings_gdf, community_gdf, k_neighbors=5):
    """Construct graphs for explanation."""
    # Construct building graphs (simplified version)
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in buildings_gdf.geometry])
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)), algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edge_list = []
    for i in range(len(coords)):
        for j in indices[i][1:]:  # Skip self
            edge_list.append([i, j])
            edge_list.append([j, i])

    building_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Construct community graph
    community_edge_list = []
    for i, comm1 in community_gdf.iterrows():
        for j, comm2 in community_gdf.iterrows():
            if i < j:
                try:
                    if comm1.geometry.touches(comm2.geometry):
                        community_edge_list.append([i, j])
                        community_edge_list.append([j, i])
                except:
                    continue

    if community_edge_list:
        community_edge_index = torch.tensor(community_edge_list, dtype=torch.long).t().contiguous()
    else:
        community_edge_index = torch.tensor([[], []], dtype=torch.long)

    return building_edge_index, community_edge_index


def explain_community_layer(model, community_features, community_edge_index, device, num_samples=50):
    """Explain community layer using GNNExplainer."""
    # Create wrapper model
    wrapper_model = CommunityGNNWrapper(model)

    # Configure explainer
    model_config = ModelConfig(
        mode=ModelMode.classification,
        task_level=ModelTaskLevel.node,
        return_type=ModelReturnType.log_probs
    )

    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config
    )

    # Select sample nodes for explanation
    num_nodes = community_features.size(0)
    sample_indices = np.random.choice(num_nodes, min(num_samples, num_nodes), replace=False)

    all_feature_importance = []
    all_edge_importance = []

    for node_idx in sample_indices:
        explanation = explainer(
            x=community_features,
            edge_index=community_edge_index,
            node_idx=node_idx
        )

        # Get feature importance
        if hasattr(explanation, 'node_mask'):
            feature_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
            all_feature_importance.append(feature_importance)

        # Get edge importance
        if hasattr(explanation, 'edge_mask'):
            edge_importance = explanation.edge_mask.cpu().numpy()
            all_edge_importance.append(edge_importance)

    return {
        'feature_importance': np.mean(all_feature_importance, axis=0) if all_feature_importance else None,
        'edge_importance': np.mean(all_edge_importance, axis=0) if all_edge_importance else None,
        'sample_indices': sample_indices
    }


def explain_building_layer(model, building_features, building_edge_index, device, num_samples=50):
    """Explain building layer using GNNExplainer."""
    # Create wrapper model
    input_dim = building_features.size(1)
    wrapper_model = BuildingGNNWrapper(model, input_dim)

    # Configure explainer
    model_config = ModelConfig(
        mode=ModelMode.classification,
        task_level=ModelTaskLevel.node,
        return_type=ModelReturnType.log_probs
    )

    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config
    )

    # Select sample nodes for explanation
    num_nodes = building_features.size(0)
    sample_indices = np.random.choice(num_nodes, min(num_samples, num_nodes), replace=False)

    all_feature_importance = []
    all_edge_importance = []

    for node_idx in sample_indices:
        try:
            explanation = explainer(
                x=building_features,
                edge_index=building_edge_index,
                node_idx=node_idx
            )

            # Get feature importance
            if hasattr(explanation, 'node_mask'):
                feature_importance = explanation.node_mask.mean(dim=0).cpu().numpy()
                all_feature_importance.append(feature_importance)

            # Get edge importance
            if hasattr(explanation, 'edge_mask'):
                edge_importance = explanation.edge_mask.cpu().numpy()
                all_edge_importance.append(edge_importance)

        except Exception as e:
            print(f"Warning: Failed to explain node {node_idx}: {e}")
            continue

    return {
        'feature_importance': np.mean(all_feature_importance, axis=0) if all_feature_importance else None,
        'edge_importance': np.mean(all_edge_importance, axis=0) if all_edge_importance else None,
        'sample_indices': sample_indices
    }


def run_explanation(model_path, buildings_path, community_path, device_id=None, sample_ratio=0.1):
    """Run GNNExplainer analysis."""
    # Setup device
    device = get_device(device_id)
    print(f"Using device: {device}")

    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model architecture from checkpoint or use default
    num_building_features = 13  # Adjust based on your actual features
    num_community_features = 19  # Adjust based on your actual features

    model = URHGNModel(num_building_features, num_community_features, device=device)

    # Load state dict (handling potential key differences)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Load and prepare data
    print("Loading data for explanation...")
    buildings_gdf, community_gdf, building_feature_names, community_feature_names, _, _ = load_data_for_explanation(
        buildings_path, community_path
    )

    # Construct graphs
    building_edge_index, community_edge_index = construct_graphs_for_explanation(
        buildings_gdf, community_gdf, k_neighbors=5
    )

    # Prepare feature tensors
    building_scaler = StandardScaler()
    community_scaler = StandardScaler()

    building_features = building_scaler.fit_transform(buildings_gdf[building_feature_names])
    community_features = community_scaler.fit_transform(community_gdf[community_feature_names])

    building_features_tensor = torch.tensor(building_features, dtype=torch.float32)
    community_features_tensor = torch.tensor(community_features, dtype=torch.float32)

    # Calculate sample sizes
    num_building_samples = max(10, int(len(buildings_gdf) * sample_ratio))
    num_community_samples = max(5, int(len(community_gdf) * sample_ratio))

    print(f"Explaining {num_building_samples} buildings and {num_community_samples} communities...")

    # Explain community layer
    print("Explaining community layer...")
    community_results = explain_community_layer(
        model, community_features_tensor.to(device), community_edge_index,
        device, num_samples=num_community_samples
    )

    # Explain building layer
    print("Explaining building layer...")
    building_results = explain_building_layer(
        model, building_features_tensor.to(device), building_edge_index,
        device, num_samples=num_building_samples
    )

    # Save results
    print("Saving results...")
    os.makedirs("explanations", exist_ok=True)

    # Save feature importance
    if community_results['feature_importance'] is not None:
        community_feature_importance = pd.DataFrame({
            'feature': community_feature_names,
            'importance': community_results['feature_importance']
        }).sort_values('importance', ascending=False)
        community_feature_importance.to_csv("explanations/community_feature_importance.csv", index=False)

    if building_results['feature_importance'] is not None:
        building_feature_importance = pd.DataFrame({
            'feature': building_feature_names,
            'importance': building_results['feature_importance']
        }).sort_values('importance', ascending=False)
        building_feature_importance.to_csv("explanations/building_feature_importance.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("GNNESTAINER RESULTS SUMMARY")
    print("="*60)

    if community_results['feature_importance'] is not None:
        print("\nTop 5 Most Important Community Features:")
        top_community = community_feature_importance.head()
        for _, row in top_community.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    if building_results['feature_importance'] is not None:
        print("\nTop 5 Most Important Building Features:")
        top_building = building_feature_importance.head()
        for _, row in top_building.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    print("="*60)
    print("Results saved to 'explanations/' directory")

    return community_results, building_results


if __name__ == "__main__":
    # Configuration
    model_path = "results/best_model.pth"  # Path to your trained model
    buildings_path = "data/shp/buildings.shp"
    community_path = "data/shp/community.shp"

    # Set device ID if you want to use a specific GPU
    device_id = None  # Auto-select

    # Run explanation
    community_results, building_results = run_explanation(
        model_path, buildings_path, community_path,
        device_id=device_id, sample_ratio=0.1
    )