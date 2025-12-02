"""
URHGN: Urban Renewal Hierarchical Graph Network

Core model implementation for urban renewal potential prediction using
dual-layer graph convolutional networks with multi-head attention.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

# Configure CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class MultiHeadFeatureAttention(nn.Module):
    """
    Multi-head attention mechanism for fusing building and community features.
    """

    def __init__(self, building_dim, comm_dim, num_heads=8):
        super(MultiHeadFeatureAttention, self).__init__()
        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(building_dim + comm_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x1, x2):
        outputs_x1 = []
        outputs_x2 = []

        for head in self.attention_heads:
            combined = torch.cat([x1, x2], dim=1)
            attention_weights = head(combined)

            weighted_x1 = x1 * attention_weights[:, 0].unsqueeze(1)
            weighted_x2 = x2 * attention_weights[:, 1].unsqueeze(1)

            outputs_x1.append(weighted_x1)
            outputs_x2.append(weighted_x2)

        final_x1 = torch.mean(torch.stack(outputs_x1), dim=0)
        final_x2 = torch.mean(torch.stack(outputs_x2), dim=0)

        return final_x1, final_x2


class URHGNModel(nn.Module):
    """
    Urban Renewal Hierarchical Graph Network (URHGN) model.

    Dual-layer graph neural network that processes both building and community
    features with multi-head attention fusion.
    """

    def __init__(self, num_building_features, num_community_features, device=None):
        super(URHGNModel, self).__init__()

        # Device management
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Community layer GAT
        self.comm_conv1 = GATv2Conv(num_community_features, 64, heads=8, dropout=0.3)
        self.comm_conv2 = GATv2Conv(64*8, 32, heads=4, dropout=0.3)

        # Multi-head attention layer
        self.feature_attention = MultiHeadFeatureAttention(
            building_dim=num_building_features,
            comm_dim=32*4,
        )

        # Building layer GAT
        self.building_conv1 = GATv2Conv(num_building_features + 32*4, 64, heads=8, dropout=0.3)
        self.building_conv2 = GATv2Conv(64*8, 32, heads=4, dropout=0.3)
        self.building_conv3 = GATv2Conv(32*4, 2, heads=1, dropout=0.3)

        self.dropout = nn.Dropout(0.3)

    def forward(self, building_features, building_edge_indices, community_features,
                community_edge_index, building_to_comm_mapping):
        """
        Forward pass of URHGN model.

        Args:
            building_features: Building feature tensor [num_buildings, num_features]
            building_edge_indices: Dictionary of building edge indices by community
            community_features: Community feature tensor [num_communities, num_features]
            community_edge_index: Community edge index tensor
            building_to_comm_mapping: Mapping from buildings to communities

        Returns:
            Log probabilities [num_buildings, 2]
        """
        # Process community layer
        comm_x = self.comm_conv1(community_features, community_edge_index)
        comm_x = F.relu(comm_x)
        comm_x = self.dropout(comm_x)

        comm_x = self.comm_conv2(comm_x, community_edge_index)
        comm_x = F.relu(comm_x)
        comm_x = self.dropout(comm_x)

        # Map community features to buildings
        building_comm_features = comm_x[building_to_comm_mapping]

        # Apply attention mechanism
        weighted_building, weighted_comm = self.feature_attention(
            building_features, building_comm_features
        )

        # Feature fusion
        fused_building_features = torch.cat([weighted_building, weighted_comm], dim=1)

        # Process building layer in batches
        batch_data_list = []
        global_indices = []

        for comm_id, data in building_edge_indices.items():
            if len(data['local_to_global']) > 0:
                local_building_x = fused_building_features[data['local_to_global']]

                graph_data = Data(
                    x=local_building_x,
                    edge_index=data['edge_index']
                )

                batch_data_list.append(graph_data)
                global_indices.extend(data['local_to_global'].tolist())

        if batch_data_list:
            batch = Batch.from_data_list(batch_data_list).to(self.device)

            x = self.building_conv1(batch.x, batch.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.building_conv2(x, batch.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.building_conv3(x, batch.edge_index)

            combined_output = torch.zeros((building_features.size(0), 2), device=self.device)
            combined_output[global_indices] = x

            return F.log_softmax(combined_output, dim=1)
        else:
            return F.log_softmax(torch.zeros((building_features.size(0), 2), device=self.device), dim=1)

    def get_device(self):
        """Get the device the model is running on."""
        return self.device