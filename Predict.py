import os
import time
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from DoubleGCN_model import MultiHeadFeatureAttention, DoubleGCN
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Batch
from datetime import timedelta

# 设置设备
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============ 图构建函数 ============
def construct_community_graph(communities_gdf):
    """构建社区图"""
    print("构建社区图...")
    start_time = time.time()
    
    # 找到相邻社区
    edge_list = []
    for idx, comm in communities_gdf.iterrows():
        neighbors = communities_gdf[communities_gdf.geometry.touches(comm.geometry)].index
        for neighbor in neighbors:
            edge_list.append([idx, neighbor])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
    
    # 准备社区特征
    poi_cols = [f'poi_{i}' for i in range(16)]
    other_cols = ['population', 'canopy_cov', 'price']
    features = communities_gdf[poi_cols + other_cols].values
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    x = torch.tensor(features, dtype=torch.float32).to(device)
    
    elapsed_time = time.time() - start_time
    print(f"社区图构建完成，用时：{timedelta(seconds=elapsed_time)}")
    
    return edge_index, x

def construct_building_graphs(df, default_k=5):
    """构建建筑物图"""
    print("构建建筑物图...")
    start_time = time.time()
    edge_indices = {}
    
    for comm_id, group in df.groupby('comm_id'):
        if len(group) < 2:
            continue
            
        coords = np.column_stack([
            group.geometry.centroid.x.astype(np.float32),
            group.geometry.centroid.y.astype(np.float32)
        ])
        
        k = min(default_k, len(group) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        edge_list = []
        local_to_global = group.index.tolist()
        
        for i in range(len(group)):
            for j in indices[i, 1:]:
                edge_list.append([i, j])
        
        if edge_list:
            edge_indices[comm_id] = {
                'edge_index': torch.tensor(edge_list, dtype=torch.long).t().to(device),
                'local_to_global': torch.tensor(local_to_global, dtype=torch.long).to(device)
            }
    
    elapsed_time = time.time() - start_time
    print(f"建筑物图构建完成，用时：{timedelta(seconds=elapsed_time)}")
    
    return edge_indices

def prepare_building_features(df, feature_columns):
    """准备建筑物特征"""
    print("准备建筑物特征...")
    
    features = df[feature_columns].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return torch.tensor(features, dtype=torch.float32).to(device)

def load_model_and_predict(model_path, buildings_gdf, communities_gdf, output_path):
    """
    加载训练好的模型并进行预测
    
    Args:
        model_path: 模型文件路径
        buildings_gdf: 建筑物数据
        communities_gdf: 社区数据
        output_path: 输出文件路径
    """
    print("=" * 60)
    print("开始加载模型并进行预测")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 加载模型
    print(f"1. 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 打印 Training_model.py 保存的模型信息 (可选，用于确认)
    if 'best_fold' in checkpoint:
        print(f"最优模型来自交叉验证的第 {checkpoint['best_fold']} 折")
    if 'best_metrics' in checkpoint:
        print(f"最优模型在验证集上的性能: {checkpoint['best_metrics']}")
    if 'training_timestamp' in checkpoint:
        print(f"模型训练时间戳: {checkpoint['training_timestamp']}")

    # 2. 初始化模型
    print("2. 初始化模型...")
    model = DoubleGCN(
        num_building_features=checkpoint['num_building_features'],
        num_community_features=checkpoint['num_community_features']
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功!")
    
    # 3. 构建图结构
    print("3. 构建图结构...")
    comm_edge_index, comm_x = construct_community_graph(communities_gdf)
    building_edge_indices = construct_building_graphs(buildings_gdf)
    
    # 4. 准备特征
    print("4. 准备建筑物特征...")
    # 根据 Training_model.py 中的定义，硬编码建筑物特征列表
    building_features = ['perimeter', 'floor', 'height', 'area']
    function_cols = [col for col in buildings_gdf.columns if 'function_' in col]
    roof_cols = [col for col in buildings_gdf.columns if 'roof_' in col]
    rs_features = [col for col in buildings_gdf.columns if col.startswith('hsr_')]

    building_x = prepare_building_features(buildings_gdf, building_features + function_cols + roof_cols + rs_features) # 传递正确的特征列表
    
    # 创建建筑物到社区的映射
    comm_id_to_idx = {id: i for i, id in enumerate(communities_gdf.comm_id.unique())}
    building_to_comm_mapping = torch.tensor(
        buildings_gdf['comm_id'].map(comm_id_to_idx).values, 
        dtype=torch.long
    ).to(device)
    
    print(f"数据准备完成: {len(buildings_gdf)} 栋建筑物, {len(communities_gdf)} 个社区")
    
    # 5. 进行预测
    print("5. 开始预测...")
    predict_start_time = time.time()
    
    with torch.no_grad():
        # 获取模型输出
        output = model(building_x, building_edge_indices, comm_x, comm_edge_index, 
                      building_to_comm_mapping)
        
        # 计算概率 - 这里的probability_renewal就是你需要的更新潜力值
        probabilities = F.softmax(output, dim=1)
        renewal_potential = probabilities[:, 1].cpu().numpy()  # 类别1的概率
        
    predict_time = time.time() - predict_start_time
    print(f"预测完成，用时: {predict_time:.2f}秒")
    
    # 6. 保存结果
    print("6. 保存预测结果...")
    
    # 创建输出数据框
    buildings_with_predictions = buildings_gdf.copy()
    
    # 只添加一个关键列：更新潜力（0-1概率值）
    buildings_with_predictions['potential'] = renewal_potential
    
    # 保存文件
    try:
        buildings_with_predictions.to_file(output_path, driver='ESRI Shapefile')
        print(f"预测结果已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存Shapefile时出错: {e}")
        # 尝试保存为GeoPackage
        output_gpkg = output_path.replace('.shp', '.gpkg')
        buildings_with_predictions.to_file(output_gpkg, driver='GPKG')
        print(f"已保存为GeoPackage格式: {output_gpkg}")
    
    total_time = time.time() - start_time
    print(f"总用时: {timedelta(seconds=total_time)}")
    
    print(f"\n新增列说明:")
    print(f"  - renewal_potential: 城市更新潜力值 (0-1, 越接近1潜力越高)")
    
    return buildings_with_predictions

def main():
    """主函数"""
    print("城市更新潜力预测系统")
    print("=" * 40)
    
    # 设置文件路径
    model_path = r"code/best_urban_renewal_model.pth"
    buildings_path = r"~/code/Data/buildings_with_reliable_negative_0327.shp"
    communities_path = r"~/code/Data/outbjTAZ/community.shp"
    output_path = r"code/buildings_with_renewal_predictions.shp"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件!")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # 加载数据
        print("加载数据...")
        buildings_gdf = gpd.read_file(buildings_path)
        communities_gdf = gpd.read_file(communities_path)
        print(f"建筑物数据: {len(buildings_gdf)} 栋")
        print(f"社区数据: {len(communities_gdf)} 个")
        
        # 进行预测
        predicted_buildings = load_model_and_predict(
            model_path=model_path,
            buildings_gdf=buildings_gdf,
            communities_gdf=communities_gdf,
            output_path=output_path
        )
        
        print("\n预测完成!")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
