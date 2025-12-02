"""
Traditional Machine Learning Models for Urban Renewal Prediction
===============================================================

This script implements traditional ML models for building-level urban renewal prediction:
- MLP (Multi-Layer Perceptron)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Features include 5-fold cross-validation and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, average_precision_score,
                            confusion_matrix, roc_auc_score)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

# Set global font to Arial
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20
rcParams['font.weight'] = 'bold'


class MLPNet(torch.nn.Module):
    """Multi-Layer Perceptron for binary classification"""

    def __init__(self, num_features):
        super(MLPNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.classifier = torch.nn.Linear(128, 2)

        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.classifier(x)
        return torch.log_softmax(x, dim=1)


def interpolate_roc_curve(fpr, tpr, mean_fpr):
    """Interpolate TPR values at mean FPR points using numpy.interp"""
    return np.interp(mean_fpr, fpr, tpr)


def interpolate_pr_curve(recall, precision, mean_recall):
    """Interpolate precision values at mean recall points using numpy.interp"""
    return np.interp(mean_recall, recall[::-1], precision[::-1])


def create_save_dir(save_path):
    """Create directory if it doesn't exist"""
    os.makedirs(save_path, exist_ok=True)


def plot_roc_curves(all_results, save_path='./results/'):
    """Plot ROC curves with all folds and mean curve"""
    create_save_dir(save_path)

    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
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

        axes[idx].set_xlabel('False Positive Rate', fontsize=28, fontweight='bold', labelpad=10)
        axes[idx].set_ylabel('True Positive Rate', fontsize=28, fontweight='bold', labelpad=10)
        axes[idx].set_title(model_name, fontsize=30, fontweight='bold', pad=15)
        axes[idx].legend(loc='lower right', fontsize=20, framealpha=0.7)
        axes[idx].grid(False)
        for spine in axes[idx].spines.values():
            spine.set_linewidth(2)
        axes[idx].tick_params(labelsize=22)
        axes[idx].set_xlim([-0.01, 1.01])
        axes[idx].set_ylim([-0.01, 1.01])

        for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
            label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(f'{save_path}roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curves saved to {save_path}roc_curves.png")


def plot_pr_curves(all_results, save_path='./results/'):
    """Plot PR curves with all folds and mean curve"""
    create_save_dir(save_path)

    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
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

        axes[idx].set_xlabel('Recall', fontsize=28, fontweight='bold', labelpad=10)
        axes[idx].set_ylabel('Precision', fontsize=28, fontweight='bold', labelpad=10)
        axes[idx].set_title(model_name, fontsize=30, fontweight='bold', pad=15)
        axes[idx].legend(loc='lower left', fontsize=20, framealpha=0.7)
        axes[idx].grid(False)
        for spine in axes[idx].spines.values():
            spine.set_linewidth(2)
        axes[idx].tick_params(labelsize=22)
        axes[idx].set_xlim([-0.01, 1.01])
        axes[idx].set_ylim([-0.01, 1.01])

        for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
            label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(f'{save_path}pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PR curves saved to {save_path}pr_curves.png")


def train_mlp(X, y, train_idx, val_idx):
    """Train MLP model and return predictions"""
    x = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    model = MLPNet(num_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Handle class imbalance with weighted loss
    class_counts = torch.bincount(y_tensor)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0
    best_model_state = None
    patience = 20
    counter = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x[train_idx])
        loss = criterion(out, y_tensor[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x[val_idx])
            pred = out.max(1)[1]
            val_f1 = f1_score(y_tensor[val_idx], pred)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(x[val_idx])
        pred = out.max(1)[1].numpy()
        pred_proba = torch.exp(out[:, 1]).numpy()

    return pred, pred_proba, y_tensor[val_idx].numpy()


def train_sklearn_model(model, X, y, train_idx, val_idx):
    """Train sklearn model and return predictions"""
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[val_idx])
    pred_proba = model.predict_proba(X[val_idx])[:, 1]
    return pred, pred_proba, y[val_idx]


def load_and_preprocess_data(data_path):
    """Load and preprocess the urban renewal dataset"""
    # Load dataset
    df = gpd.read_file(data_path)
    print(f"Dataset loaded successfully with {len(df)} buildings")

    # Extract feature columns
    building_features = ['perimeter', 'floor', 'height', 'area']
    function_cols = [col for col in df.columns if 'function_' in col]
    roof_cols = [col for col in df.columns if 'roof_' in col]
    rs_features = [col for col in df.columns if col.startswith('hsr_')]

    # Combine all features
    features = df[building_features + function_cols + roof_cols + rs_features].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Process labels (convert to binary classification)
    y = df['renewal'].values
    mask = (y == 1) | (y == 2)

    y_train = y[mask].copy()
    y_train = np.where(y_train == 2, 0, y_train)
    print("Label distribution:", Counter(y_train))

    return X[mask], y_train


def main(data_path, save_path='./results/'):
    """Main function to run all traditional ML models with 5-fold cross-validation"""

    # Load and preprocess data
    X, y = load_and_preprocess_data(data_path)

    # Initialize all models
    models = {
        'MLP': None,
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Store results for each model
    all_results = {model_name: [] for model_name in models.keys()}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\nStarting 5-fold cross-validation...")
    print("="*100)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining Fold {fold+1}")

        for model_name, model in models.items():
            if model_name == 'MLP':
                pred, pred_proba, y_true = train_mlp(X, y, train_idx, val_idx)
            else:
                pred, pred_proba, y_true = train_sklearn_model(
                    model, X, y, train_idx, val_idx)

            # Calculate evaluation metrics
            acc = accuracy_score(y_true, pred)
            prec = precision_score(y_true, pred)
            rec = recall_score(y_true, pred)
            f1 = f1_score(y_true, pred)
            roc_auc = roc_auc_score(y_true, pred_proba)
            pr_auc = average_precision_score(y_true, pred_proba)

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
                'y_pred_proba': pred_proba
            })

            print(f"{model_name} - Fold {fold+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, "
                  f"Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    # Print average results for each model
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
    summary_df.to_csv(f'{save_path}traditional_model_comparison_summary.csv', index=False)
    print(f"\nSummary table saved to {save_path}traditional_model_comparison_summary.csv")

    # Generate all plots
    print("\n" + "="*100)
    print("Generating visualizations...")
    print("="*100)

    plot_roc_curves(all_results, save_path)
    plot_pr_curves(all_results, save_path)

    print("\nAll visualizations completed successfully!")


if __name__ == "__main__":
    # Update this path to your actual data file
    data_path = "path/to/your/buildings_data.shp"  # Replace with actual path
    main(data_path)