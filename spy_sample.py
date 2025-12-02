"""
Spy sampling for reliable negative example selection in urban renewal prediction.

This script implements ensemble-based spy sampling to identify high-quality
negative examples from unlabeled building data.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


def extract_features(df):
    """Extract and standardize building features."""
    # Define feature categories
    building_features = ['perimeter', 'floor', 'height', 'area']
    function_features = [col for col in df.columns if col.startswith('function_')]
    roof_features = [col for col in df.columns if col.startswith('roof_')]
    rs_features = [col for col in df.columns if col.startswith('hsr_')]

    # Combine all available features
    all_features = []
    feature_names = []

    for feature_list, name in [(building_features, 'building'),
                               (function_features, 'function'),
                               (roof_features, 'roof'),
                               (rs_features, 'rs')]:
        available = [f for f in feature_list if f in df.columns]
        if available:
            all_features.extend(available)
            feature_names.extend(available)

    print(f"Extracted {len(all_features)} features:")
    print(f"  Building: {len([f for f in building_features if f in df.columns])}")
    print(f"  Function: {len([f for f in function_features if f in df.columns])}")
    print(f"  Roof: {len([f for f in roof_features if f in df.columns])}")
    print(f"  Remote Sensing: {len([f for f in rs_features if f in df.columns])}")

    if not all_features:
        raise ValueError("No valid features found in the dataset")

    # Standardize features
    feature_data = df[all_features].fillna(df[all_features].median())
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_data)

    return standardized_features, all_features, scaler


def calculate_ensemble_uncertainty(probs_list):
    """Calculate prediction uncertainty from ensemble predictions."""
    # Stack predictions from all classifiers
    ensemble_probs = np.array(probs_list)  # Shape: (n_classifiers, n_samples, n_classes)

    # Calculate mean predictions
    mean_probs = np.mean(ensemble_probs, axis=0)

    # Calculate prediction entropy (uncertainty measure)
    entropy_values = np.array([entropy(probs) for probs in mean_probs])

    # Calculate standard deviation across classifiers
    std_probs = np.std(ensemble_probs, axis=0)
    mean_std = np.mean(std_probs, axis=1)

    # Combined uncertainty score
    uncertainty_score = (entropy_values + mean_std) / 2

    return uncertainty_score, mean_probs


def apply_spy_sampling(buildings_path, spy_ratio=0.2, random_state=42):
    """
    Apply spy sampling to identify reliable negative examples.

    Args:
        buildings_path: Path to buildings shapefile
        spy_ratio: Ratio of positive samples to use as spies
        random_state: Random seed for reproducibility

    Returns:
        GeoDataFrame with reliable negative labels
    """
    print(f"Applying spy sampling with spy ratio: {spy_ratio}")

    # Load data
    gdf = gpd.read_file(buildings_path)
    print(f"Loaded {len(gdf)} buildings")

    # Check target variable
    if 'renewal' not in gdf.columns:
        raise ValueError("Target variable 'renewal' not found in the dataset")

    # Extract features
    features, feature_names, scaler = extract_features(gdf)

    # Identify positive and unlabeled samples, Your initial dataset should exclude negative samples
    positive_mask = gdf['renewal'] == 1
    negative_mask = gdf['renewal'] == 2
    unlabeled_mask = ~positive_mask & ~negative_mask

    positive_count = np.sum(positive_mask)
    unlabeled_count = np.sum(unlabeled_mask)

    print(f"Found {positive_count} positive samples")
    print(f"Found {np.sum(negative_mask)} labeled negative samples")
    print(f"Found {unlabeled_count} unlabeled samples")

    if unlabeled_count == 0:
        print("No unlabeled samples found. Returning original data.")
        return gdf

    # Prepare labels for ensemble training (1 for positive, 0 for unlabeled)
    y_ensemble = np.where(positive_mask, 1, 0)

    # Initialize ensemble classifiers
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        SVC(probability=True, random_state=random_state, kernel='rbf'),
        LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1),
        KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=1000)
    ]

    print("Training ensemble classifiers...")

    # Get predictions from all classifiers
    all_probs = []
    all_preds = []

    for i, clf in enumerate(classifiers):
        try:
            # Get cross-validated predictions
            probs = cross_val_predict(clf, features, y_ensemble, cv=5, method='predict_proba')
            preds = cross_val_predict(clf, features, y_ensemble, cv=5)

            all_probs.append(probs)
            all_preds.append(preds)

            # Calculate individual classifier performance
            acc = accuracy_score(y_ensemble, preds)
            prec = precision_score(y_ensemble, preds, zero_division=0)
            rec = recall_score(y_ensemble, preds, zero_division=0)
            f1 = f1_score(y_ensemble, preds, zero_division=0)

            print(f"  Classifier {i+1}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

        except Exception as e:
            print(f"  Classifier {i+1} failed: {e}")
            continue

    if not all_probs:
        raise RuntimeError("All ensemble classifiers failed to train")

    # Calculate ensemble uncertainty
    uncertainty_scores, mean_probs = calculate_ensemble_uncertainty(all_probs)

    # Select reliable negatives based on uncertainty
    unlabeled_indices = np.where(unlabeled_mask)[0]
    unlabeled_uncertainty = uncertainty_scores[unlabeled_mask]

    # Determine threshold for reliable negatives
    # Select samples with lowest uncertainty (most confident predictions)
    n_reliable = max(int(unlabeled_count * 0.3), 10)  # At least 10 samples
    reliable_threshold = np.percentile(unlabeled_uncertainty, 30)  # Top 30% most confident

    reliable_mask = unlabeled_uncertainty <= reliable_threshold
    reliable_unlabeled_indices = unlabeled_indices[reliable_mask]

    print(f"Selected {len(reliable_unlabeled_indices)} reliable negative examples")

    # Create new labels
    new_labels = gdf['renewal'].copy()
    new_labels.iloc[reliable_unlabeled_indices] = 2  # Mark as reliable negative

    # Add metadata
    gdf['original_label'] = gdf['renewal']
    gdf['reliable_negative'] = new_labels
    gdf['is_reliable_negative'] = gdf.index.isin(reliable_unlabeled_indices)
    gdf['uncertainty_score'] = uncertainty_scores
    gdf['ensemble_prediction_prob'] = mean_probs[:, 1]

    # Print summary
    label_counts = gdf['reliable_negative'].value_counts()
    print("\nLabel distribution after spy sampling:")
    print(f"  Positive (1): {label_counts.get(1, 0)}")
    print(f"  Reliable Negative (2): {label_counts.get(2, 0)}")
    print(f"  Original Negative (2): {np.sum(negative_mask)}")
    print(f"  Remaining Unlabeled: {np.sum(gdf['reliable_negative'] == 0)}")

    return gdf


def evaluate_sampling_quality(original_gdf, sampled_gdf):
    """Evaluate the quality of spy sampling results."""
    print("Evaluating sampling quality...")

    # Basic statistics
    original_positive = np.sum(original_gdf['renewal'] == 1)
    original_negative = np.sum(original_gdf['renewal'] == 2)
    reliable_negative = np.sum(sampled_gdf['is_reliable_negative'])

    print(f"Original data: {original_positive} positive, {original_negative} negative")
    print(f"Reliable negatives identified: {reliable_negative}")

    if reliable_negative > 0:
        reliable_gdf = sampled_gdf[sampled_gdf['is_reliable_negative']]
        print(f"Reliable negative uncertainty stats:")
        print(f"  Mean: {reliable_gdf['uncertainty_score'].mean():.4f}")
        print(f"  Std: {reliable_gdf['uncertainty_score'].std():.4f}")
        print(f"  Min: {reliable_gdf['uncertainty_score'].min():.4f}")
        print(f"  Max: {reliable_gdf['uncertainty_score'].max():.4f}")

        # Compare with unlabeled samples
        unlabeled_gdf = sampled_gdf[~sampled_gdf['is_reliable_negative'] & (sampled_gdf['reliable_negative'] == 0)]
        if len(unlabeled_gdf) > 0:
            print(f"Remaining unlabeled uncertainty stats:")
            print(f"  Mean: {unlabeled_gdf['uncertainty_score'].mean():.4f}")
            print(f"  Std: {unlabeled_gdf['uncertainty_score'].std():.4f}")


def main():
    """Main function for spy sampling."""
    # Configuration
    buildings_path = "data/buildings.shp"  # Path to your buildings shapefile
    output_path = "data/buildings_with_negative.shp"
    spy_ratio = 0.2  # Ratio of positive samples to use as spies

    try:
        # Apply spy sampling
        result_gdf = apply_spy_sampling(buildings_path, spy_ratio=spy_ratio)

        # Evaluate quality
        evaluate_sampling_quality(gpd.read_file(buildings_path), result_gdf)

        # Save results
        result_gdf.to_file(output_path)
        print(f"\nResults saved to: {output_path}")

        # Also save as CSV for easy inspection
        csv_path = output_path.replace('.shp', '.csv')
        result_gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        print(f"CSV summary saved to: {csv_path}")

    except Exception as e:
        print(f"Error during spy sampling: {e}")
        raise


if __name__ == "__main__":
    main()