"""
Spy sampling for reliable negative example selection in urban renewal prediction.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df):
    """
    Data preprocessing function
    """
    print("Starting data preprocessing...")

    # Separate different types of features
    building_features = ['perimeter', 'floor', 'height', 'area']
    function_cols = [col for col in df.columns if 'function_' in col]
    roof_cols = [col for col in df.columns if 'roof_' in col]
    rs_features = [col for col in df.columns if col.startswith('hsr_')]

    # Print feature information
    print("Feature Statistics:")
    print(f"Building features: {len(building_features)}")
    print(f"Function features: {len(function_cols)}")
    print(f"Roof features: {len(roof_cols)}")
    print(f"Remote sensing features: {len(rs_features)}")

    features = df[building_features+function_cols+roof_cols+rs_features].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print(f"Processed features shape: {features.shape}")
    return features


def prepare_spy_samples(df, spy_percentage=0.2):
    """
    Prepare spy samples
    """
    print("Preparing spy samples...")

    # Get positive sample indices
    positive_idx = df[df['renewal'] == 1].index
    n_positive = len(positive_idx)

    # Randomly select some positive samples as spy samples
    n_spy = int(n_positive * spy_percentage)
    spy_idx = np.random.choice(positive_idx, size=n_spy, replace=False)

    # Get unknown sample indices
    unknown_idx = df[df['renewal'] == 0].index

    print(f"Number of spy samples: {len(spy_idx)}")
    print(f"Number of unknown samples: {len(unknown_idx)}")

    return spy_idx, unknown_idx

def select_reliable_negative_samples(X, spy_idx, unknown_idx, n_samples=5000):
    """Density and uncertainty-based negative sample selectionm, The sample size should be determined based on your requirements. The default is 5000."""
    print("Starting negative sample selection using density-based approach...")

    classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGB': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    # Create labels for spy samples
    y_spy = np.zeros(len(X))
    y_spy[spy_idx] = 1

    # Get distribution characteristics of positive samples
    positive_kde = KernelDensity(bandwidth=0.5)
    positive_kde.fit(X[spy_idx])

    # Store classifier prediction results for each unknown sample
    sample_scores = {}

    # Train and predict with each classifier
    for clf_name, clf in classifiers.items():
        print(f"\nProcessing classifier: {clf_name}")

        # Train classifier
        current_idx = np.concatenate([spy_idx, unknown_idx])
        X_current = X[current_idx]
        y_current = y_spy[current_idx]
        clf.fit(X_current, y_current)

        # Predict unknown samples
        probs = clf.predict_proba(X[unknown_idx])

        # Store prediction results
        for idx, prob in zip(unknown_idx, probs):
            if idx not in sample_scores:
                sample_scores[idx] = []
            sample_scores[idx].append(prob[0])

    # Calculate comprehensive score for each sample
    final_scores = {}
    for idx, scores in sample_scores.items():
        # Classifier prediction uncertainty
        score_std = np.std(scores)

        # Difference from positive sample distribution
        density_score = np.exp(positive_kde.score_samples(X[idx].reshape(1, -1))[0])

        # Calculate comprehensive score: samples with high uncertainty and some difference from positive distribution are more likely to be selected
        final_scores[idx] = score_std * (1 - density_score)

    # select top N samples by score 
    sorted_samples = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    selected_samples = [idx for idx, _ in sorted_samples[:n_samples]]

    print(f"\nSelected {len(selected_samples)} negative samples with highest scores")
    print(f"Score range: {sorted_samples[0][1]:.4f} - {sorted_samples[min(n_samples, len(sorted_samples)-1)][1]:.4f}")

    return selected_samples


def apply_spy_sampling(buildings_path, output_path=None, spy_ratio=0.2, n_negative_samples=5000):
    """
    Apply spy sampling to identify reliable negative examples.

    Args:
        buildings_path: Path to buildings shapefile
        output_path: Path to save output shapefile (optional)
        spy_ratio: Ratio of positive samples to use as spies
        n_negative_samples: Number of negative samples to select

    Returns:
        GeoDataFrame with reliable negative labels and list of selected indices
    """
    print(f"Applying spy sampling with spy ratio: {spy_ratio}")

    # Load data
    gdf = gpd.read_file(buildings_path)
    print(f"Loaded {len(gdf)} buildings")

    # Check target variable
    if 'renewal' not in gdf.columns:
        raise ValueError("Target variable 'renewal' not found in the dataset")

    # Preprocess data
    gdf.fillna(0, inplace=True)
    gdf.loc[gdf['renewal'] == 2, 'renewal'] = 0

    # Extract features
    features = preprocess_data(gdf)

    # Prepare spy samples
    spy_idx, unknown_idx = prepare_spy_samples(gdf, spy_ratio)

    # Select reliable negative samples
    reliable_negative_idx = select_reliable_negative_samples(
        features, spy_idx, unknown_idx, n_samples=n_negative_samples
    )

    # Create new labels
    new_labels = gdf['renewal'].copy()
    new_labels.iloc[reliable_negative_idx] = 2  # Mark as reliable negative

    # Add metadata
    gdf['original_label'] = gdf['renewal']
    gdf['reliable_negative'] = new_labels

    # Print summary
    label_counts = gdf['reliable_negative'].value_counts()
    print("\nLabel distribution after spy sampling:")
    print(f"  Positive (1): {label_counts.get(1, 0)}")
    print(f"  Reliable Negative (2): {label_counts.get(2, 0)}")
    print(f"  Remaining Unlabeled: {label_counts.get(0, 0)}")

    # Save results if output path provided
    if output_path:
        gdf.to_file(output_path)
        print(f"\nResults saved to: {output_path}")

        # Save indices as CSV
        csv_path = output_path.replace('.shp', '_negative_indices.csv')
        pd.DataFrame({
            'index': reliable_negative_idx,
            'original_renewal': gdf.loc[reliable_negative_idx, 'renewal'].values
        }).to_csv(csv_path, index=False)
        print(f"Negative indices saved to: {csv_path}")

    return gdf, reliable_negative_idx


def main():
    """Main function for spy sampling."""
    # Set random seed
    np.random.seed(42)

    # Configuration
    buildings_path = r'data/shp/buildings.shp'
    output_path = r'data/buildings_with_negatives.shp'

    try:
        # Apply spy sampling
        result_gdf, negative_indices = apply_spy_sampling(
            buildings_path,
            output_path=output_path,
            spy_ratio=0.2,
            n_negative_samples=5000
        )

        print(f"\nSuccessfully identified {len(negative_indices)} reliable negative samples")
        return result_gdf, negative_indices

    except Exception as e:
        print(f"Error during spy sampling: {e}")
        raise


if __name__ == "__main__":
    main()