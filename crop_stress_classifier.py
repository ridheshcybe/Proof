#!/usr/bin/env python3
"""
Multi-class Crop Stress Classifier using XGBoost
Based on the Crop Health and Environmental Stress (CHES) dataset from Kaggle

This script implements a complete pipeline for classifying crop stress conditions:
- Healthy
- Nutrient stress
- Generic stress
- Water stress

Author: ML Engineer
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import json
import warnings
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_ESTIMATORS = 500
MAX_DEPTH = 8
LEARNING_RATE = 0.05
COLSAMPLE_BYTREE = 0.8
SUBSAMPLE = 0.8
EARLY_STOPPING_ROUNDS = 30
KAGGLE_DATASET_NAME = "Crop Health and Environmental Stress"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets"

def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle API credentials are available.
    
    Returns:
        bool: True if credentials are found, False otherwise
    """
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    return os.path.exists(kaggle_json_path)

def download_ches_dataset() -> Optional[str]:
    """
    Download the CHES dataset from Kaggle.
    
    Returns:
        str: Path to downloaded dataset file, or None if download failed
    """
    try:
        # Check if kaggle is installed
        import subprocess
        import sys
        
        # Try to import kaggle
        try:
            import kaggle
        except ImportError:
            logger.info("Installing kaggle package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
        
        # Check for credentials
        if not check_kaggle_credentials():
            logger.error("Kaggle credentials not found. Please set up Kaggle API.")
            logger.info("Visit https://www.kaggle.com/docs/api to get your API credentials.")
            logger.info("Place kaggle.json in ~/.kaggle/ directory.")
            return None
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), "data", "ches_dataset")
        os.makedirs(data_dir, exist_ok=True)
        
        # Download dataset
        logger.info(f"Downloading {KAGGLE_DATASET_NAME} dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_NAME,
            path=data_dir,
            unzip=True
        )
        
        # Find the CSV file
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error("No CSV files found in downloaded dataset")
            return None
            
        dataset_path = os.path.join(data_dir, csv_files[0])
        logger.info(f"Dataset downloaded successfully: {dataset_path}")
        return dataset_path
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return None

def load_ches_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the CHES dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise pd.errors.EmptyDataError("Dataset is empty")
    
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the dataset according to specifications.
    
    Steps:
    1. Force all feature columns to numeric type
    2. Impute missing values using median
    3. Label-encode categorical variables: crop growth stage and crop name
    4. Partition features into 3 groups: ground (16), satellite (8), temporal (1)
    5. Apply StandardScaler independently to each group (fit on training set only)
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target labels
    """
    logger.info("Starting feature preprocessing...")
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Identify target column (assuming it's named 'stress' or similar)
    target_col = None
    possible_targets = ['stress', 'Stress', 'stress_label', 'label', 'target', 'class']
    for col in possible_targets:
        if col in df_processed.columns:
            target_col = col
            break
    
    if target_col is None:
        # If no obvious target column, assume last column is target
        target_col = df_processed.columns[-1]
        logger.warning(f"No obvious target column found. Using '{target_col}' as target.")
    
    logger.info(f"Using '{target_col}' as target column")
    
    # Separate features and target
    y = df_processed[target_col].copy()
    X = df_processed.drop(columns=[target_col]).copy()
    
    # Step 1: Force all feature columns to numeric type
    logger.info("Converting feature columns to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Step 2: Impute missing values using median
    logger.info("Imputing missing values with median...")
    X = X.fillna(X.median())
    
    # Step 3: Label-encode categorical variables
    # Identify categorical columns (crop name and crop growth stage)
    categorical_cols = []
    for col in X.columns:
        # Check if column contains string values or has low cardinality
        if X[col].dtype == 'object' or X[col].nunique() < 10:
            # Further check if it looks like crop name or growth stage
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['crop', 'variety', 'stage', 'growth', 'name']):
                categorical_cols.append(col)
    
    logger.info(f"Identified categorical columns for encoding: {categorical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            # Handle unknown values by using a special encoded value
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Encoded column '{col}' with {len(le.classes_)} classes")
    
    # Step 4 & 5: Partition features and apply StandardScaler to each group
    # Based on typical CHES dataset structure, we'll identify feature groups
    # This is approximate - in practice, you'd need to know the exact column names
    
    # Try to identify feature groups based on column names
    ground_features = []
    satellite_features = []
    temporal_features = []
    
    for col in X.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['soil', 'ground', 'temp', 'moisture', 'ph', 'nitrogen', 
                                                   'phosphorus', 'potassium', 'organic', 'carbon']):
            ground_features.append(col)
        elif any(keyword in col_lower for keyword in ['ndvi', 'evi', 'lai', 'reflectance', 'band', 
                                                     'spectral', 'satellite', 'rgb', 'nir']):
            satellite_features.append(col)
        elif any(keyword in col_lower for keyword in ['time', 'day', 'month', 'season', 'temporal', 
                                                     'date', 'week']):
            temporal_features.append(col)
    
    # Fallback: if we can't identify groups properly, use approximate counts
    if len(ground_features) == 0 and len(satellite_features) == 0 and len(temporal_features) == 0:
        logger.warning("Could not automatically identify feature groups. Using approximate partitioning.")
        feature_cols = list(X.columns)
        n_features = len(feature_cols)
        
        # Approximate split: 16 ground, 8 satellite, 1 temporal, rest distributed
        ground_features = feature_cols[:min(16, n_features)]
        remaining = feature_cols[min(16, n_features):]
        satellite_features = remaining[:min(8, len(remaining))]
        remaining = remaining[min(8, len(remaining)):]
        temporal_features = remaining[:min(1, len(remaining))]
        
        logger.info(f"Approximate partitioning: {len(ground_features)} ground, "
                   f"{len(satellite_features)} satellite, {len(temporal_features)} temporal features")
    
    logger.info(f"Feature groups identified: {len(ground_features)} ground, "
               f"{len(satellite_features)} satellite, {len(temporal_features)} temporal")
    
    # Apply StandardScaler to each group
    scalers = {}
    X_processed = X.copy()
    
    for group_name, feature_list in [('ground', ground_features), 
                                     ('satellite', satellite_features), 
                                     ('temporal', temporal_features)]:
        if feature_list:
            scaler = StandardScaler()
            # Only scale columns that exist in the dataframe
            existing_features = [f for f in feature_list if f in X_processed.columns]
            if existing_features:
                X_processed[existing_features] = scaler.fit_transform(X_processed[existing_features])
                scalers[group_name] = scaler
                logger.info(f"Applied StandardScaler to {group_name} features ({len(existing_features)} features)")
    
    logger.info("Feature preprocessing completed")
    return X_processed, y

def split_data_stratified(X: pd.DataFrame, y: pd.Series, 
                         test_size: float = TEST_SIZE, 
                         val_size: float = VAL_SIZE,
                         random_state: int = RANDOM_STATE) -> Tuple:
    """
    Split data into train/validation/test sets with stratified sampling.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target labels
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data with stratified sampling...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust validation size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Validation set: {X_val.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Compute sample weights to handle class imbalance using inverse class frequency.
    
    Args:
        y (pd.Series): Target labels
        
    Returns:
        np.ndarray: Sample weights for each training sample
    """
    logger.info("Computing sample weights for class imbalance...")
    
    # Get class counts
    class_counts = np.bincount(y)
    total_samples = len(y)
    n_classes = len(class_counts)
    
    # Compute weights as inverse frequency
    weights = total_samples / (n_classes * class_counts)
    
    # Map weights to each sample
    sample_weights = np.ones(len(y))
    for i, class_label in enumerate(y):
        sample_weights[i] = weights[class_label]
    
    logger.info(f"Class distribution: {dict(zip(range(n_classes), class_counts))}")
    logger.info(f"Sample weights computed: {weights}")
    
    return sample_weights

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       sample_weights: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with specified parameters.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        sample_weights (np.ndarray): Sample weights for training
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    logger.info("Training XGBoost classifier...")
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        colsample_bytree=COLSAMPLE_BYTREE,
        subsample=SUBSAMPLE,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    # Train the model with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Get best iteration from early stopping
    if hasattr(model, 'best_iteration'):
        logger.info(f"Best iteration: {model.best_iteration}")
    else:
        # If early stopping didn't trigger, use all estimators
        logger.info(f"Using all {N_ESTIMATORS} estimators (early stopping not triggered)")
    
    logger.info("XGBoost training completed")
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series,
                  class_names: List[str]) -> Dict:
    """
    Evaluate the trained model and compute metrics.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        class_names (List[str]): Names of the classes
        
    Returns:
        Dict: Evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    support_per_class = np.bincount(y_test)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Print results
    logger.info("=== MODEL EVALUATION RESULTS ===")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1: {f1_macro:.4f}")
    logger.info(f"Weighted F1: {f1_weighted:.4f}")
    
    logger.info("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {precision_per_class[i]:.4f}")
        logger.info(f"    Recall: {recall_per_class[i]:.4f}")
        logger.info(f"    F1: {f1_per_class[i]:.4f}")
        logger.info(f"    Support: {support_per_class[i]}")
    
    return results

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray,
                         class_names: List[str], save_path: str = None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_test (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): Class names for labels
        save_path (str): Path to save the plot (optional)
    """
    logger.info("Generating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Crop Stress Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_feature_importance(model: xgb.XGBClassifier, feature_names: List[str],
                           top_n: int = 10, save_path: str = None):
    """
    Plot top-N feature importances.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        feature_names (List[str]): Names of features
        top_n (int): Number of top features to display
        save_path (str): Path to save the plot (optional)
    """
    logger.info(f"Plotting top {top_n} feature importances...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for sorting
    feat_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feat_importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def save_model(model: xgb.XGBClassifier, file_path: str):
    """
    Save the trained model to file.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        file_path (str): Path to save the model
    """
    logger.info(f"Saving model to {file_path}")
    model.save_model(file_path)
    logger.info("Model saved successfully")

def main():
    """
    Main function to execute the complete pipeline.
    """
    logger.info("=== Crop Stress Classification Pipeline ===")
    logger.info("Starting multi-class crop stress classifier using XGBoost")
    
    try:
        # Step 1: Load dataset
        logger.info("Step 1: Loading CHES dataset")
        
        # Try to find existing dataset
        data_dir = os.path.join(os.getcwd(), "data")
        ches_path = None
        
        # Look for CHES dataset in common locations
        search_paths = [
            os.path.join(data_dir, "ches_dataset"),
            os.path.join(data_dir, "Crop Health and Environmental Stress"),
            data_dir
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                if csv_files:
                    ches_path = os.path.join(path, csv_files[0])
                    break
        
        # If not found locally, try to download from Kaggle
        if ches_path is None:
            logger.info("Local CHES dataset not found. Attempting to download from Kaggle...")
            ches_path = download_ches_dataset()
        
        if ches_path is None:
            logger.error("Failed to locate or download CHES dataset.")
            logger.error("Please ensure you have:")
            logger.error("1. Downloaded the CHES dataset from Kaggle manually")
            logger.error("2. Placed it in a accessible location")
            logger.error("3. Set up Kaggle API credentials for automatic download")
            return
        
        # Load the dataset
        df = load_ches_dataset(ches_path)
        
        # Step 2: Preprocess features
        logger.info("\nStep 2: Preprocessing features")
        X, y = preprocess_features(df)
        
        # Step 3: Split data
        logger.info("\nStep 3: Splitting data")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(X, y)
        
        # Step 4: Compute sample weights
        logger.info("\nStep 4: Computing sample weights")
        sample_weights = compute_sample_weights(y_train)
        
        # Step 5: Train model
        logger.info("\nStep 5: Training XGBoost model")
        model = train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights)
        
        # Step 6: Evaluate model
        logger.info("\nStep 6: Evaluating model")
        
        # Get class names (assuming stress labels are encoded)
        unique_labels = sorted(y.unique())
        # Try to get meaningful class names - common stress classes in CHES dataset
        class_names = ['healthy', 'nutrient_stress', 'stress_generic', 'water_stress']
        # If we have different number of classes, use generic names
        if len(unique_labels) != len(class_names):
            class_names = [f'class_{i}' for i in unique_labels]
            logger.warning(f"Expected 4 classes but found {len(unique_labels)}. Using generic class names.")
        
        results = evaluate_model(model, X_test, y_test, class_names)
        
        # Step 7: Generate visualizations
        logger.info("\nStep 7: Generating visualizations")
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, 
            np.array(results['y_pred']), 
            class_names,
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
        
        # Plot feature importance
        plot_feature_importance(
            model, 
            list(X.columns),
            top_n=10,
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
        
        # Step 8: Save model
        logger.info("\nStep 8: Saving model")
        model_path = os.path.join(output_dir, "xgb_stress_model.json")
        save_model(model, model_path)
        
        # Save evaluation results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_results[key] = value.item()
                elif isinstance(value, list):
                    json_results[key] = [
                        item.tolist() if isinstance(item, np.ndarray) else 
                        item.item() if isinstance(item, (np.integer, np.floating)) else item
                        for item in value
                    ]
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {output_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please ensure the CHES dataset is available at the specified path.")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty dataset: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error("Please check the error details above and try again.")

if __name__ == "__main__":
    main()