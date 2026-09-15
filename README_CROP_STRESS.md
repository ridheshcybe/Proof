# Crop Stress Classifier using XGBoost

This repository contains a complete implementation of a multi-class crop stress classifier using XGBoost based on the Crop Health and Environmental Stress (CHES) dataset from Kaggle.

## Overview

The script implements a complete machine learning pipeline for classifying crop stress into four categories:
- Healthy
- Nutrient stress
- Generic stress
- Water stress

## Features

- **Real Data Loading**: Loads the actual CHES dataset from Kaggle (no dummy/synthetic data)
- **Complete Preprocessing Pipeline**:
  - Forces all features to numeric type
  - Imputes missing values using median
  - Label-encodes categorical variables (crop name, growth stage)
  - Partitions features into ground (16), satellite (8), and temporal (1) groups
  - Applies StandardScaler independently to each group
  - Stratified 70/15/15 train/validation/test split
- **XGBoost Model**:
  - Multi-class softmax objective
  - 500 trees, max_depth=8, learning_rate=0.05
  - colsample_bytree=0.8, subsample=0.8
  - Early stopping with patience=30 on validation log-loss
  - Handles class imbalance via sample weights (inverse class frequency)
- **Comprehensive Evaluation**:
  - Test accuracy, macro precision, macro recall, macro F1, weighted F1
  - Per-class precision, recall, F1, support
  - Confusion matrix visualization
  - Top-10 feature importance bar chart
- **Model Persistence**: Saves trained model as XGBoost JSON format
- **Error Handling**: Robust error handling for missing files and data issues
- **Modular Design**: Separate functions for loading, preprocessing, training, and evaluation

## Requirements

The following packages are required:
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- xgboost>=2.0.0
- matplotlib>=3.3.0
- seaborn>=0.11.0

These are already included in the existing `requirements.txt` file.

## Usage

### Option 1: Automatic Download (Recommended)

1. Ensure you have Kaggle API credentials set up:
   - Get your API token from [Kaggle Account](https://www.kaggle.com/<username>/account)
   - Place `kaggle.json` in `~/.kaggle/` directory

2. Run the script:
   ```bash
   python crop_stress_classifier.py
   ```

### Option 2: Manual Dataset Placement

1. Download the CHES dataset from Kaggle:
   - Go to [Crop Health and Environmental Stress dataset](https://www.kaggle.com/datasets)
   - Download and extract the CSV file

2. Place the dataset in one of these locations:
   - `data/ches_dataset/` (recommended)
   - `data/Crop Health and Environmental Stress/`
   - Directly in `data/` folder

3. Run the script:
   ```bash
   python crop_stress_classifier.py
   ```

## Output

After running the script, you will find:
- Trained model: `output/xgb_stress_model.json`
- Evaluation results: `output/evaluation_results.json`
- Confusion matrix plot: `output/confusion_matrix.png`
- Feature importance plot: `output/feature_importance.png`

## Expected Output

The script will print:
- Dataset loading information
- Preprocessing steps
- Data split statistics
- Model training progress
- Final evaluation metrics:
  - Test Accuracy
  - Macro Precision, Recall, F1
  - Weighted F1
  - Per-class metrics (Precision, Recall, F1, Support)
- Visualizations will be displayed and saved

## Notes

- The script automatically detects feature groups based on column names
- If automatic detection fails, it uses approximate partitioning (16 ground, 8 satellite, 1 temporal)
- Class names are assumed to be: healthy, nutrient_stress, stress_generic, water_stress
- If the dataset has different class labels, generic names will be used
- All random seeds are fixed for reproducibility
- The script includes comprehensive error handling and logging

## Troubleshooting

### Common Issues

1. **Kaggle Authentication Error**:
   - Ensure `kaggle.json` is in `~/.kaggle/` directory
   - Visit [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions

2. **Dataset Not Found**:
   - Manually download CHES dataset from Kaggle
   - Place it in the `data/` directory as described above

3. **Memory Issues**:
   - The CHES dataset is relatively small, but if you encounter memory issues
   - Consider increasing system memory or using a machine with more RAM

4. **Feature Detection Issues**:
   - Check the console output for detected feature groups
   - The script logs how many features were identified in each group

## License

This implementation is provided for educational and research purposes.

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
- Kaggle CHES Dataset: Crop Health and Environmental Stress