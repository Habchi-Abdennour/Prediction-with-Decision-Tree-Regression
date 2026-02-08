# Housing Price Prediction with Decision Tree Regression

## Overview
This project predicts median house values using a Decision Tree Regressor with preprocessing pipelines for handling numerical and categorical features.

## Dataset
- **Source**: `kaggle.com/datasets/camnugent/california-housing-prices`
- **Target**: `median_house_value`
- **Features**: Numerical columns and `ocean_proximity` (categorical)

## Preprocessing Pipeline
- **Numerical Features**: Median imputation for missing values
- **Categorical Features**: One-hot encoding (drop first category)
- **Train/Test Split**: 80/20 split with `random_state=47`


### Initial Model
- **Algorithm**: Decision Tree Regressor
- **Criterion**: Squared error
- **Max Depth**: 6

### Hyperparameter Tuning
- **Method**: RandomizedSearchCV (40 iterations, 5-fold CV)
- **Parameters Tested**:
    - `max_depth`: [None, 3, 5, 8, 12, 20]
    - `min_samples_leaf`: [1, 5, 10, 20, 50]
    - `min_samples_split`: [2, 10, 20, 50]
    - `max_features`: [None, 'sqrt', 'log2', 0.5]

## Evaluation Metrics
- R² Score (training and testing)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Key Libraries
`pandas`, `numpy`, `scikit-learn`, 
