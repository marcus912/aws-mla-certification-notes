# Machine Learning Fundamentals

**Tags:** `#core` `#important`

## Overview
Core ML concepts and terminology for the AWS MLA exam.

## Learning Types

### Supervised Learning
- **Classification** - Predict discrete labels (binary, multi-class)
- **Regression** - Predict continuous values
- Requires labeled training data

### Unsupervised Learning
- **Clustering** - Group similar data points (K-means, DBSCAN)
- **Dimensionality Reduction** - PCA, t-SNE
- **Anomaly Detection** - Identify outliers
- No labeled data required

### Semi-Supervised Learning
- Combines small labeled + large unlabeled datasets
- Cost-effective when labeling is expensive

### Reinforcement Learning
- Agent learns through rewards/penalties
- Use cases: Gaming, robotics, optimization

## Model Performance Metrics

### Classification Metrics
- **Accuracy** - Correct predictions / Total predictions
- **Precision** - TP / (TP + FP) - Of predicted positives, how many correct?
- **Recall (Sensitivity)** - TP / (TP + FN) - Of actual positives, how many found?
- **F1 Score** - 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC** - Area under ROC curve (0.5 = random, 1.0 = perfect)

### Regression Metrics
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of determination (explained variance)

## Common Challenges

### Overfitting `#gotcha`
- Model too complex, memorizes training data
- **Solutions:** Regularization (L1/L2), dropout, more data, cross-validation

### Underfitting
- Model too simple, poor performance on training data
- **Solutions:** More features, more complex model, reduce regularization

### Class Imbalance `#exam-tip`
- Unequal distribution of classes
- **Solutions:** SMOTE, class weights, under/over sampling

## Bias-Variance Tradeoff
- **High Bias** - Underfitting (too simple)
- **High Variance** - Overfitting (too complex)
- **Goal:** Balance both

## Related Topics
- [Model Training & Evaluation](./model-training-evaluation.md)
- [Feature Engineering](./feature-engineering.md)
