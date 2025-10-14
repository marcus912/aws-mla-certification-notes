# Model Training & Evaluation

**Tags:** `#core` `#important`

## Overview
Techniques and best practices for training ML models and evaluating their performance.

## Problem Types

### Regression Problem `#core`
- **Definition:** Predict a **continuous numerical value**
- **Output:** Real number (can be any value in a range)
- **Examples:**
  - Predict house prices ($150,000, $275,500, $1.2M)
  - Forecast temperature (72.5°F, 18.3°C)
  - Estimate sales revenue ($45,678.90)
  - Predict stock prices, age, weight, time
- **Algorithms:** Linear Regression, XGBoost (regression), Neural Networks
- **Metrics:** RMSE, MAE, R², MSE

### Classification Problem `#core`
- **Definition:** Predict a **discrete category/class label**
- **Output:** Categorical value (limited set of options)
- **Types:**
  - **Binary classification** - 2 classes (Yes/No, Fraud/Not Fraud, Cat/Dog)
  - **Multi-class classification** - 3+ classes (Red/Green/Blue, Species A/B/C/D)
  - **Multi-label classification** - Multiple labels per sample (Tags: Sport, Outdoor, Fun)
- **Examples:**
  - Email spam detection (Spam/Not Spam)
  - Image classification (Cat, Dog, Bird, Fish)
  - Disease diagnosis (Positive/Negative)
  - Sentiment analysis (Positive, Neutral, Negative)
- **Algorithms:** Logistic Regression, XGBoost (classification), SVM, Neural Networks
- **Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC

### Key Difference
| Aspect | Regression | Classification |
|--------|-----------|----------------|
| Output | Continuous numbers | Discrete categories |
| Example | Predict $45,678.90 | Predict "High Price" or "Low Price" |
| Evaluation | RMSE, MAE | Accuracy, F1 Score |

## Training Concepts

### Bias in Machine Learning `#important`

**Definition:** Error from overly simplistic assumptions in the learning algorithm.

- **High bias** → Model is too simple → **Underfitting**
- Fails to capture patterns in data
- Poor performance on both training and test data
- **Example:** Using linear model for non-linear relationship
- **Solution:** More complex model, more features, reduce regularization

### Variance in Machine Learning
- **High variance** → Model is too sensitive to training data → **Overfitting**
- Captures noise as patterns
- Good on training data, poor on test data
- **Solution:** More training data, regularization, simpler model, cross-validation

### Bias-Variance Tradeoff
- **Goal:** Minimize total error = Bias² + Variance + Irreducible Error
- **Sweet spot:** Balance between underfitting and overfitting

## Regularization `#important`

**Purpose:** Prevent overfitting by adding penalty for model complexity

### L1 Regularization (Lasso) `#exam-tip`
- **How it works:** Adds penalty = λ × |weights|
- **Effect:** Drives some weights to exactly zero → **Feature selection**
- **Result:** Sparse models (few non-zero features)
- **When to use:**
  - Many features, want automatic feature selection
  - Interpretability important (fewer features)
  - Suspect many features are irrelevant
- **AWS algorithms:** Linear Learner (`l1`), XGBoost (`alpha`)

### L2 Regularization (Ridge) `#exam-tip`
- **How it works:** Adds penalty = λ × weights²
- **Effect:** Shrinks weights toward zero but doesn't eliminate them
- **Result:** All features kept but with reduced impact
- **When to use:**
  - Features are correlated
  - Want to keep all features but reduce overfitting
  - More stable than L1
- **AWS algorithms:** Linear Learner (`wd` = weight decay), XGBoost (`lambda`)

### Elastic Net
- **Combination:** L1 + L2 regularization
- **When to use:** Want feature selection + stability
- **Linear Learner:** Set both `l1` and `wd` (weight decay)

### Regularization Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|-----------|-----------|-------------|
| Feature selection | ✅ Yes (zeros out) | ❌ No (shrinks) | ✅ Yes |
| Handles correlated features | ❌ Picks one randomly | ✅ Yes, shrinks both | ✅ Yes |
| Sparsity | ✅ Sparse model | ❌ Dense model | ⚠️ Some sparsity |
| Stability | ⚠️ Less stable | ✅ More stable | ✅ Stable |
| Interpretability | ✅ Easier (fewer features) | ❌ Harder (all features) | ⚠️ Moderate |

### Exam Tips `#exam-tip`
- **L1 for feature selection:** Use when you have many features
- **L2 for stability:** Use when features are correlated
- **High λ (lambda):** More regularization, simpler model (may underfit)
- **Low λ:** Less regularization, complex model (may overfit)
- **Overfitting solution:** Increase regularization (higher λ)
- **Underfitting solution:** Decrease regularization (lower λ)

## Model Monitoring `#exam-tip`

### Data Drift
- **Definition:** Statistical change in input data distribution over time
- **Explanation:** Production data becomes different from training data
- **Example:**
  - Model trained on summer weather data, now getting winter data
  - E-commerce model trained pre-pandemic, behavior changed post-pandemic
  - Credit model trained in 2020, economic conditions changed in 2025
- **Detection:** Compare distributions using statistical tests (KS test, Chi-square)
- **Impact:** Model performance degrades even though model hasn't changed
- **Solution:** Retrain model with recent data, feature recalculation

### Model Drift (Concept Drift)
- Relationship between features and target changes
- **Example:** What constitutes "spam" evolves over time

### Feature Drift
- Individual features change distribution
- Monitor feature statistics over time

### Feature Attribute Drift `#exam-tip`
- **Definition:** Change in the **importance/ranking** of features over time
- **Detection using NDCG (Normalized Discounted Cumulative Gain):**
  - **NDCG** - Metric originally from information retrieval (search ranking quality)
  - Measures how well feature rankings match between training and production
  - **Formula:** Compares actual feature ranking vs ideal ranking
  - **Score:** 0 to 1 (1 = perfect ranking match)
  - **How it works:**
    1. Rank features by importance during training (baseline)
    2. Rank features by importance in production (current)
    3. Calculate NDCG score comparing rankings
    4. Low NDCG score → feature importance has shifted → potential drift
- **Example:**
  - Training: Most important features were [age, income, location]
  - Production: Most important features became [location, age, credit_score]
  - Feature ranking changed → model assumptions may be invalid
- **Tool:** SageMaker Model Monitor can track feature attribute drift

### Prediction Drift
- Model output distribution changes
- May indicate data or model drift

### AWS Detection Tools
- **SageMaker Model Monitor** - Automated drift detection
- **SageMaker Clarify** - Bias and explainability monitoring
- **CloudWatch** - Custom metrics and alarms

## Evaluation Techniques

### Cross-Validation
- **K-Fold CV** - Split data into k folds, train k times
- Reduces overfitting, better performance estimate
- **Stratified K-Fold** - Preserves class distribution

### Train/Validation/Test Split
- **Training (60-80%)** - Fit model parameters
- **Validation (10-20%)** - Tune hyperparameters
- **Test (10-20%)** - Final evaluation
- **Never use test data for any decisions** `#gotcha`

### Holdout Method
- Single train/test split
- Faster but higher variance in estimate

## Hyperparameter Tuning

### Common Hyperparameters Across Algorithms `#exam-tip`

#### Learning Rate
- **What:** Step size for gradient descent updates
- **Range:** Typically 0.001 to 0.3
- **High learning rate:** Fast training, may overshoot optimal solution
- **Low learning rate:** Slow training, may get stuck in local minimum
- **Adaptive:** Some algorithms (Adam, AdaGrad) adjust automatically
- **Algorithms:** Linear Learner, XGBoost, Neural Networks, DeepAR

#### Mini-Batch Size
- **What:** Number of samples processed before updating model weights
- **Small batches (32-128):**
  - More frequent updates, noisier gradient
  - Better generalization, more regularization effect
  - Slower training per epoch
- **Large batches (512-2048):**
  - Fewer updates, smoother gradient
  - Faster training per epoch
  - May overfit, needs more regularization
- **Algorithms:** Linear Learner, Neural Networks, XGBoost

#### Number of Epochs
- **What:** Complete passes through training dataset
- **Too few:** Underfitting (model hasn't learned enough)
- **Too many:** Overfitting (model memorizes training data)
- **Early stopping:** Stop when validation error stops improving

### Tuning Methods

#### Grid Search
- **How:** Try all combinations of hyperparameter values
- **Pros:** Exhaustive, guaranteed to find best in grid
- **Cons:** Exponentially slow (n^d for n values, d parameters)
- **When:** Few hyperparameters, small search space

#### Random Search
- **How:** Randomly sample hyperparameter combinations
- **Pros:** More efficient than grid search, better for high dimensions
- **Cons:** May miss optimal combination
- **When:** Many hyperparameters, large search space

#### Bayesian Optimization `#exam-tip`
- **How:** Smart search using previous results to guide next tries
- **Pros:** Most efficient, fewer iterations needed
- **Cons:** More complex, slight overhead per iteration
- **When:** Expensive training jobs (SageMaker default)
- **SageMaker:** Automatic Model Tuning uses this

### SageMaker Automatic Model Tuning
- **Algorithm:** Bayesian optimization
- **Setup:** Specify metric to optimize, hyperparameter ranges
- **Limits:** Max 750 training jobs per tuning job
- **Strategies:**
  - **Bayesian** (default) - Smart search
  - **Random** - Random sampling
  - **Grid** - Exhaustive search (small spaces only)
- **Best practices:**
  - Start with wide ranges, narrow down
  - Use log scale for learning rate
  - Monitor cost (each job = charged training time)

## Exam Tips `#exam-tip`
- **Data drift:** Input data distribution changes
- **Model drift:** Input-output relationship changes
- **Always monitor production models** for drift
- **NDCG for feature ranking:** Detect when feature importance shifts
- **Regression vs Classification:** Output type determines problem type
- **High bias = underfitting, High variance = overfitting**

## Related Topics
- [Machine Learning Fundamentals](./ml-fundamentals.md)
- [Feature Engineering](./feature-engineering.md)
- [MLOps & Deployment](./mlops-deployment.md)
