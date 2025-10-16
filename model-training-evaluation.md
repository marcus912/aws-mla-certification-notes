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

**Mechanism:** Model makes strong assumptions about data shape/patterns, ignoring complexity

- **High bias** → Model is too simple → **Underfitting**
- **What happens:** Model has too few parameters or limited flexibility to fit data
- Fails to capture true patterns (treats non-linear as linear, misses interactions)
- Poor performance on both training and test data (model can't learn the pattern)
- **Example:** Using linear model for quadratic relationship (y = x²)
- **Visual:** Model draws straight line through curved data
- **Solution:** More complex model, add polynomial features, reduce regularization, more layers (neural nets)

### Variance in Machine Learning
- **High variance** → Model is too sensitive to training data → **Overfitting**
- Captures noise as patterns
- Good on training data, poor on test data
- **Solution:** More training data, regularization, simpler model, cross-validation

### Bias-Variance Tradeoff
- **Goal:** Minimize total error = Bias² + Variance + Irreducible Error
- **Sweet spot:** Balance between underfitting and overfitting

## Loss Functions `#important`

**Purpose:** Quantify how wrong a model's predictions are. Training optimizes (minimizes) the loss function.

### Classification Loss Functions `#exam-tip`

#### Binary Cross-Entropy (Log Loss)
- **Use for:** Binary classification (2 classes)
- **Formula:** `Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- **Range:** 0 to ∞ (0 = perfect predictions)
- **Mechanism:** Heavily penalizes confident wrong predictions
- **AWS algorithms:** Linear Learner (binary_classifier), Neural Networks
- **When prediction is wrong but confident:** Loss explodes (log approaches infinity)

#### Categorical Cross-Entropy
- **Use for:** Multi-class classification (3+ classes)
- **Formula:** `Loss = -Σ(y_i · log(ŷ_i))` across all classes
- **Requirement:** One-hot encoded labels
- **AWS algorithms:** Linear Learner (multiclass_classifier), Image Classification
- **Output layer:** Softmax activation (probabilities sum to 1)

#### Hinge Loss
- **Use for:** Support Vector Machines (SVM), binary classification
- **Formula:** `Loss = max(0, 1 - y·ŷ)`
- **Range:** 0 to ∞
- **Characteristic:** Creates margin between classes
- **Less common** in AWS built-in algorithms (XGBoost uses log loss)

### Regression Loss Functions `#exam-tip`

#### Mean Squared Error (MSE)
- **Use for:** Regression tasks
- **Formula:** `MSE = (1/n)·Σ(y - ŷ)²`
- **Mechanism:** Squares errors → heavily penalizes large errors
- **Pros:** Smooth, differentiable, emphasizes large errors
- **Cons:** Sensitive to outliers (outliers get squared)
- **AWS algorithms:** Linear Learner (regressor), XGBoost (regression)

#### Mean Absolute Error (MAE)
- **Use for:** Regression with outliers
- **Formula:** `MAE = (1/n)·Σ|y - ŷ|`
- **Mechanism:** Absolute value → treats all errors equally
- **Pros:** Robust to outliers, easy to interpret
- **Cons:** Not differentiable at zero (optimization harder)
- **When to use:** Data has outliers, don't want them to dominate loss

#### Huber Loss
- **Use for:** Regression with some outliers (compromise)
- **Mechanism:** MSE for small errors, MAE for large errors
- **Pros:** Combines best of both (smooth + robust)
- **Tunable:** Delta parameter controls transition point
- **AWS:** Available in some algorithms (less common)

### Loss Function Selection Guide `#exam-tip`

| Problem Type | Loss Function | Reason |
|--------------|---------------|--------|
| Binary classification | Binary Cross-Entropy | Probabilistic outputs, penalizes confidence |
| Multi-class classification | Categorical Cross-Entropy | Handles multiple classes with softmax |
| Regression (normal data) | MSE | Emphasizes large errors, smooth optimization |
| Regression (with outliers) | MAE | Robust to outliers, equal error treatment |
| Regression (some outliers) | Huber Loss | Balanced approach |

**Key Exam Concepts:** `#exam-tip`
- **Cross-entropy for classification** (most common)
- **MSE for regression** (AWS default for Linear Learner, XGBoost)
- **Outliers present?** Use MAE instead of MSE
- **Loss goes to zero** = Perfect predictions (rarely happens)
- **Loss increasing during training?** Learning rate too high or model diverging

## Regularization `#important`

**Purpose:** Prevent overfitting by adding penalty for model complexity

### L1 Regularization (Lasso) `#exam-tip`
- **How it works:** Adds penalty = λ × |weights| (absolute value)
- **Critical mechanism:** Forces weights to **EXACTLY zero** (complete elimination)
- **Why zeros?** Absolute value penalty has sharp corner at zero, gradient "snaps" weights to zero
- **Effect:** Automatic feature selection - removes features by zeroing their weights
- **Result:** Sparse models (many weights = 0, only important features remain)
- **When to use:**
  - Many features, want automatic feature selection
  - Interpretability important (fewer features)
  - Suspect many features are irrelevant
- **AWS algorithms:** Linear Learner (`l1`), XGBoost (`alpha`)

### L2 Regularization (Ridge) `#exam-tip`
- **How it works:** Adds penalty = λ × weights² (squared)
- **Critical mechanism:** Shrinks weights toward zero but **NEVER exactly zero** (approaches asymptotically)
- **Why not zero?** Squared penalty gets gentler near zero, weights shrink but don't eliminate
- **Effect:** All features kept but with reduced magnitude/impact
- **Result:** Dense models (all features present, none eliminated)
- **When to use:**
  - Features are correlated (L2 keeps all, L1 randomly picks one)
  - Want to keep all features but reduce overfitting
  - More stable than L1 (small data changes don't flip feature selection)
- **AWS algorithms:** Linear Learner (`wd` = weight decay), XGBoost (`lambda`)

### Elastic Net
- **Combination:** L1 + L2 regularization
- **When to use:** Want feature selection + stability
- **Linear Learner:** Set both `l1` and `wd` (weight decay)

### Regularization Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|-----------|-----------|-------------|
| **Weights become zero?** | ✅ **Yes - EXACTLY zero** | ❌ **No - approaches zero, never reaches** | ✅ Yes (from L1 component) |
| Feature selection | ✅ Yes (eliminates features) | ❌ No (keeps all features) | ✅ Yes |
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
- **What it means:** Features that were important during training become less important in production (or vice versa)
- **Example:**
  - **Training:** Most important features were [age, income, location]
  - **Production:** Most important features became [location, age, credit_score]
  - **Problem:** "credit_score" wasn't important before, now it is → model assumptions may be invalid

**Detection using NDCG (Normalized Discounted Cumulative Gain):**
- **Key concept:** NDCG measures if feature rankings match between training and production
- **Score range:** 0 to 1 (1 = perfect match, 0 = completely different)
- **Mechanism:**
  1. Rank features by importance during training (baseline ranking)
  2. Rank features by importance in production (current ranking)
  3. Compare rankings: Do the top features match? Are they in similar order?
  4. **Low NDCG score** → rankings don't match → feature importance has shifted → drift detected
- **Origin:** NDCG comes from search engine ranking (measures if search results are in right order)
- **Tool:** SageMaker Model Monitor automatically tracks feature attribute drift using NDCG

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
  - More frequent updates, **noisier gradient** estimates
  - **Regularization mechanism:** Noise prevents model from memorizing specific training patterns
  - **Why it helps:** Each update based on small subset → variations force model to learn general patterns, not specifics
  - Better generalization (natural regularization effect)
  - Slower training per epoch (more updates needed)
- **Large batches (512-2048):**
  - Fewer updates, **smoother gradient** (averaged over many samples)
  - **Less noise** → can overfit more easily (gradient points to training data specifics)
  - Faster training per epoch (fewer weight updates)
  - May overfit, needs explicit regularization (L1/L2)
- **Trade-off:** Small batch = implicit regularization from noise, Large batch = speed but needs explicit regularization
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
