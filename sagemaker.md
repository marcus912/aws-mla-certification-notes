# Amazon SageMaker

**Tags:** `#core` `#important` `#hands-on`

## Overview
Fully managed ML service for building, training, and deploying models at scale.

## Core Components

### SageMaker Studio
- Web-based IDE for ML development
- Jupyter notebooks, experiment tracking, debugging

### SageMaker Notebooks
- Managed Jupyter notebook instances
- Pre-configured with ML frameworks

### Built-in Algorithms `#exam-tip`
- **Linear Learner** - Linear regression, classification
- **XGBoost** - Gradient boosted trees
- **K-Means** - Clustering
- **PCA** - Dimensionality reduction
- **Factorization Machines** - Recommendations
- **DeepAR** - Time series forecasting
- **BlazingText** - Text classification, Word2Vec
- **Object Detection** - Images
- **Image Classification** - ResNet CNN
- **Semantic Segmentation** - Pixel-level classification

## Algorithm Hyperparameters `#important`

### Linear Learner Hyperparameters `#exam-tip`

#### Regularization
- **`l1`** - L1 regularization (Lasso)
  - Range: 0 to 1
  - Effect: Feature selection, sparse models
  - Use when: Many features, want feature selection
- **`wd`** (weight decay) - L2 regularization (Ridge)
  - Range: 0 to 1
  - Effect: Shrink all weights, prevent overfitting
  - Use when: Features are correlated
- **Elastic Net:** Set both `l1` and `wd` for combined effect

#### Training Parameters
- **`learning_rate`**
  - Range: 0.0001 to 1.0
  - Default: Auto-tuned
  - Critical for convergence speed
- **`mini_batch_size`**
  - Range: 1 to 10000
  - Default: 1000
  - Affects training speed and generalization

#### Class Imbalance Handling
- **`balance_multiclass_weights`** `#exam-tip`
  - Values: `true` or `false`
  - Purpose: Handle imbalanced classes
  - Effect: Assign higher weight to minority classes
  - **When to use:**
    - Imbalanced classification (fraud, anomaly detection)
    - One class much rarer than others
    - Want to avoid bias toward majority class
  - **Alternative:** Use class weights, SMOTE (preprocessing)

#### Optimization Objectives `#exam-tip`
- **`target_precision`**
  - Range: 0 to 1
  - **Unique to Linear Learner**
  - Use when: Precision is critical (e.g., spam detection - don't want false positives)
  - Model optimizes to achieve this precision target

- **`target_recall`**
  - Range: 0 to 1
  - **Unique to Linear Learner**
  - Use when: Recall is critical (e.g., fraud detection - don't want to miss fraud)
  - Model optimizes to achieve this recall target

- **Cannot set both:** Choose `target_precision` OR `target_recall`, not both

#### Predictor Type
- **`predictor_type`**
  - `binary_classifier` - Binary classification
  - `multiclass_classifier` - Multi-class classification
  - `regressor` - Regression

### XGBoost Hyperparameters `#exam-tip`

#### Tree Parameters
- **`max_depth`**
  - Range: 1 to 10+
  - Default: 6
  - Effect: Deeper trees = more complex model (overfitting risk)
- **`num_round`**
  - Number of boosting rounds (trees to build)
  - More rounds = more complex model
- **`min_child_weight`**
  - Minimum sum of instance weight in a child
  - Higher value = more conservative (prevent overfitting)

#### Regularization
- **`alpha`** - L1 regularization
  - Default: 0
  - Increase for feature selection
- **`lambda`** - L2 regularization
  - Default: 1
  - Increase to prevent overfitting

#### Learning
- **`eta`** (learning rate)
  - Range: 0 to 1
  - Default: 0.3
  - Lower = slower, more robust
- **`subsample`**
  - Fraction of training data to sample per tree
  - Range: 0 to 1
  - Default: 1
  - Lower value prevents overfitting

#### Class Imbalance
- **`scale_pos_weight`**
  - Ratio of negative to positive class
  - Use for imbalanced binary classification
  - Calculate as: (# negative samples) / (# positive samples)

### Hyperparameter Comparison `#exam-tip`

| Hyperparameter | Linear Learner | XGBoost | Purpose |
|----------------|---------------|---------|---------|
| L1 regularization | `l1` | `alpha` | Feature selection |
| L2 regularization | `wd` | `lambda` | Prevent overfitting |
| Learning rate | `learning_rate` | `eta` | Convergence speed |
| Batch size | `mini_batch_size` | N/A | Training speed |
| Class weights | `balance_multiclass_weights` | `scale_pos_weight` | Handle imbalance |
| Target metric | `target_precision`, `target_recall` | N/A | Optimize specific metric |

### Exam Scenarios `#exam-tip`

**Scenario 1:** Fraud detection (99% not fraud, 1% fraud)
- **Problem:** Severe class imbalance, must catch fraud (high recall)
- **Solution:**
  - Linear Learner: Set `balance_multiclass_weights=true` AND `target_recall=0.95`
  - XGBoost: Set `scale_pos_weight=99` (99:1 ratio)
  - Preprocessing: Apply SMOTE to balance training data

**Scenario 2:** Model overfitting (perfect on training, poor on test)
- **Solution:**
  - Linear Learner: Increase `wd` (L2 regularization)
  - XGBoost: Increase `lambda`, decrease `max_depth`, increase `min_child_weight`
  - Both: Add more training data, use cross-validation

**Scenario 3:** Too many features (1000+), want interpretable model
- **Solution:**
  - Linear Learner: Increase `l1` for feature selection
  - XGBoost: Increase `alpha` (L1 regularization)
  - Result: Sparse model with only important features

**Scenario 4:** Email spam detection (precision critical - don't mark good emails as spam)
- **Solution:**
  - Linear Learner: Set `target_precision=0.95`
  - Trade-off: May miss some spam (lower recall) to avoid false positives

### Training

**Training Jobs**
- Specify algorithm, compute resources, data location
- **Input modes:** File, Pipe (streaming)
- **Instance types:** ml.m5, ml.p3 (GPU), ml.p4d

**Managed Spot Training** `#exam-tip`
- Up to 90% cost savings
- Use checkpoints for interruption handling

**Automatic Model Tuning (Hyperparameter Optimization)**
- Bayesian optimization
- Max 750 training jobs per tuning job

### Deployment

**Endpoints**
- Real-time inference
- Auto-scaling supported
- **Multi-model endpoints** - Host multiple models on one endpoint

**Batch Transform**
- Offline predictions on large datasets
- No persistent endpoint needed

**Inference Pipelines**
- Chain preprocessing + prediction (2-15 containers)

**SageMaker Serverless Inference** `#exam-tip`
- Pay per use, auto-scales to zero
- Good for intermittent traffic

### Model Monitoring

**SageMaker Model Monitor**
- Detects data drift, model quality degradation
- Automatic baseline creation
- Continuous monitoring

## SageMaker Autopilot `#exam-tip`
- AutoML - automatically builds, trains, tunes models
- White-box approach (provides notebooks)
- Supports tabular data (CSV)

## SageMaker Ground Truth
- Data labeling service
- Human labelers + active learning
- Built-in labeling workflows

## SageMaker Feature Store `#important`
- Centralized repository for ML features
- Online + offline stores
- Feature versioning, sharing, discovery

## SageMaker Pipelines
- CI/CD for ML workflows
- Define, orchestrate, automate ML workflows
- Integrates with MLOps tools

## Pricing Model `#exam-tip`
- Pay for compute (training, inference)
- Pay for storage (S3, EBS)
- No charge for Studio (pay for underlying compute)

## Exam Tips
- Know which built-in algorithms for which use cases
- Understand endpoint vs batch transform tradeoffs
- Spot training requires checkpointing
- Serverless inference for intermittent/unpredictable traffic

## Gotchas `#gotcha`
- Training data must be in S3
- Built-in algorithms expect specific data formats (RecordIO-protobuf, CSV, JSON)
- Endpoint deployment can take 5-10 minutes

## Related Topics
- [MLOps & Deployment](./mlops-deployment.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
