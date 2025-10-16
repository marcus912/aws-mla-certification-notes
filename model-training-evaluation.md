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

### SageMaker Automatic Model Tuning (HPO) `#important` `#exam-tip`

**Overview:** Automated hyperparameter optimization using Bayesian search to find best model configuration.

**Algorithm:** Bayesian optimization (default and recommended)

**Key Concepts:**
- **Objective metric:** What to optimize (e.g., validation:accuracy, validation:rmse)
- **Hyperparameter ranges:** Search space for each parameter
- **Training jobs:** Each configuration tested is a separate training job
- **Warm start:** Reuse results from previous tuning jobs

#### Configuration

**Basic Setup:**
1. **Define objective metric** - Must be emitted by training algorithm
   - Classification: `validation:accuracy`, `validation:f1`, `validation:auc`
   - Regression: `validation:rmse`, `validation:mae`, `validation:r2`
2. **Specify hyperparameter ranges** - Three types:
   - **IntegerParameterRange** - Discrete integers (e.g., batch_size: 32-512)
   - **ContinuousParameterRange** - Continuous values (e.g., learning_rate: 0.001-0.3)
   - **CategoricalParameterRange** - Fixed choices (e.g., optimizer: ['sgd', 'adam'])
3. **Set resource limits**
   - Max concurrent training jobs (default: 2)
   - Max total training jobs (default: 100, max: 750)
   - Max runtime per training job

**Tuning Strategies:**
- **Bayesian** (default) - Smart search, learns from previous jobs ✅ **Recommended**
- **Random** - Random sampling, good for initial exploration
- **Grid** - Exhaustive search, only for small spaces (< 10 combinations)
- **Hyperband** - Advanced, allocates more resources to promising configs

#### Best Practices `#exam-tip`

**1. Hyperparameter Range Selection**

**Start Wide, Then Narrow:**
```
Iteration 1 (Wide exploration):
  learning_rate: 0.0001 to 1.0

Iteration 2 (Narrow refinement):
  learning_rate: 0.01 to 0.1  (based on Iteration 1 results)
```

**Use Logarithmic Scale for Learning Rate:** `#important`
- **Why:** Learning rates vary by orders of magnitude (0.001 vs 0.1)
- **How:** Use log scale in range definition
- **Example:** `learning_rate: (0.001, 0.3, 'Logarithmic')`
- **Effect:** Samples more values in lower ranges (0.001, 0.003, 0.01, 0.03, 0.1)

**Linear Scale for Other Parameters:**
- Batch size: Linear or Integer scale
- Tree depth: Integer scale
- Number of layers: Integer scale

**2. Hyperparameter Priority** `#exam-tip`

**Tune these first (high impact):**
1. **Learning rate** - Most important, biggest impact
2. **Regularization** (L1/L2, dropout) - Controls overfitting
3. **Batch size** - Affects convergence and generalization
4. **Model architecture** (layers, tree depth) - Core complexity

**Tune these second (medium impact):**
5. Number of epochs (with early stopping)
6. Optimizer choice (SGD, Adam, RMSprop)
7. Activation functions

**Usually don't tune:**
- Momentum (use defaults)
- Numerical stability parameters
- Random seeds

**3. Number of Hyperparameters**

**General rule:** `#exam-tip`
- **1-3 hyperparameters:** Grid search acceptable
- **3-6 hyperparameters:** Bayesian optimization (recommended)
- **6+ hyperparameters:** Random search first, then Bayesian on promising region

**Recommended tuning combinations:**

**Minimal (Fast & Cost-effective):**
- Learning rate only

**Standard (Balanced):**
- Learning rate + Batch size + Regularization (L1/L2)

**Comprehensive (High accuracy needed):**
- Learning rate + Batch size + Regularization + Model architecture + Epochs

**4. Training Job Limits**

**Max Concurrent Jobs:** `#exam-tip`
- **Default: 2** - Safe, prevents resource exhaustion
- **Increase to 5-10:** If you have budget and want faster results
- **Keep low (1-2):** For Bayesian optimization (learns from sequential results)
- **Higher values:** Better for Random search (truly parallel)

**Max Total Jobs:**
- **Rule of thumb:** 10x number of hyperparameters being tuned
- **Example:** Tuning 3 hyperparameters → 30 total jobs minimum
- **Budget permitting:** 20x for thoroughness

**Early Stopping:** `#important`
- Enable to stop poor-performing jobs early
- **Saves cost** - Don't waste time on bad configurations
- **How:** Monitor validation metric, stop if not improving

**5. Objective Metric Selection** `#exam-tip`

**Key principle:** Metric must be emitted by training algorithm

| Problem Type | Recommended Metric | Why |
|--------------|-------------------|-----|
| Binary classification (balanced) | `validation:auc` | Threshold-independent, robust |
| Binary classification (imbalanced) | `validation:f1` | Balances precision and recall |
| Multi-class classification | `validation:accuracy` | Simple, interpretable |
| Regression (normal) | `validation:rmse` | Penalizes large errors |
| Regression (outliers) | `validation:mae` | Robust to outliers |
| Custom metric | Define in training script | Must emit during training |

**Gotcha:** `#gotcha`
- Metric must be prefixed with `train:` or `validation:`
- Use `validation:` metrics (not `train:`) - avoids overfitting
- Metric name must exactly match algorithm output

**6. Warm Start** `#exam-tip`

**What it is:** Reuse hyperparameter search results from previous tuning jobs

**When to use:**
- **Iterative refinement:** Start with wide range, narrow down in second job
- **Incremental dataset:** Reuse tuning from smaller dataset as starting point
- **Similar problem:** Transfer knowledge from related task
- **Resume interrupted job:** Continue from where it stopped

**Benefits:**
- ✅ Faster convergence (don't start from scratch)
- ✅ Cost savings (fewer jobs needed)
- ✅ Transfer learning for hyperparameters

**Configuration:**
```python
parent_tuning_jobs = [
    {'HyperParameterTuningJobName': 'previous-job-name'}
]
```

**Limitation:** Can only warm start from jobs with same hyperparameter names

**7. Cost Optimization** `#exam-tip`

**Cost = (# of training jobs) × (instance cost per hour) × (training time)**

**Strategies to reduce cost:**

1. **Use Spot Instances with HPO**
   - Up to 90% savings on training cost
   - Combine checkpointing with spot for robustness
   - HPO automatically retries failed spot jobs

2. **Limit max training jobs**
   - Start with 20-30 jobs, not 750
   - Analyze results, then run more if needed

3. **Use smaller instances for exploration**
   - Run HPO on smaller instance type (e.g., ml.m5.large)
   - Once optimal hyperparameters found, retrain on larger instance

4. **Enable early stopping**
   - Stop unpromising jobs early
   - Can save 30-50% of tuning cost

5. **Start with Random search**
   - Cheaper than Bayesian for initial exploration
   - Switch to Bayesian for refinement

**8. Common Pitfalls** `#gotcha`

**Overfitting to validation set:**
- **Problem:** Too many tuning iterations overfit validation data
- **Solution:** Hold out separate test set, never tune on it

**Tuning too many hyperparameters:**
- **Problem:** Search space explodes, need 100s of jobs
- **Solution:** Fix less important parameters, tune only critical ones

**Using training metrics instead of validation:**
- **Problem:** Optimize for training performance → overfitting
- **Solution:** Always use `validation:metric`, never `train:metric`

**Insufficient training jobs:**
- **Problem:** 10 jobs for 5 hyperparameters = underfitting search space
- **Solution:** Use 10x rule (50 jobs minimum)

**Wrong scale for learning rate:**
- **Problem:** Linear scale misses good values in 0.001-0.01 range
- **Solution:** Always use logarithmic scale for learning rate

**9. Monitoring and Analysis** `#exam-tip`

**During tuning:**
- Monitor in SageMaker Console → Training Jobs
- Check objective metric trend (improving?)
- Watch for early stopping triggers
- Monitor cost accumulation

**After tuning:**
- **Best training job:** Automatically identified by SageMaker
- **Hyperparameter importance plot:** See which parameters matter most
- **Objective metric vs hyperparameter plots:** Visualize relationships
- **Export results:** Download all job results for analysis

**SageMaker provides:**
- Automatic ranking of all jobs
- Best hyperparameter configuration
- Visualization of search progress

**10. Exam-Focused Best Practices Summary** `#exam-tip`

**If exam asks "Best practice for HPO":**
- ✅ Use Bayesian optimization (default, most efficient)
- ✅ Use logarithmic scale for learning rate
- ✅ Start with wide ranges, narrow down iteratively
- ✅ Tune learning rate first (highest impact)
- ✅ Use validation metrics (not training metrics)
- ✅ Enable early stopping (cost savings)
- ✅ Use warm start for iterative refinement
- ✅ Combine with Spot instances for cost optimization
- ✅ Limit concurrent jobs to 2-5 for Bayesian (learns sequentially)
- ✅ Budget 10x jobs per hyperparameter tuned

**If exam asks "Cost optimization for HPO":**
- ✅ Use Managed Spot Training (90% savings)
- ✅ Enable early stopping
- ✅ Limit max total training jobs
- ✅ Start with smaller instances
- ✅ Use Random search for initial exploration (cheaper)

#### Limits and Quotas `#exam-tip`

- **Max training jobs per tuning job:** 750
- **Max parallel tuning jobs:** 100 (account limit)
- **Max concurrent training jobs:** 100 (default: 2)
- **Max tuning job runtime:** 5 days (default: no limit)
- **Warm start:** Max 5 parent tuning jobs

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
