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

## Ensemble Methods `#important` `#exam-tip`

**Definition:** Combine multiple models to create a stronger predictor than any individual model.

**Key Principle:** "Wisdom of the crowd" - diverse models make different errors, averaging reduces overall error.

### Bagging (Bootstrap Aggregating) `#core`

**Mechanism:** Train multiple models **in parallel** on different random subsets of data, then average predictions.

**How it works:**
1. Create multiple bootstrap samples (random sampling with replacement) from training data
2. Train separate model on each sample (usually same algorithm, different data)
3. **Aggregate predictions:**
   - Classification: Majority vote
   - Regression: Average predictions

**Key characteristics:**
- **Parallel training** - Models trained independently
- **Reduces variance** - Averaging reduces overfitting
- **Works best with:** High-variance models (deep decision trees)
- **Doesn't reduce bias** - If base model is biased, averaging won't fix it

**Example Algorithm: Random Forest**
- Bagging with decision trees
- Additional randomization: Each tree sees random subset of features
- Result: Diverse trees that average to robust predictions

**Advantages:**
- ✅ Reduces overfitting (variance reduction)
- ✅ Parallel training (fast on multiple cores)
- ✅ Handles outliers well (averaging reduces impact)
- ✅ Out-of-bag error estimation (free validation set)

**Disadvantages:**
- ❌ Doesn't reduce bias (can't fix underfitting)
- ❌ Less interpretable (ensemble of many models)
- ❌ More memory (stores multiple models)

**When to use:** `#exam-tip`
- Model is overfitting (high variance)
- Have high-variance base learners (deep trees)
- Want stability and robustness
- Can train in parallel (have computational resources)

### Boosting `#core` `#important`

**Mechanism:** Train models **sequentially**, each focusing on errors from previous models.

**How it works:**
1. Train first model on original data
2. Identify misclassified/poorly predicted samples
3. Train second model, **giving more weight to errors from first model**
4. Repeat: Each new model focuses on fixing previous mistakes
5. **Combine predictions:** Weighted sum (better models get more weight)

**Key characteristics:**
- **Sequential training** - Models trained one after another
- **Reduces bias AND variance** - Fixes underfitting and overfitting
- **Works best with:** Weak learners (shallow decision trees)
- **Adaptive** - Each model learns from previous errors

**Common Boosting Algorithms:**

#### AdaBoost (Adaptive Boosting)
- **Reweights samples:** Increase weight of misclassified samples
- **Model weights:** Better models get higher weight in ensemble
- **Stopping:** Fixed number of models or perfect accuracy
- **Sensitive to outliers** (gives them high weight)

#### Gradient Boosting `#exam-tip`
- **Fits residuals:** Each model predicts errors (residuals) from previous model
- **Gradient descent:** Optimizes loss function by adding models
- **More flexible:** Can optimize any differentiable loss function
- **Less sensitive to outliers** than AdaBoost

#### XGBoost (Extreme Gradient Boosting) `#important`
- **AWS built-in algorithm** - Most important for exam!
- Enhanced gradient boosting with:
  - Regularization (L1/L2) to prevent overfitting
  - Parallel tree construction (faster than standard gradient boosting)
  - Handling missing values automatically
  - Tree pruning (removes branches that don't improve performance)
- **Default choice** for tabular data on AWS SageMaker
- Wins most Kaggle competitions

**Advantages:**
- ✅ High accuracy (often best performer)
- ✅ Reduces both bias and variance
- ✅ Handles complex relationships
- ✅ Feature importance built-in

**Disadvantages:**
- ❌ Sequential training (slower than bagging)
- ❌ Prone to overfitting if not tuned (use early stopping)
- ❌ Sensitive to hyperparameters (needs tuning)
- ❌ Can overfit noisy data (AdaBoost especially)

**When to use:** `#exam-tip`
- Need highest accuracy
- Have weak learners (shallow trees work well)
- Want to reduce both bias and variance
- Tabular/structured data (XGBoost excels here)
- Have time for hyperparameter tuning

### Bagging vs Boosting Comparison `#exam-tip`

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel (independent) | Sequential (dependent) |
| **Goal** | Reduce variance | Reduce bias AND variance |
| **Weighting** | Equal weight for all samples | Higher weight for misclassified |
| **Base learners** | High-variance (deep trees) | Low-variance (shallow trees/stumps) |
| **Speed** | Fast (parallel) | Slower (sequential) |
| **Overfitting risk** | Low (averaging reduces) | Higher (can overfit errors) |
| **Example algorithm** | Random Forest | XGBoost, AdaBoost, Gradient Boosting |
| **AWS algorithm** | Random Forest (not built-in) | **XGBoost** (built-in) `#important` |

### Exam Decision Framework `#exam-tip`

**Choose Bagging (Random Forest) when:**
- Model is **overfitting** (high variance problem)
- Need **parallel training** (fast results)
- Want **stability** and robustness
- Have **noisy data** with outliers

**Choose Boosting (XGBoost) when:**
- Need **highest accuracy** (most important)
- Model is **underfitting** (high bias problem)
- Have **tabular/structured data** (XGBoost best here)
- Can afford **sequential training** time
- **Default choice for AWS SageMaker** on tabular data

**Key Exam Scenarios:**

| Scenario | Solution | Reason |
|----------|----------|--------|
| "Best algorithm for tabular data classification?" | **XGBoost** | Boosting, AWS built-in, highest accuracy |
| "Model overfitting, reduce variance?" | **Bagging/Random Forest** | Averaging reduces variance |
| "Model underfitting, reduce bias?" | **Boosting/XGBoost** | Sequential learning reduces bias |
| "Need fast parallel training?" | **Bagging** | Models train independently |
| "Kaggle competition on structured data?" | **XGBoost** | Wins most competitions |
| "Noisy data with outliers?" | **Bagging** | Averaging dampens outlier impact |

### Stacking (Bonus Concept)

**Brief mention:** Combines different types of models (e.g., XGBoost + Neural Net + SVM), uses another model to combine their predictions. More complex than bagging/boosting.
- **Not commonly asked on exam**
- **If mentioned:** Use multiple diverse algorithms, meta-learner combines them

## Model Performance Metrics

### Confusion Matrix `#core` `#exam-tip`

**Visual representation of classification performance:**

```
                Predicted
              Pos    Neg
Actual  Pos   TP  |  FN
        Neg   FP  |  TN
```

**Definitions:**
- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Incorrectly predicted positive (Type I error)
- **FN (False Negative):** Incorrectly predicted negative (Type II error)

**Example - Fraud Detection:**
```
                Predicted Fraud    Predicted Not Fraud
Actual Fraud         90 (TP)            10 (FN)
Not Fraud            5 (FP)            895 (TN)
```
- TP: Correctly caught 90 fraudulent transactions
- FN: Missed 10 fraudulent transactions (bad!)
- FP: 5 false alarms (investigated but no fraud)
- TN: Correctly identified 895 legitimate transactions

### Classification Metrics `#exam-tip`

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Measures:** Overall correctness
- **Range:** 0 to 1 (1 = perfect)
- **Use when:** Balanced classes
- **Don't use when:** Imbalanced classes (99% negative → 99% accuracy by predicting all negative)

**Example from confusion matrix above:**
- Accuracy = (90 + 895) / (90 + 895 + 5 + 10) = 985/1000 = 0.985 = 98.5%

#### Precision
```
Precision = TP / (TP + FP)
```
- **Measures:** Of all positive predictions, how many were correct?
- **Question:** When model says "positive", how often is it right?
- **Use when:** False positives are costly (spam detection, medical diagnosis)
- **Trade-off:** High precision often means lower recall

**Example:** Precision = 90 / (90 + 5) = 90/95 = 0.947 = 94.7%
- When model predicts fraud, it's correct 94.7% of the time

#### Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)
```
- **Measures:** Of all actual positives, how many did we find?
- **Question:** When answer is "positive", how often does model find it?
- **Use when:** False negatives are costly (fraud detection, disease screening)
- **Trade-off:** High recall often means lower precision

**Example:** Recall = 90 / (90 + 10) = 90/100 = 0.90 = 90%
- Model catches 90% of all fraud (misses 10%)

#### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Measures:** Harmonic mean of precision and recall
- **Range:** 0 to 1 (1 = perfect)
- **Use when:** Need balance between precision and recall
- **Best for:** Imbalanced datasets

**Example:** F1 = 2 × (0.947 × 0.90) / (0.947 + 0.90) = 0.923
- Balanced measure: 92.3% performance

#### Precision vs Recall Trade-off `#exam-tip`

| Scenario | Optimize For | Reason |
|----------|--------------|--------|
| Spam email detection | **Precision** | Don't want good emails in spam (FP costly) |
| Fraud detection | **Recall** | Must catch all fraud (FN costly) |
| Disease screening | **Recall** | Can't miss sick patients (FN = death) |
| Product recommendations | **Precision** | Don't annoy users with bad suggestions |

**Decision threshold:**
- **Lower threshold** → More positives predicted → Higher recall, lower precision
- **Higher threshold** → Fewer positives predicted → Higher precision, lower recall

### ROC Curve & AUC `#important` `#exam-tip`

#### ROC Curve (Receiver Operating Characteristic)
**Visual tool showing classifier performance across all thresholds**

**Axes:**
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR) = Recall = TP / (TP + FN)

**How to read:**
- **Top-left corner = Perfect** (TPR=1, FPR=0) - All positives found, no false alarms
- **Diagonal line = Random** (50% chance) - Model no better than coin flip
- **Curve above diagonal = Good** - Better than random
- **Curve below diagonal = Bad** - Worse than random (flip predictions!)

#### AUC (Area Under Curve)
```
AUC Score Range: 0 to 1
```
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.9 - 0.99:** Excellent
- **AUC = 0.8 - 0.89:** Good
- **AUC = 0.7 - 0.79:** Fair
- **AUC = 0.5:** Random (useless)
- **AUC < 0.5:** Worse than random

**Key advantage:** Threshold-independent (evaluates all thresholds)

**Use when:**
- Comparing multiple models
- Class imbalance present
- Care about ranking (is fraud case scored higher than legitimate?)

### Precision-Recall Curve `#exam-tip`

**Alternative to ROC curve, especially for imbalanced datasets**

**Axes:**
- **X-axis:** Recall (TP rate)
- **Y-axis:** Precision

**When to use P-R curve vs ROC curve:**

| Situation | Use |
|-----------|-----|
| Balanced classes | ROC curve |
| **Imbalanced classes** (e.g., 1% positive) | **P-R curve** (better shows performance on minority class) |
| Care about false positive rate | ROC curve |
| Care about precision on positives | P-R curve |

**Why P-R curve for imbalance?**
- ROC can be misleadingly optimistic when negatives dominate
- P-R curve focuses on positive class performance
- Example: 1% fraud → P-R curve shows if model actually finds fraud, ROC looks good just by predicting all negatives

**Exam tip:** `#exam-tip`
- **Imbalanced dataset + minority class important** → Precision-Recall curve
- **Balanced dataset or need overall performance** → ROC-AUC

### Regression Metrics `#important`

#### RMSE (Root Mean Squared Error)
```
RMSE = √[(1/n) × Σ(y - ŷ)²]
```
- **Units:** Same as target variable (e.g., dollars if predicting price)
- **Mechanism:** Square errors → take average → square root
- **Interpretation:** Average prediction error magnitude
- **Sensitivity:** Penalizes large errors heavily (squared)
- **Use when:** Large errors are particularly bad

**Example:** Predicting house prices
- Actual: $300K, Predicted: $320K → Error = $20K
- Squared error = 400M → contributes heavily to RMSE
- If RMSE = $25K → "On average, predictions are off by $25K"

#### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|y - ŷ|
```
- **Units:** Same as target variable
- **Mechanism:** Take absolute value of errors → average
- **Interpretation:** Average absolute prediction error
- **Sensitivity:** Treats all errors equally (linear)
- **Use when:** Outliers present, want robust metric

**Example:** Predicting house prices
- Actual: $300K, Predicted: $320K → Error = $20K
- Absolute error = 20K (not squared)
- If MAE = $18K → "On average, predictions are off by $18K"

#### MSE (Mean Squared Error)
```
MSE = (1/n) × Σ(y - ŷ)²
```
- **Units:** Squared units (e.g., dollars²)
- **Not directly interpretable** (squared units are weird)
- **Use for:** Training optimization (smooth gradients)
- **Related:** RMSE = √MSE (makes MSE interpretable)

#### R² (R-Squared, Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ(y - ŷ)²   (residual sum of squares)
SS_tot = Σ(y - ȳ)²   (total sum of squares)
```
- **Range:** 0 to 1 (can be negative for terrible models)
- **Interpretation:** Proportion of variance explained by model
- **R² = 1.0:** Perfect predictions (explains 100% of variance)
- **R² = 0.5:** Model explains 50% of variance
- **R² = 0:** Model no better than predicting mean
- **R² < 0:** Model worse than predicting mean (bad!)

**Example:**
- R² = 0.85 → Model explains 85% of house price variation
- Remaining 15% due to unmeasured factors or noise

#### Regression Metric Comparison `#exam-tip`

| Metric | Units | Interpretation | Outlier Sensitive? | Use Case |
|--------|-------|----------------|-------------------|----------|
| **RMSE** | Same as y | Average error magnitude | **Yes** (squared) | Default choice, penalize large errors |
| **MAE** | Same as y | Average absolute error | **No** (linear) | Outliers present, want robustness |
| **MSE** | y² | Loss function value | **Yes** (squared) | Training optimization only |
| **R²** | Unitless | % variance explained | Moderate | Model comparison, overall fit |

**Exam scenarios:** `#exam-tip`
- **"Outliers in data, which metric?"** → MAE (robust)
- **"Want to penalize large errors heavily?"** → RMSE or MSE
- **"Compare models on different datasets?"** → R² (unitless)
- **"Training a regression model?"** → MSE (smooth optimization)

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
