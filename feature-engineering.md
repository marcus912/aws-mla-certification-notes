# Feature Engineering

**Tags:** `#core` `#important` `#hands-on`

## Overview
Techniques for preparing, transforming, and creating features from raw data for ML models.

## Handling Missing Data

### Mean Replacement (Mean Imputation)
- **"Mean"** = Average value of the feature
- Replace missing values with the **arithmetic mean** of existing values
- **Example:** If ages are [25, 30, ?, 40], replace ? with mean = (25+30+40)/3 = 31.67
- **Pros:** Simple, works for numerical data
- **Cons:** Reduces variance, doesn't capture relationships between features
- **Use when:** Data is Missing Completely At Random (MCAR)

### Median Replacement (Median Imputation)
- Replace missing values with the **median** (middle value when sorted)
- **Example:** If ages are [25, 30, ?, 40, 100], replace ? with median = 30
- **Pros:** Robust to outliers, better than mean for skewed distributions
- **Cons:** Still reduces variance
- **Use when:** Data has outliers or is skewed `#exam-tip`
- **Better than mean when:** Distribution is not normal

### Other Imputation Methods
- **Mode replacement** - Most frequent value (for categorical data)
- **Forward/backward fill** - Use previous/next value (time series)
- **Model-based** - Predict missing values using other features
- **Drop rows** - Remove rows with missing data (if minimal)

## Handling Class Imbalance `#important`

### Undersampling
- **Definition:** Reduce the number of majority class samples
- **Example:** 1000 fraud (minority) vs 10,000 non-fraud (majority) → randomly remove 9,000 non-fraud samples
- **Pros:** Balances classes, reduces training time
- **Cons:** Loss of potentially useful data, may underfit
- **When to use:** Very large datasets where losing data is acceptable

### Oversampling
- Duplicate or create synthetic minority class samples
- **Pros:** No data loss
- **Cons:** Risk of overfitting, longer training time

### SMOTE (Synthetic Minority Over-sampling Technique) `#exam-tip`
- **Definition:** Creates **synthetic** samples for minority class using interpolation
- **How it works:**
  1. Select a minority class sample
  2. Find its k-nearest neighbors (typically k=5) in minority class
  3. Randomly select one neighbor
  4. Create synthetic sample along the line between them
  5. New sample = Original + random(0,1) × (Neighbor - Original)
- **Pros:** Better than simple duplication, reduces overfitting
- **Cons:** Can create noisy samples in overlapping regions
- **Available in:** SageMaker Data Wrangler
- **Best practice:** Apply SMOTE only to training data, not validation/test `#gotcha`

### Class Weights
- Assign higher penalty to misclassifying minority class
- No data manipulation needed
- Supported by most ML algorithms

## Data Preprocessing

### Shuffling in ML `#important`
- **Definition:** Randomly reorder training data samples
- **Purpose:**
  - Prevents model from learning order-dependent patterns
  - Ensures random batches during training
  - Breaks temporal or sequential correlations
  - Improves gradient descent convergence
- **When to shuffle:**
  - Always for non-sequential data (tabular, images)
  - Before splitting train/validation/test sets
  - At the start of each training epoch
- **When NOT to shuffle:** Time series data (preserves temporal order) `#gotcha`
- **In SageMaker:** Set `ShuffleConfig` in training job

### Normalization/Scaling
- **Min-Max Scaling** - Scale to [0,1] range
- **Standardization (Z-score)** - Mean=0, StdDev=1
- **Robust Scaling** - Use median and IQR (handles outliers)

### Encoding Categorical Variables
- **One-hot encoding** - Binary columns for each category
- **Label encoding** - Integer mapping (ordinal data)
- **Target encoding** - Replace with target mean

## Feature Selection
- **Filter methods** - Correlation, chi-square
- **Wrapper methods** - Forward/backward selection
- **Embedded methods** - L1 regularization, tree importance

## AWS Tools for Feature Engineering

### SageMaker Data Wrangler `#hands-on`
- Visual interface for data prep
- 300+ built-in transformations
- Supports SMOTE, encoding, scaling, imputation
- Generates code for production

### SageMaker Processing Jobs
- Run preprocessing scripts at scale
- Supports scikit-learn, Spark, custom containers

### AWS Glue DataBrew
- No-code data preparation
- 250+ transformations
- Good for exploratory data prep

## Exam Tips `#exam-tip`
- **Mean vs Median:** Use median for skewed data or outliers
- **Undersampling vs SMOTE:** SMOTE preferred when you can't afford to lose data
- **Always shuffle:** Except for time series
- **Handle missing data before scaling:** Imputation first, then normalization
- **SMOTE only on training data:** Never on test set

## Related Topics
- [Machine Learning Fundamentals](./ml-fundamentals.md)
- [Data Services](./data-services.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
