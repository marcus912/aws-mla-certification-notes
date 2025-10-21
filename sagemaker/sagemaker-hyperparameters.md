# SageMaker: Algorithm Hyperparameters

**Tags:** `#core` `#important` `#exam-tip`

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

**Decision Framework:** Ask "What's the cost of being wrong?"

- **`target_precision`**
  - Range: 0 to 1
  - **Unique to Linear Learner**
  - **Use when FALSE POSITIVES are costly** (wrongly predicting positive is bad)
  - Trade-off: May miss some positives (lower recall) to avoid false alarms
  - Examples:
    - Spam detection (don't want good emails marked as spam)
    - Medical test screening (don't want false cancer diagnoses)
    - Product recommendations (don't want to annoy users with bad suggestions)
  - Model optimizes to achieve this precision target

- **`target_recall`**
  - Range: 0 to 1
  - **Unique to Linear Learner**
  - **Use when FALSE NEGATIVES are costly** (missing positives is bad)
  - Trade-off: May have more false alarms (lower precision) to catch all positives
  - Examples:
    - Fraud detection (must catch all fraud, false alarms are tolerable)
    - Disease screening (can't miss any sick patients)
    - Security threats (better to investigate false alarms than miss real threats)
  - Model optimizes to achieve this recall target

- **Cannot set both:** Choose `target_precision` OR `target_recall`, not both
- **Default:** If neither is set, optimizes for overall accuracy

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
  - Mechanism: Limits how many levels deep each tree can grow
  - Effect: Controls tree complexity (deeper = more splits = more overfitting risk)
  - Lower value = simpler trees, less overfitting
- **`num_round`**
  - Number of boosting rounds (trees to build)
  - More rounds = more complex model
- **`min_child_weight`**
  - Minimum sum of instance weight (# of samples) needed in a leaf node
  - Mechanism: Prevents splits when child nodes would have too few samples
  - Higher value = more conservative (stops tree from making splits on small groups)
  - Effect: Prevents overfitting by avoiding overly specific rules

#### Regularization
- **`alpha`** - L1 regularization
  - Default: 0
  - Increase for feature selection
- **`lambda`** - L2 regularization
  - Default: 1
  - Increase to prevent overfitting

#### Learning
- **`eta`** (learning rate / step size shrinkage)
  - Range: 0 to 1
  - Default: 0.3
  - Effect: Controls weight update size per iteration
  - Prevents overfitting by making smaller, gradual updates
  - Lower = slower training, more robust, less overfitting
- **`subsample`**
  - Fraction of training data to sample per tree (stochastic gradient boosting)
  - Range: 0 to 1
  - Default: 1
  - Mechanism: Introduces randomness by training each tree on random subset
  - Lower value (e.g., 0.7-0.8) prevents overfitting through variance reduction

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

#### Instance Types for Training `#exam-tip`

**General Purpose (M Family)**
- **ml.m5**, **ml.m6i** - Balanced CPU/memory
- **vCPUs:** 2-96, **Memory:** 8-384 GB
- **Use cases:**
  - Small to medium datasets
  - Classical ML (XGBoost, Linear Learner)
  - Prototyping and development
  - Cost-effective baseline
- **When to use:** Tabular data, tree-based models, don't need GPU

**Compute Optimized (C Family)**
- **ml.c5**, **ml.c6i** - High CPU performance
- **vCPUs:** 2-96, **Memory:** 4-192 GB
- **Use cases:**
  - CPU-intensive algorithms
  - Batch transform with high throughput
  - Inference with CPU
- **When to use:** Need high CPU but not GPU, cost-conscious

**GPU Training (P Family)** `#important`
- **ml.p3** - NVIDIA Tesla V100 GPUs
  - **GPUs:** 1-8 per instance
  - **GPU Memory:** 16 GB per GPU
  - **Use cases:** Deep learning training (CNNs, RNNs, Transformers)
  - **Best for:** Computer vision, NLP, large neural networks

- **ml.p4d** - NVIDIA A100 GPUs (most powerful)
  - **GPUs:** 8 per instance
  - **GPU Memory:** 40 GB per GPU
  - **Use cases:** Large-scale training, distributed training
  - **Best for:** Huge models (GPT-like), fastest training
  - **Cost:** Most expensive (use for production, not experiments)

- **ml.p4de** - NVIDIA A100 GPUs with 80 GB memory
  - **Best for:** Extremely large models with huge memory needs

**GPU Training (G Family)**
- **ml.g4dn** - NVIDIA T4 GPUs
  - **GPUs:** 1-8 per instance
  - **GPU Memory:** 16 GB per GPU
  - **Use cases:** Cost-effective GPU training, mixed training/inference
  - **Best for:** Medium-sized models, budget-conscious GPU needs
  - **Trade-off:** Slower than P3 but cheaper

- **ml.g5** - NVIDIA A10G GPUs
  - **Best for:** Balance between cost and performance

**Accelerated Computing (Inf/Trn Family)** `#exam-tip`
- **ml.inf1** - AWS Inferentia chips
  - **Purpose:** Cost-optimized inference (not training)
  - **Use cases:** Deploy models for inference only
  - **Best for:** High-throughput inference workloads

- **ml.trn1** - AWS Trainium chips
  - **Purpose:** Cost-optimized training
  - **Use cases:** Train deep learning models at lower cost than P3/P4
  - **Best for:** Large-scale training with cost optimization

**Instance Type Selection Guide:** `#exam-tip`

| Workload | Instance Type | Reason |
|----------|---------------|--------|
| Tabular data (XGBoost, Linear Learner) | **ml.m5** | No GPU needed, balanced resources |
| Small neural network prototyping | **ml.g4dn** | Cost-effective GPU |
| Large image classification training | **ml.p3** | V100 GPUs for deep learning |
| Huge transformer models | **ml.p4d** | A100 GPUs, most powerful |
| Cost-sensitive GPU training | **ml.g4dn** or **ml.trn1** | Lower cost than P3 |
| CPU-only inference | **ml.c5** or **ml.m5** | No GPU overhead |
| High-throughput inference | **ml.inf1** | AWS Inferentia optimized |
| Distributed training (multi-GPU) | **ml.p3.8xlarge** or **ml.p4d.24xlarge** | 8 GPUs per instance |

**Key Exam Concepts:** `#exam-tip`
1. **GPU vs CPU:**
   - Deep learning (CNNs, RNNs) → GPU (P3, G4dn)
   - Classical ML (XGBoost, trees) → CPU (M5, C5)

2. **Training vs Inference:**
   - Training: P3, P4d (high performance)
   - Inference: G4dn, Inf1, C5 (cost-optimized)

3. **Cost Optimization:**
   - Cheapest GPU: G4dn (T4)
   - Fastest: P4d (A100)
   - Balance: P3 (V100)
   - CPU: M5 (general), C5 (compute-heavy)

4. **When to use each:**
   - **M5:** Default for non-GPU workloads
   - **C5:** CPU-intensive, need more CPU than memory
   - **P3:** Standard GPU training (most common exam answer)
   - **P4d:** Largest models, distributed training
   - **G4dn:** Budget GPU training/inference
   - **Inf1:** High-volume inference only

**Managed Spot Training** `#exam-tip`
- Up to 90% cost savings
- Use checkpoints for interruption handling
- Works with any instance type
- **Best practice:** Use for training, not inference

**Automatic Model Tuning (Hyperparameter Optimization)**
- Bayesian optimization
- Max 750 training jobs per tuning job

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [SageMaker Training & Fine-Tuning](./sagemaker-training.md)
- [AWS ML Algorithms](../aws-services/aws-ml-algorithms.md)
- [Model Training & Evaluation](../core-ml/model-training-evaluation.md)
