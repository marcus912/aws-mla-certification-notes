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

## Transfer Learning & Fine-Tuning `#important` `#exam-tip`

### What is Transfer Learning?

**Definition:** Use a pre-trained model as starting point instead of training from scratch.

**Key Concept:** Pre-trained models have already learned useful features from large datasets. You adapt them to your specific task with less data and training time.

**Common Use Cases:**
- **Computer Vision:** Use models trained on ImageNet (1.4M images, 1000 classes)
- **NLP:** Use models trained on massive text corpora (BERT, GPT)
- **Time Series:** Use models trained on similar time series data

### Why Transfer Learning? `#exam-tip`

**Benefits:**

| Benefit | Explanation | Exam Relevance |
|---------|-------------|----------------|
| **Less training data needed** | Pre-trained features work with 100s vs 1000s of images | Small dataset scenarios |
| **Faster training** | Start closer to optimal, fewer epochs needed | Cost optimization questions |
| **Better performance** | Leverages knowledge from large datasets | Limited data scenarios |
| **Lower cost** | Less compute time = lower training costs | Budget-constrained scenarios |

**When to Use Transfer Learning:** `#important`

✅ **Use Transfer Learning when:**
- Small dataset (< 10,000 samples)
- Similar problem domain exists (ImageNet for images, BERT for text)
- Limited compute budget
- Need quick results
- Task is similar to pre-trained model's task

❌ **Train from Scratch when:**
- Very large dataset (millions of samples)
- Unique domain (medical images, satellite imagery)
- Have significant compute resources
- Pre-trained models perform poorly on your domain

### Transfer Learning Strategies `#exam-tip`

**Three Main Approaches:**

#### 1. Feature Extraction (Frozen Base)
**How:** Use pre-trained model as fixed feature extractor, only train final classification layer.

**Process:**
1. Load pre-trained model (e.g., ResNet-50)
2. **Freeze** all layers except final layer
3. Replace final layer with new layer for your classes
4. Train only the new layer

**When to use:**
- **Very small dataset** (100-1000 samples)
- **Similar domain** to pre-trained model
- **Limited compute**

**Pros:**
- ✅ Very fast training (only one layer)
- ✅ Less overfitting (fewer parameters)
- ✅ Lowest cost

**Cons:**
- ❌ Less flexible (can't adapt features)
- ❌ May not work if domain is different

#### 2. Fine-Tuning (Partially Frozen)
**How:** Freeze early layers, train later layers + final layer.

**Process:**
1. Load pre-trained model
2. **Freeze early layers** (low-level features like edges)
3. **Unfreeze later layers** (high-level features)
4. Train unfrozen layers + new final layer
5. Use **lower learning rate** (don't destroy pre-trained weights)

**When to use:**
- **Medium dataset** (1,000-10,000 samples)
- **Somewhat different domain**
- **More compute available**

**Pros:**
- ✅ Better adaptation to your data
- ✅ Still faster than training from scratch
- ✅ Balances performance and cost

**Cons:**
- ❌ Risk of overfitting if dataset too small
- ❌ Need to tune learning rate carefully

#### 3. Full Fine-Tuning (All Layers)
**How:** Train entire network with pre-trained weights as initialization.

**Process:**
1. Load pre-trained model
2. **Unfreeze all layers**
3. Train entire network with **very low learning rate**
4. Pre-trained weights as initialization (not random)

**When to use:**
- **Large dataset** (10,000+ samples)
- **Different domain** from pre-trained
- **Maximum performance needed**

**Pros:**
- ✅ Maximum adaptation to your data
- ✅ Best performance potential

**Cons:**
- ❌ Longest training time
- ❌ Highest cost
- ❌ Risk of overfitting without enough data

### Transfer Learning Decision Framework `#important` `#exam-tip`

| Dataset Size | Domain Similarity | Strategy | Reason |
|--------------|------------------|----------|--------|
| **Small (< 1K)** | Very similar | **Feature Extraction** | Avoid overfitting, use pre-trained features |
| **Small (< 1K)** | Different | **Fine-Tuning** (few layers) | Need some adaptation, be careful |
| **Medium (1K-10K)** | Similar | **Fine-Tuning** (later layers) | Balance adaptation and overfitting |
| **Medium (1K-10K)** | Different | **Full Fine-Tuning** (low LR) | Need significant adaptation |
| **Large (> 10K)** | Similar | **Fine-Tuning** or **From Scratch** | Either works |
| **Large (> 10K)** | Different | **Train from Scratch** | Have data to learn domain-specific features |

**Key Exam Principle:** `#exam-tip`
- **Smaller dataset → Less training** (Feature Extraction)
- **Larger dataset → More training** (Fine-Tuning or From Scratch)
- **Similar domain → Use more pre-trained** (Feature Extraction)
- **Different domain → Train more layers** (Fine-Tuning or From Scratch)

### SageMaker Transfer Learning Support `#important`

**Popular Frameworks with Pre-trained Models:**

#### 1. TensorFlow/Keras
**Pre-trained Models Available:**
- **ImageNet:** ResNet, VGG, Inception, MobileNet, EfficientNet
- **BERT, GPT** for NLP

**SageMaker Implementation:**
```python
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    framework_version='2.12',
    py_version='py310'
)

estimator.fit({'training': s3_training_data})
```

**In train.py:**
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load pre-trained ResNet50 (without top layer)
base_model = ResNet50(
    weights='imagenet',  # Pre-trained on ImageNet
    include_top=False,   # Remove final classification layer
    input_shape=(224, 224, 3)
)

# Freeze base model (Feature Extraction)
base_model.trainable = False

# Add custom layers for your task
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)
```

**Fine-Tuning (Unfreeze later layers):**
```python
# After initial training, unfreeze some layers
base_model.trainable = True

# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:100]:
    layer.trainable = False  # Freeze first 100 layers

# Re-compile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
model.fit(train_data, epochs=10)
```

#### 2. PyTorch
**Pre-trained Models Available:**
- **torchvision.models:** ResNet, VGG, Inception, MobileNet
- **Hugging Face Transformers:** BERT, GPT, RoBERTa

**SageMaker Implementation:**
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    framework_version='2.0',
    py_version='py310'
)
```

**In train.py:**
```python
import torch
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Feature Extraction: Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Only final layer will be trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**Fine-Tuning: Unfreeze layers**
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

#### 3. Hugging Face Transformers (NLP)
**Pre-trained Models for NLP:**
- BERT, DistilBERT, RoBERTa
- GPT-2, GPT-3
- T5, BART

**SageMaker Implementation:**
```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39'
)
```

**In train.py:**
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes
)

# All layers trainable by default (Fine-Tuning)
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,  # Low learning rate for fine-tuning
    num_train_epochs=3,
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

### SageMaker Built-in Image Classification Algorithm `#exam-tip`

**Uses Transfer Learning Automatically:**

**SageMaker Image Classification algorithm:**
- Built on **ResNet** architecture
- Pre-trained on **ImageNet**
- Automatically does transfer learning
- **Two modes:**
  - **Full training mode:** Train all layers (large dataset)
  - **Transfer learning mode:** Freeze early layers (small dataset)

**Configuration:**
```python
import sagemaker
from sagemaker import image_uris

# Get built-in Image Classification algorithm
image_uri = image_uris.retrieve('image-classification', region, version='latest')

estimator = sagemaker.estimator.Estimator(
    image_uri,
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    output_path=s3_output
)

# Hyperparameters
estimator.set_hyperparameters(
    num_classes=10,
    num_training_samples=5000,
    use_pretrained_model=1,  # Use transfer learning (1=yes, 0=no)
    epochs=10,
    learning_rate=0.001,
    mini_batch_size=128
)

estimator.fit({'train': s3_train_data, 'validation': s3_val_data})
```

**Key Hyperparameter:** `#exam-tip`
- **`use_pretrained_model`**
  - `1` - Transfer learning (default, recommended for small datasets)
  - `0` - Train from scratch (for very large datasets)

### Best Practices for Fine-Tuning `#exam-tip`

**1. Learning Rate** `#important`
- **Critical:** Use lower learning rate than training from scratch
- **Feature Extraction:** Normal LR (0.001-0.01)
- **Fine-Tuning:** 10-100x lower LR (0.00001-0.0001)
- **Why:** Prevent destroying pre-trained weights

**2. Data Preprocessing**
- **Match pre-trained model's preprocessing**
- Example: ImageNet models expect images normalized with specific mean/std
- **Don't change:** Input size, normalization method

**3. Layer Freezing Strategy**
- **Early layers:** Low-level features (edges, textures) - usually freeze
- **Later layers:** High-level features (object parts) - unfreeze for fine-tuning
- **Final layer:** Always replace and train

**4. Gradual Unfreezing** (Advanced)
- Start with frozen base, train final layer
- Unfreeze last block, train
- Gradually unfreeze more layers
- **Benefit:** More stable training

**5. Data Augmentation**
- Use more aggressive augmentation for small datasets
- Prevents overfitting during fine-tuning
- Examples: rotation, flip, color jitter, random crops

### Exam Scenarios `#important` `#exam-tip`

| Scenario | Solution | Reasoning |
|----------|----------|-----------|
| "Classify 500 images into 10 custom categories" | **Transfer learning** (Feature Extraction) with ResNet/VGG | Small dataset, use pre-trained ImageNet features |
| "Train from scratch failing with 1000 images" | **Switch to transfer learning** | Not enough data for from-scratch training |
| "Custom medical images, ImageNet transfer learning not working" | **Fine-tune later layers** or **train from scratch** | Domain too different from ImageNet |
| "Reduce training cost for image classification" | **Use transfer learning** instead of from scratch | Faster training = lower cost |
| "Classify text into sentiment (limited labeled data)" | **Fine-tune pre-trained BERT** | Transfer learning for NLP |
| "Need quick prototype for image classification" | **SageMaker Image Classification with `use_pretrained_model=1`** | Built-in transfer learning |
| "10,000 images, similar to ImageNet" | **Fine-tune ResNet** (unfreeze later layers) | Medium dataset, similar domain |
| "100,000 images, very different domain" | **Train from scratch** | Large dataset, unique domain |

**Key Exam Questions to Expect:**

1. **"Limited training data, how to improve model?"**
   → **Transfer learning** (leverage pre-trained models)

2. **"Model training too expensive?"**
   → **Transfer learning** (faster, cheaper than from scratch)

3. **"Which pre-trained model for images?"**
   → **ResNet, VGG, Inception** (ImageNet pre-trained)

4. **"How to adapt ImageNet model to custom classes?"**
   → **Replace final layer, freeze base, train final layer**

5. **"Training from scratch overfitting with small dataset?"**
   → **Switch to transfer learning** (Feature Extraction mode)

### Transfer Learning vs Training from Scratch `#exam-tip`

| Aspect | Transfer Learning | Training from Scratch |
|--------|------------------|----------------------|
| **Data needed** | 100s-1000s | 10,000s-millions |
| **Training time** | Hours | Days-weeks |
| **Cost** | $ | $$$ |
| **Performance (small data)** | High | Poor (overfits) |
| **Performance (large data)** | High | Very high |
| **Domain similarity** | Must be similar | Any domain |
| **Exam default** | **Recommended** for most scenarios | Only with huge datasets |

**Exam Tip:** `#important`
- If dataset is **small** (< 10K) → **Transfer learning** is almost always the answer
- If question mentions **limited data** or **cost optimization** → **Transfer learning**
- If question mentions **quick results** → **Transfer learning**
- Only choose **from scratch** if explicitly stated: "massive dataset" or "completely unique domain"

## SageMaker JumpStart `#important` `#exam-tip`

### What is JumpStart?

**Definition:** Pre-built ML solutions and foundation models with one-click deployment in SageMaker Studio.

**Key Concept:** Accelerate ML development by starting with pre-trained models and solution templates instead of building from scratch.

**Main Components:**
1. **Foundation Models** - Pre-trained LLMs and vision models (Llama 2, Falcon, BLOOM, Stable Diffusion)
2. **Solution Templates** - End-to-end ML solutions for common problems
3. **Fine-tuning Support** - Customize models with your data
4. **One-click Deployment** - Deploy models to endpoints instantly

### Key Features `#exam-tip`

**1. Foundation Models Library**
- **Text Models:** Llama 2, Llama 3, Falcon, BLOOM, Flan-T5, GPT-J
- **Multimodal:** CLIP, Stable Diffusion (text-to-image)
- **Vision:** Object detection, image classification models
- **From:** Hugging Face, AI21, Cohere, Meta, Stability AI

**2. Solution Templates**
Pre-built end-to-end solutions for:
- Fraud detection
- Predictive maintenance
- Demand forecasting
- Churn prediction
- Credit risk prediction
- Product recommendations

**3. Fine-tuning Capabilities**
- Fine-tune foundation models on your data
- Bring your own training data
- Automated training job setup
- Cost estimation before training

**4. One-Click Deployment**
- Deploy to SageMaker endpoints
- No code required
- Automatic infrastructure setup
- Immediate inference

### JumpStart vs Bedrock `#important` `#exam-tip`

**When to use JumpStart:**

✅ **Choose JumpStart when:**
- Need to **fine-tune** open-source foundation models (Llama, Falcon)
- Want **full control** over model deployment (instance types, endpoints)
- Need **SageMaker integration** (Pipelines, Experiments, Feature Store)
- Want to use **solution templates** for common ML problems
- Prefer **open-source models** (not proprietary)
- Need **custom deployment options** (VPC, security configurations)

**Use Cases:**
- Fine-tune Llama 2 on proprietary data
- Deploy Stable Diffusion for image generation
- Use pre-built fraud detection solution
- Customize open-source LLMs

**When to use Bedrock:**

✅ **Choose Bedrock when:**
- Need **serverless** foundation model access (no infrastructure management)
- Want **proprietary models** (Claude, Titan)
- Need **simple API access** (no deployment needed)
- Building **RAG applications** (Knowledge Bases)
- Want **managed guardrails** and safety features
- Prefer **pay-per-token** pricing

**Use Cases:**
- Build chatbot with Claude
- RAG over company documents
- Quick prototyping without deployment

### Comparison Table `#exam-tip`

| Aspect | JumpStart | Bedrock |
|--------|-----------|---------|
| **Model Types** | Open-source (Llama, Falcon, BLOOM) | Proprietary + open-source (Claude, Titan, Llama) |
| **Deployment** | SageMaker endpoints (you manage) | Serverless API (AWS manages) |
| **Infrastructure** | Choose instances, configure VPC | Fully managed, serverless |
| **Fine-tuning** | Full control, any framework | Managed fine-tuning (limited models) |
| **Pricing** | Pay for endpoints (hourly) | Pay per token/request |
| **Integration** | SageMaker ecosystem (Pipelines, etc.) | API-first, simpler integration |
| **Customization** | Complete (code, containers, config) | Limited (API parameters, prompts) |
| **Solution Templates** | ✅ Yes (fraud, churn, demand forecasting) | ❌ No |
| **Best For** | ML teams needing control & customization | Developers needing quick API access |

### Key Exam Distinctions `#exam-tip`

**JumpStart:**
- "Deploy open-source LLM (Llama, Falcon)"
- "Fine-tune foundation model with custom data"
- "Use pre-built ML solution template"
- "Full control over deployment configuration"
- "Integrate with SageMaker Pipelines"

**Bedrock:**
- "Serverless access to Claude or Titan"
- "Build chatbot without managing infrastructure"
- "RAG application with Knowledge Bases"
- "Pay-per-token pricing model"
- "Simple API access to foundation models"

### Using JumpStart `#exam-tip`

**Access Methods:**

**1. SageMaker Studio UI**
```
SageMaker Studio → JumpStart → Browse Models → Select → Deploy
```

**2. Python SDK**
```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy Llama 2 model
model = JumpStartModel(
    model_id="meta-textgeneration-llama-2-7b",
    region="us-east-1"
)

# Deploy to endpoint
predictor = model.deploy()

# Inference
response = predictor.predict({
    "inputs": "What is machine learning?",
    "parameters": {"max_new_tokens": 256}
})
```

**3. Fine-tuning Example**
```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

# Fine-tune Llama 2
estimator = JumpStartEstimator(
    model_id="meta-textgeneration-llama-2-7b",
    environment={"accept_eula": "true"}
)

# Train on your data
estimator.fit({
    "training": "s3://bucket/training-data"
})

# Deploy fine-tuned model
predictor = estimator.deploy()
```

### Solution Templates `#exam-tip`

**Pre-built End-to-End Solutions:**

| Solution | Problem Type | What's Included |
|----------|--------------|-----------------|
| **Fraud Detection** | Classification | Data prep, model training, evaluation, deployment |
| **Predictive Maintenance** | Time series | Feature engineering, DeepAR, monitoring |
| **Demand Forecasting** | Time series | Historical data analysis, forecasting models |
| **Churn Prediction** | Classification | Feature engineering, XGBoost, explainability |
| **Credit Risk** | Classification | Tabular data processing, model interpretation |
| **Product Recommendations** | Personalization | Collaborative filtering, cold-start handling |

**Benefits:**
- ✅ **Accelerated development** - Working solution in minutes
- ✅ **Best practices** - Pre-configured with AWS best practices
- ✅ **Customizable** - Modify notebooks and code
- ✅ **Production-ready** - Includes deployment and monitoring

**Exam Scenario:** `#exam-tip`
- "Quickly prototype fraud detection system" → **JumpStart Fraud Detection solution**
- "Need working demand forecasting baseline fast" → **JumpStart solution template**

### Use Cases `#exam-tip`

**1. Foundation Model Deployment**
- **Problem:** Need to deploy Llama 2 for text generation
- **Solution:** JumpStart one-click deployment
- **Benefit:** No infrastructure setup, ready-to-use endpoint

**2. Fine-tuning Open-Source LLMs**
- **Problem:** Claude (Bedrock) doesn't support your use case, need open-source
- **Solution:** JumpStart fine-tune Llama 2/Falcon on your data
- **Benefit:** Full control, customization

**3. Rapid Prototyping**
- **Problem:** Need proof-of-concept for churn prediction
- **Solution:** JumpStart Churn Prediction solution template
- **Benefit:** Working baseline in hours, not weeks

**4. Image Generation**
- **Problem:** Generate images from text descriptions
- **Solution:** JumpStart Stable Diffusion deployment
- **Benefit:** One-click deployment of Stable Diffusion model

### Exam Scenarios `#important` `#exam-tip`

| Scenario | Solution | Reasoning |
|----------|----------|-----------|
| "Deploy Llama 2 for internal chatbot" | **JumpStart** | Open-source LLM deployment |
| "Need Claude for customer-facing chatbot" | **Bedrock** | Proprietary model, managed service |
| "Fine-tune Falcon on domain-specific text" | **JumpStart** | Fine-tuning open-source models |
| "Quick fraud detection baseline needed" | **JumpStart solution template** | Pre-built end-to-end solution |
| "Serverless text generation API" | **Bedrock** | No infrastructure management |
| "Deploy Stable Diffusion for image generation" | **JumpStart** | One-click model deployment |
| "Need full control over LLM deployment" | **JumpStart** | SageMaker endpoint customization |
| "Build RAG application with company docs" | **Bedrock Knowledge Bases** | Managed RAG solution |

### Best Practices `#exam-tip`

**1. Model Selection**
- Review model card (performance, license, use cases)
- Test in playground before deployment
- Consider model size vs. inference cost

**2. Fine-tuning**
- Start with small models (7B parameters) before large (70B)
- Use appropriate instance types (ml.g5 or ml.p3 for training)
- Validate fine-tuned model before production deployment

**3. Deployment**
- Use smallest instance type that meets latency requirements
- Consider Serverless Inference for variable traffic
- Enable auto-scaling for production workloads

**4. Cost Optimization**
- Delete unused endpoints (pay for uptime)
- Use Batch Transform for offline inference
- Consider Bedrock for pay-per-token if usage is low

### Limitations `#exam-tip`

- **Not all models support fine-tuning** - Check model card
- **Open-source licenses** - Understand usage restrictions (some models not for commercial use)
- **Endpoint costs** - Pay for running endpoints (unlike Bedrock serverless)
- **Model updates** - Manual process to update to newer model versions
- **Limited to SageMaker** - Models deploy only to SageMaker endpoints

### JumpStart vs Other Options `#exam-tip`

| Need | JumpStart | Bedrock | SageMaker Custom |
|------|-----------|---------|------------------|
| **Open-source LLM** | ✅ Best | ✅ Some models | ✅ Bring your own |
| **Proprietary LLM (Claude)** | ❌ No | ✅ Best | ❌ No |
| **One-click deploy** | ✅ Yes | ✅ API only | ❌ Manual setup |
| **Fine-tune control** | ✅ Full | ⚠️ Limited | ✅ Complete |
| **Solution templates** | ✅ Yes | ❌ No | ❌ No |
| **Serverless** | ❌ Endpoints | ✅ Yes | ❌ Endpoints |
| **Custom frameworks** | ⚠️ Limited | ❌ No | ✅ Any framework |

**Exam Decision Tree:**
1. Need **proprietary model (Claude, Titan)** → **Bedrock**
2. Need **open-source LLM (Llama, Falcon)** → **JumpStart**
3. Need **solution template (fraud, churn)** → **JumpStart**
4. Need **serverless** → **Bedrock**
5. Need **full ML control** → **SageMaker Custom**

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
