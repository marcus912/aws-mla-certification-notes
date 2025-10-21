# SageMaker: Training & Fine-Tuning

**Tags:** `#core` `#important` `#exam-tip`

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

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [SageMaker Hyperparameters](./sagemaker-hyperparameters.md)
- [SageMaker JumpStart](./sagemaker-jumpstart.md)
- [Model Training & Evaluation](../core-ml/model-training-evaluation.md)
