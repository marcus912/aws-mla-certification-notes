# Quick Reference Cheat Sheet

**Tags:** `#important`

## AWS Service Selection

| Use Case | Service |
|----------|---------|
| Build custom ML models | SageMaker |
| Pre-trained foundation models (LLMs, image gen) | Bedrock |
| Chatbot, summarization, content generation | Bedrock |
| Text analysis (sentiment, entities) | Comprehend |
| Text-to-speech | Polly |
| Speech-to-text | Transcribe |
| Translation | Translate |
| Image/video analysis | Rekognition |
| Document OCR (forms, tables) | Textract |
| Chatbots | Lex |
| Enterprise search | Kendra |
| Personalization/recommendations | Personalize |
| Fraud detection | Fraud Detector |
| Human review of ML predictions | Augmented AI (A2I) |
| Data labeling (public data) | Ground Truth + Mechanical Turk |
| Data labeling (confidential data) | Ground Truth + Private workforce |
| Anomaly detection in metrics | Lookout for Metrics |
| Manufacturing defect detection | Lookout for Vision |
| Enterprise Q&A (generative AI) | Q Business |
| AI coding assistant | Q Developer |
| Data labeling | Ground Truth |
| Feature storage | Feature Store |
| Secure data lake (column/row security) | Lake Formation |
| Raw data storage for ML | S3 (Data Lake) |
| Structured BI/analytics | Redshift (Data Warehouse) |

## SageMaker Built-in Algorithm Selection

| Problem Type | Algorithm | Best For |
|--------------|-----------|----------|
| Binary/multi-class classification (tabular) | Linear Learner, XGBoost, KNN | XGBoost (accuracy), Linear Learner (speed) |
| Regression (tabular) | Linear Learner, XGBoost, KNN | XGBoost (default choice) |
| Time series forecasting | DeepAR | Probabilistic forecasts |
| Recommendations | Factorization Machines | Sparse user-item data |
| Clustering | K-Means | Group similar items |
| Dimensionality reduction | PCA | Feature reduction |
| Topic modeling | LDA, NTM | LDA (interpretable), NTM (large/fast) |
| Text classification | BlazingText | Fast sentiment analysis |
| Word embeddings | BlazingText | Word2Vec mode |
| Sequence-to-sequence | Seq2Seq | Machine translation |
| Object embeddings | Object2Vec | Similarity learning |
| Image classification | Image Classification | Single label per image |
| Object detection | Object Detection | Bounding boxes |
| Semantic segmentation | Semantic Segmentation | Pixel-level masks |
| Anomaly detection (general) | Random Cut Forest | Unsupervised outliers |
| Anomaly detection (IP) | IP Insights | Account security |

## Data Format Requirements `#exam-tip`

| Algorithm | Supported Formats |
|-----------|-------------------|
| Linear Learner | RecordIO-protobuf, CSV |
| XGBoost | CSV, LibSVM, Parquet |
| KNN | RecordIO-protobuf, CSV |
| Factorization Machines | RecordIO-protobuf (only) |
| DeepAR | JSON Lines |
| K-Means | RecordIO-protobuf, CSV |
| PCA | RecordIO-protobuf, CSV |
| LDA | RecordIO-protobuf, CSV |
| NTM | RecordIO-protobuf, CSV |
| Random Cut Forest | RecordIO-protobuf, CSV |
| IP Insights | CSV |
| Image Classification | RecordIO, Image files (JPG, PNG) |
| Object Detection | RecordIO, Image files + JSON |
| Semantic Segmentation | Image files + PNG masks |
| BlazingText | Text file (one sentence per line) |
| Seq2Seq | RecordIO-protobuf, JSON |
| Object2Vec | JSON |

## Instance Type Selection `#exam-tip`

### Training Instance Types

| Instance Family | Type | GPU | Best For | Cost |
|-----------------|------|-----|----------|------|
| **M (General)** | ml.m5, ml.m6i | No | Tabular data, XGBoost, prototyping | $ |
| **C (Compute)** | ml.c5, ml.c6i | No | CPU-intensive, high throughput | $ |
| **P3 (GPU)** | ml.p3 | NVIDIA V100 | Deep learning training (standard) | $$$ |
| **P4d (GPU)** | ml.p4d | NVIDIA A100 | Large models, distributed training | $$$$ |
| **G4dn (GPU)** | ml.g4dn | NVIDIA T4 | Cost-effective GPU training | $$ |
| **G5 (GPU)** | ml.g5 | NVIDIA A10G | Balance cost/performance | $$$ |
| **Trn1 (AWS)** | ml.trn1 | Trainium | Cost-optimized DL training | $$ |

### Inference Instance Types

| Instance Family | Type | GPU | Best For | Cost |
|-----------------|------|-----|----------|------|
| **M (General)** | ml.m5 | No | CPU inference, small models | $ |
| **C (Compute)** | ml.c5 | No | High-throughput CPU inference | $ |
| **G4dn (GPU)** | ml.g4dn | NVIDIA T4 | GPU inference, cost-effective | $$ |
| **P3 (GPU)** | ml.p3 | NVIDIA V100 | Low-latency GPU inference | $$$ |
| **Inf1 (AWS)** | ml.inf1 | Inferentia | High-throughput inference | $$ |

### Quick Selection Guide

| Scenario | Choose |
|----------|--------|
| Train XGBoost/Linear Learner on tabular data | ml.m5 |
| Train CNN/RNN (image/text) | ml.p3 |
| Train huge transformer model | ml.p4d |
| Cost-sensitive GPU training | ml.g4dn or ml.trn1 |
| CPU inference | ml.c5 or ml.m5 |
| GPU inference (cost-optimized) | ml.g4dn or ml.inf1 |
| Distributed multi-GPU training | ml.p3.8xlarge or ml.p4d.24xlarge |
| Spot training (90% savings) | Any instance + Managed Spot |

### Key Rules `#exam-tip`
- **Deep Learning (CNNs, RNNs, Transformers)** → GPU (P3, P4d, G4dn)
- **Classical ML (XGBoost, Random Forest, Linear)** → CPU (M5, C5)
- **Training:** Use P-family (P3/P4d) or G4dn for GPU
- **Inference:** Use cheaper options (C5, M5, G4dn, Inf1)
- **Cost optimization:** G4dn (cheapest GPU) or Spot instances
- **Fastest training:** P4d (A100 GPUs)

## Key Formulas

### Classification Metrics
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Important Hyperparameters `#exam-tip`

### Regularization (Prevent Overfitting)

| Parameter | Algorithm | Mechanism | Effect | Use Case |
|-----------|-----------|-----------|--------|----------|
| `l1` | Linear Learner | Forces weights to EXACTLY zero | Feature selection (eliminates features) | Many features, want sparsity |
| `wd` | Linear Learner | Shrinks toward zero (never reaches) | Keeps all features, reduces magnitude | Correlated features |
| `alpha` | XGBoost | L1 penalty (absolute value) | Feature selection | Many features, want sparsity |
| `lambda` | XGBoost | L2 penalty (squared) | Shrinks all weights smoothly | Prevent overfitting, stable models |

### Class Imbalance

| Parameter | Algorithm | Mechanism | When to Use |
|-----------|-----------|-----------|-------------|
| `balance_multiclass_weights` | Linear Learner | Auto-assigns higher weight to minority classes | Imbalanced classes (fraud, rare events) |
| `scale_pos_weight` | XGBoost | Ratio of negative/positive samples | Binary imbalance (set to neg/pos ratio) |
| `target_recall` | Linear Learner | Optimizes to catch all positives (tolerates false alarms) | Fraud/disease detection (can't miss positives) |
| `target_precision` | Linear Learner | Optimizes to avoid false positives (may miss some) | Spam detection (false alarms costly) |

### Training Control

| Parameter | Algorithms | Range | Mechanism | Effect |
|-----------|-----------|-------|-----------|--------|
| `learning_rate` / `eta` | Linear Learner, XGBoost | 0.001-0.3 | Step size shrinkage for weight updates | Lower = slower, more robust, less overfitting |
| `mini_batch_size` | Linear Learner | 1-10000 | Samples per weight update | Small = noisy gradient (regularization effect) |
| `max_depth` | XGBoost | 1-10+ | Limits tree depth (# of levels) | Lower = simpler trees, less overfitting |
| `num_round` | XGBoost | Varies | Number of boosting iterations | More rounds = more complex model |
| `subsample` | XGBoost | 0-1 | Fraction of data per tree (stochastic boosting) | < 1 adds randomness, prevents overfitting |

### Cost Optimization Tips `#exam-tip`
- Use Spot instances with checkpointing (90% savings)
- Batch Transform instead of persistent endpoints
- Serverless Inference for intermittent traffic
- Multi-model endpoints to share resources
- S3 Intelligent-Tiering for training data

## Common Exam Scenarios

| Scenario | Solution |
|----------|----------|
| Cost-effective training | Managed Spot Training |
| Intermittent inference | Serverless Inference |
| Real-time low-latency | SageMaker Endpoint |
| Batch predictions | Batch Transform |
| Multiple models, one endpoint | Multi-model Endpoints |
| Data drift detection | Model Monitor |
| Automate ML workflow | SageMaker Pipelines |
| AutoML (no ML expertise) | SageMaker Autopilot |
| Custom containers | Bring Your Own Container (BYOC) |
| Real-time anomaly detection | Kinesis Data Analytics + RCF |
| Bias detection in models | SageMaker Clarify |
| Class imbalance | SMOTE, undersampling, class weights |
| Missing data with outliers | Median replacement |
| Feature importance drift | NDCG score tracking |
| Extract data from invoices/forms | Textract (not Rekognition) |
| Moderate user content | Rekognition (images), Comprehend (text) |
| Build recommendation engine | Personalize |
| Low-confidence predictions | Augmented AI (human review) |
| Gradient boosting on tabular data | XGBoost |
| Time series with confidence intervals | DeepAR |
| Translate languages | Seq2Seq (custom) or Translate (pre-trained) |
| Generate text (chatbot, summarization) | Bedrock (Claude, Titan) |
| Generate images from text | Bedrock (Stable Diffusion, Titan Image) |
| Q&A over company documents | Bedrock + RAG or Q Business |
| Limited training data for image classification | Transfer learning (ResNet, VGG pre-trained) |
| Reduce training cost | Transfer learning instead of from scratch |
| 500 images for custom classification | Transfer learning (Feature Extraction) |
| Find objects in images | Object Detection |
| Pixel-level segmentation | Semantic Segmentation |
| Account takeover detection | IP Insights |
| Topic discovery in documents | LDA or NTM |
| Column-level data security | Lake Formation (column permissions) |
| Row-level data filtering | Lake Formation (row-level security) |
| Tag-based data permissions | Lake Formation (LF-Tags) |
| Cross-account data sharing | Lake Formation |
| Store raw data for ML | S3 Data Lake |
| Structured BI reporting | Redshift (Data Warehouse) |
| Train deep learning model (CNN) | ml.p3 (GPU instances) |
| Train XGBoost on tabular data | ml.m5 (CPU instances) |
| Cost-effective GPU training | ml.g4dn (T4 GPU) |
| Fastest deep learning training | ml.p4d (A100 GPU) |
| High-throughput inference | ml.inf1 (Inferentia) or ml.c5 |

## Quick AI Service Selection `#exam-tip`

| If you need... | Use... |
|----------------|--------|
| **Chatbot (conversational AI)** | **Bedrock (Claude) or Lex** |
| **Text summarization** | **Bedrock** |
| **Content generation** | **Bedrock** |
| **Image generation from text** | **Bedrock (Stable Diffusion)** |
| **Semantic search** | **Bedrock Embeddings** |
| Sentiment analysis | Comprehend |
| Translate languages | Translate |
| Transcribe audio | Transcribe |
| Generate speech | Polly |
| Detect objects in images | Rekognition |
| Extract invoice data | Textract |
| Task-oriented chatbot (intents/slots) | Lex |
| Natural language search | Kendra |
| Product recommendations | Personalize |
| Detect fraud | Fraud Detector |
| Human verification | A2I |

## Security Best Practices `#important`

### SageMaker Security Checklist
- ✅ **VPC mode** - Private subnets for training/inference
- ✅ **Encryption at rest** - KMS for S3, EBS, model artifacts
- ✅ **Encryption in transit** - TLS for all communications
- ✅ **IAM least privilege** - Separate roles for training/inference/users
- ✅ **VPC Endpoints** - S3 Gateway Endpoint, SageMaker Interface Endpoints
- ✅ **Security Groups** - Restrict inbound to necessary ports only
- ✅ **Secrets Manager** - Never hardcode credentials
- ✅ **CloudTrail** - Audit all API calls
- ✅ **Macie** - Scan for PII before training

### Quick Security Decisions `#exam-tip`

| Scenario | Solution |
|----------|----------|
| Train on sensitive data | VPC + Encryption + IAM least privilege |
| Training can't access S3 in VPC | S3 VPC Gateway Endpoint |
| Find PII before training | Amazon Macie |
| Store database password | AWS Secrets Manager |
| Block specific IP | NACL (not Security Group) |
| Protect API from DDoS | API Gateway + WAF + Shield |
| Enforce HTTPS for S3 | Bucket policy: `aws:SecureTransport=true` |
| Access SageMaker from private subnet | SageMaker VPC Interface Endpoints |
| Encrypt model artifacts | `output_kms_key` parameter |

### IAM Roles for SageMaker

| Role | Needs Access To |
|------|----------------|
| **Training Job** | S3 (training data, models), ECR, CloudWatch Logs, KMS |
| **Endpoint** | S3 (model artifacts), CloudWatch Logs, ECR |
| **Data Scientist** | SageMaker Studio, S3 (read), NO delete production |

### VPC Configuration Options

| Option | Cost | Internet Access | Use Case |
|--------|------|----------------|----------|
| **No VPC** | Free | ✅ Yes | Public data, prototyping |
| **VPC + NAT Gateway** | $$ | ✅ Outbound only | Need packages from internet |
| **VPC + All Endpoints** | $ | ❌ No | Complete isolation, compliance |

### Encryption Keys

| Key Type | Managed By | Rotation | Cost | Use When |
|----------|-----------|----------|------|----------|
| AWS Managed | AWS | Auto (yearly) | Free | Easy, no requirements |
| Customer Managed | You | Manual/Auto | $1/mo | Compliance requires |

## Transfer Learning Quick Reference `#exam-tip`

| Dataset Size | Strategy | Example |
|--------------|----------|---------|
| **Small (< 1K)** | Feature Extraction (freeze base) | 500 images → Use ResNet, train only final layer |
| **Medium (1K-10K)** | Fine-Tuning (unfreeze later layers) | 5000 images → Unfreeze last block + final layer |
| **Large (> 10K)** | Fine-Tuning or From Scratch | 50K images → Fine-tune all layers or train from scratch |

### When to Use Transfer Learning
- ✅ Small dataset (< 10K samples)
- ✅ Limited compute budget
- ✅ Need quick results
- ✅ Similar domain to pre-trained model

### Pre-trained Models
- **Images:** ResNet, VGG, Inception, MobileNet (ImageNet)
- **NLP:** BERT, GPT, RoBERTa (Hugging Face)
- **SageMaker:** Image Classification algorithm (`use_pretrained_model=1`)

### Key Exam Points
- **"Limited data"** → Transfer learning
- **"Cost optimization"** → Transfer learning (faster = cheaper)
- **"Training failing on small dataset"** → Switch to transfer learning
- **Lower learning rate** for fine-tuning (10-100x lower than from scratch)
