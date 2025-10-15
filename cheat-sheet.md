# Quick Reference Cheat Sheet

**Tags:** `#important`

## AWS Service Selection

| Use Case | Service |
|----------|---------|
| Build custom ML models | SageMaker |
| Pre-trained foundation models | Bedrock |
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

## Instance Type Selection

| Use Case | Instance Type |
|----------|---------------|
| General purpose training | ml.m5.* |
| GPU training (deep learning) | ml.p3.*, ml.p4d.* |
| Large-scale training | ml.p4d.24xlarge |
| Inference (CPU) | ml.c5.*, ml.m5.* |
| Inference (GPU) | ml.g4dn.*, ml.p3.* |
| Cost optimization | Spot instances (90% savings) |

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

| Parameter | Algorithm | Effect | Use Case |
|-----------|-----------|--------|----------|
| `l1` | Linear Learner | Feature selection (zeros out) | Many features, want sparsity |
| `wd` | Linear Learner | Shrink weights (L2) | Correlated features |
| `alpha` | XGBoost | L1 regularization | Feature selection |
| `lambda` | XGBoost | L2 regularization | Prevent overfitting |

### Class Imbalance

| Parameter | Algorithm | Purpose |
|-----------|-----------|---------|
| `balance_multiclass_weights` | Linear Learner | Auto-weight minority classes |
| `scale_pos_weight` | XGBoost | Ratio of negative/positive |
| `target_recall` | Linear Learner | Optimize for recall (fraud) |
| `target_precision` | Linear Learner | Optimize for precision (spam) |

### Training Control

| Parameter | Algorithms | Range | Effect |
|-----------|-----------|-------|--------|
| `learning_rate` / `eta` | Linear Learner, XGBoost | 0.001-0.3 | Convergence speed |
| `mini_batch_size` | Linear Learner | 1-10000 | Training speed |
| `max_depth` | XGBoost | 1-10+ | Tree complexity |
| `num_round` | XGBoost | Varies | Number of trees |

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
| Gradient boosting on SageMaker | XGBoost (NOT LightGBM) |
| Time series with confidence intervals | DeepAR |
| Translate languages | Seq2Seq (custom) or Translate (pre-trained) |
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

## Quick AI Service Selection `#exam-tip`

| If you need... | Use... |
|----------------|--------|
| Sentiment analysis | Comprehend |
| Translate languages | Translate |
| Transcribe audio | Transcribe |
| Generate speech | Polly |
| Detect objects in images | Rekognition |
| Extract invoice data | Textract |
| Voice/text chatbot | Lex |
| Natural language search | Kendra |
| Product recommendations | Personalize |
| Detect fraud | Fraud Detector |
| Human verification | A2I |

## Security Best Practices
- Use VPC for training and inference
- Enable encryption at rest (S3, EBS)
- Enable encryption in transit (TLS)
- Use IAM roles for permissions
- Enable CloudWatch logging
- Use AWS KMS for key management
