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

## SageMaker Built-in Algorithm Selection

| Problem Type | Algorithm |
|--------------|-----------|
| Binary/multi-class classification (tabular) | Linear Learner, XGBoost |
| Regression (tabular) | Linear Learner, XGBoost |
| Time series forecasting | DeepAR |
| Recommendations | Factorization Machines |
| Clustering | K-Means |
| Dimensionality reduction | PCA |
| Topic modeling | LDA, NTM |
| Text classification | BlazingText |
| Image classification | Image Classification (ResNet) |
| Object detection | Object Detection |
| Semantic segmentation | Semantic Segmentation |
| Anomaly detection | Random Cut Forest |

## Data Format Requirements `#exam-tip`

| Algorithm | Supported Formats |
|-----------|-------------------|
| Linear Learner | RecordIO-protobuf, CSV |
| XGBoost | CSV, LibSVM, Parquet |
| DeepAR | JSON Lines |
| Image Classification | RecordIO, Image files (JPG, PNG) |
| Object Detection | RecordIO, Image files + JSON |
| BlazingText | Text file (one sentence per line) |

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
