# AWS MLA Exam Study Guide

**Tags:** `#exam-tip` `#important`

## Exam Overview

**AWS Certified Machine Learning - Associate (MLA-C01)**
- **Duration:** 170 minutes
- **Format:** 65 questions (multiple choice, multiple response)
- **Passing Score:** 720/1000
- **Cost:** $150 USD
- **Validity:** 3 years

## Exam Domains

| Domain | % of Exam |
|--------|-----------|
| **Domain 1:** Data Preparation | 28% |
| **Domain 2:** Model Development | 26% |
| **Domain 3:** Deployment and Orchestration | 22% |
| **Domain 4:** ML Solution Monitoring, Maintenance, and Security | 24% |

## Domain 1: Data Preparation (28%)

### Key Topics
- Data ingestion and storage (S3, data lakes)
- Data transformation and feature engineering
- Data analysis and visualization
- Handling missing data, outliers, imbalanced datasets

### Critical Services
- **Amazon S3** - Primary data storage
- **AWS Glue** - ETL, Data Catalog, crawlers
- **Amazon Athena** - SQL queries on S3
- **Amazon EMR** - Large-scale Spark/Hadoop processing
- **AWS Glue DataBrew** - No-code data prep
- **SageMaker Data Wrangler** - Visual data prep with ML focus
- **SageMaker Ground Truth** - Data labeling
- **Amazon Kinesis** - Streaming data

### Must-Know Concepts
- Mean vs median imputation
- SMOTE for class imbalance
- Undersampling vs oversampling
- Data shuffling (when and when not to)
- Train/validation/test splits
- Cross-validation
- Feature scaling (normalization, standardization)
- Encoding categorical variables (one-hot, label, target)
- Data formats (CSV, Parquet, RecordIO-protobuf)

### Common Questions
- "How to handle missing data when there are outliers?" → **Median replacement**
- "How to balance classes without losing data?" → **SMOTE**
- "Best format for large tabular datasets?" → **Parquet**
- "Service for auto-discovering schema?" → **Glue Crawler**

## Domain 2: Model Development (26%)

### Key Topics
- Selecting appropriate ML algorithms
- Model training techniques
- Hyperparameter tuning
- Model evaluation metrics

### Critical Services
- **Amazon SageMaker** - Training, built-in algorithms, Autopilot
- **SageMaker Autopilot** - AutoML
- **Built-in algorithms** - Know which algorithm for which problem

### Must-Know Algorithms
| Problem | Algorithm |
|---------|-----------|
| Tabular classification/regression | Linear Learner, XGBoost |
| Anomaly detection | Random Cut Forest |
| Time series forecasting | DeepAR |
| Recommendations | Factorization Machines |
| Clustering | K-Means |
| Dimensionality reduction | PCA |
| Image classification | Image Classification (ResNet) |
| Object detection | Object Detection |
| Text classification | BlazingText |

### Must-Know Concepts
- Regression vs classification
- Bias vs variance
- Overfitting vs underfitting
- Precision, recall, F1 score
- RMSE, MAE, R² for regression
- AUC-ROC interpretation
- Confusion matrix
- Hyperparameter tuning strategies
- Regularization (L1/L2, dropout)

### Common Questions
- "Detect fraudulent transactions in real-time?" → **Kinesis Analytics + RCF**
- "Model performs well on training but poor on test?" → **Overfitting**
- "AutoML without ML expertise?" → **SageMaker Autopilot**
- "Cost-effective training?" → **Managed Spot Training**

## Domain 3: Deployment and Orchestration (22%)

### Key Topics
- Model deployment strategies
- Real-time vs batch inference
- ML pipelines and workflows
- Cost optimization

### Critical Services
- **SageMaker Endpoints** - Real-time inference
- **SageMaker Batch Transform** - Batch predictions
- **SageMaker Serverless Inference** - Intermittent traffic
- **Multi-model Endpoints** - Multiple models, one endpoint
- **SageMaker Pipelines** - ML workflow orchestration
- **AWS Lambda** - Lightweight inference
- **Amazon ECS/EKS** - Container orchestration

### Must-Know Concepts
- Endpoint vs Batch Transform tradeoffs
- When to use Serverless Inference
- A/B testing with production variants
- Canary deployments
- Auto-scaling for endpoints
- Inference pipelines (preprocessing + prediction)
- Model registry and versioning

### Common Questions
- "Intermittent, unpredictable traffic?" → **Serverless Inference**
- "Large dataset, offline predictions?" → **Batch Transform**
- "Low-latency, real-time predictions?" → **SageMaker Endpoint**
- "Multiple models, cost-effective deployment?" → **Multi-model Endpoints**
- "Automate ML workflow?" → **SageMaker Pipelines**

## Domain 4: Monitoring, Maintenance, and Security (24%)

### Key Topics
- Model monitoring and drift detection
- Bias detection and explainability
- Security and compliance
- Model retraining strategies

### Critical Services
- **SageMaker Model Monitor** - Drift detection
- **SageMaker Clarify** - Bias and explainability
- **Amazon CloudWatch** - Metrics and logs
- **AWS KMS** - Encryption
- **AWS IAM** - Access control
- **VPC** - Network isolation

### Must-Know Concepts
- **Data drift** - Input distribution changes
- **Model drift (concept drift)** - Input-output relationship changes
- **Feature attribute drift** - Feature importance changes (NDCG)
- **Prediction drift** - Output distribution changes
- **Bias types** - Pre-training vs post-training
- **Disparate Impact** - DI < 0.8 is red flag
- **SHAP values** - Explain predictions
- Encryption at rest and in transit
- VPC for training and inference
- IAM roles for SageMaker

### Common Questions
- "Production data differs from training data?" → **Data drift** → Model Monitor
- "Detect bias in model predictions?" → **SageMaker Clarify**
- "Explain why model made prediction?" → **SHAP (Clarify)**
- "Feature importance changed over time?" → **Feature attribute drift (NDCG)**
- "Secure training data?" → **S3 encryption, VPC, IAM roles**

## AWS AI Services (Know When to Use)

### Pre-trained Services (No ML Expertise)
| Service | Purpose | Key Exam Points |
|---------|---------|----------------|
| **Comprehend** | Text analysis | Sentiment, entities, custom classification |
| **Translate** | Translation | 75+ languages, custom terminology |
| **Transcribe** | Speech-to-text | Speaker diarization, custom vocab, Medical version |
| **Polly** | Text-to-speech | Neural vs standard voices, SSML |
| **Rekognition** | Image/video analysis | Objects, faces, moderation, custom labels |
| **Textract** | Document OCR | Forms, tables (better than Rekognition for docs) |
| **Lex** | Chatbots | Intents, slots, Lambda fulfillment |
| **Kendra** | Enterprise search | Natural language, ML-powered |
| **Personalize** | Recommendations | Recipes, real-time, needs interaction data |
| **A2I** | Human review | Low-confidence predictions |
| **Lookout** | Anomaly detection | Metrics, Vision, Equipment |
| **Fraud Detector** | Fraud detection | Pre-built + custom models |
| **Q Business** | Enterprise Q&A | Generative AI over company data |

### When to Use AI Services vs SageMaker
- **AI Services:** Pre-trained, common use cases, no ML expertise
- **SageMaker:** Custom models, unique problems, full control

## High-Frequency Exam Topics `#important`

### Top 15 Must-Know Topics
1. **SageMaker Built-in Algorithms** - Which for which problem
2. **Random Cut Forest** - Anomaly detection
3. **Data drift vs Model drift** - Definitions and detection
4. **SMOTE** - Class imbalance handling
5. **Mean vs Median imputation** - When to use which
6. **Bias (SageMaker Clarify)** - Disparate Impact, pre/post-training
7. **SHAP** - Model explainability
8. **Endpoint vs Batch Transform vs Serverless** - When to use
9. **Managed Spot Training** - Cost optimization
10. **Ground Truth** - Data labeling, active learning
11. **Model Monitor** - Drift detection
12. **Rekognition vs Textract** - Natural scenes vs documents
13. **Comprehend** - Sentiment, entities, custom classification
14. **Personalize** - Recommendations, recipes
15. **Feature Store** - Centralized feature repository

### Common Traps `#gotcha`
- **Shuffling time series data** - DON'T (preserves temporal order)
- **SMOTE on test data** - DON'T (only training data)
- **Rekognition for invoices** - NO (use Textract)
- **Spot training without checkpointing** - Will lose progress
- **Not using Parquet for large data** - CSV is slow and large
- **Forgetting VPC for secure training** - Required for compliance
- **Amending other developers' commits** - Never do this

## Study Strategy

### Week 1-2: Foundations
- Review ML fundamentals
- Understand regression vs classification
- Master evaluation metrics
- Learn feature engineering techniques

### Week 3-4: SageMaker Deep Dive
- Built-in algorithms
- Training jobs and options
- Deployment options
- Model Monitor and Clarify

### Week 5-6: AWS AI Services
- Comprehend, Translate, Transcribe, Polly
- Rekognition, Textract
- Lex, Kendra, Personalize
- Specialized services (A2I, Lookout, Fraud Detector)

### Week 7-8: Data & Infrastructure
- S3, Glue, Athena, EMR
- Kinesis for streaming
- Data formats and best practices
- Security (IAM, VPC, KMS)

### Week 9-10: Practice & Review
- Practice exams
- Review cheat sheets
- Hands-on labs (critical!)
- Focus on weak areas

## Hands-On Labs (Critical!) `#hands-on`

### Must-Do Labs
1. **SageMaker Training Job** - Train with built-in algorithm (XGBoost)
2. **SageMaker Endpoint** - Deploy model, test inference
3. **Batch Transform** - Run batch predictions
4. **Model Monitor** - Set up drift detection
5. **Ground Truth** - Create labeling job
6. **Data Wrangler** - Transform data visually
7. **Rekognition** - Detect objects in images
8. **Comprehend** - Analyze text sentiment
9. **Glue Crawler** - Discover schema in S3
10. **Athena** - Query S3 data with SQL

### Free Tier Services
- SageMaker (2 months free for notebooks)
- Comprehend (5M characters/month for 12 months)
- Rekognition (5K images/month for 12 months)
- Translate (2M characters/month for 12 months)
- Transcribe (60 minutes/month for 12 months)
- S3 (5GB storage for 12 months)
- Glue (1M objects for 12 months)

## Exam Day Tips

### Time Management
- 170 minutes / 65 questions ≈ 2.6 minutes per question
- Flag difficult questions, return later
- Don't spend >4 minutes on any question

### Question Strategies
- Eliminate obviously wrong answers first
- Look for keywords: "cost-effective" → Spot training, Batch Transform
- "Real-time" → Endpoint, not Batch Transform
- "No ML expertise" → AI Services, not SageMaker custom
- "Bias detection" → Clarify
- "Anomaly" → Random Cut Forest
- Read carefully: "NOT" questions

### Common Question Patterns
- "Which service should you use?" → Service selection
- "How to optimize cost?" → Spot training, Serverless, Batch Transform
- "How to handle missing data?" → Imputation techniques
- "Model performs poorly on test data?" → Overfitting/underfitting
- "Detect drift in production?" → Model Monitor
- "Explain predictions?" → SHAP, Clarify

## Additional Resources

### AWS Documentation
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [ML Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)
- [Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### Practice
- AWS Skill Builder (official practice exams)
- Hands-on labs in AWS Console
- This notes repository!

## Final Checklist

One week before exam:
- [ ] Review all cheat sheets
- [ ] Retake practice exams (aim for 85%+)
- [ ] Review flagged/weak topics
- [ ] Do hands-on labs for unclear concepts
- [ ] Read through common exam scenarios
- [ ] Get good sleep!

Good luck with your AWS MLA certification! 🎯
