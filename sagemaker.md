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

### Training

**Training Jobs**
- Specify algorithm, compute resources, data location
- **Input modes:** File, Pipe (streaming)
- **Instance types:** ml.m5, ml.p3 (GPU), ml.p4d

**Managed Spot Training** `#exam-tip`
- Up to 90% cost savings
- Use checkpoints for interruption handling

**Automatic Model Tuning (Hyperparameter Optimization)**
- Bayesian optimization
- Max 750 training jobs per tuning job

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
