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

**📖 For detailed algorithm hyperparameters, see [SageMaker Hyperparameters](./sagemaker-hyperparameters.md)**

## Advanced Training Features

**📖 For transfer learning and fine-tuning, see [SageMaker Training & Fine-Tuning](./sagemaker-training.md)**

**📖 For pre-built models and solutions, see [SageMaker JumpStart](./sagemaker-jumpstart.md)**

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

**📖 For detailed pipeline and CI/CD information, see [MLOps CI/CD](../mlops/mlops-cicd.md)**

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
- [SageMaker Hyperparameters](./sagemaker-hyperparameters.md) - Algorithm hyperparameters in detail
- [SageMaker Training & Fine-Tuning](./sagemaker-training.md) - Transfer learning, fine-tuning
- [SageMaker JumpStart](./sagemaker-jumpstart.md) - Pre-built models and solutions
- [MLOps & Deployment](../mlops/mlops-deployment.md) - Deployment strategies
- [MLOps CI/CD](../mlops/mlops-cicd.md) - Pipelines, Model Registry
- [MLOps Experiments](../mlops/mlops-experiments.md) - Experiments, TensorBoard
- [MLOps Monitoring](../mlops/mlops-monitoring.md) - Model Monitor
- [Model Training & Evaluation](../core-ml/model-training-evaluation.md)
- [AWS ML Algorithms](../aws-services/aws-ml-algorithms.md)
- [SageMaker Clarify](./sagemaker-clarify.md)
