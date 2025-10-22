# AWS Machine Learning Associate (MLA) Certification Notes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Certification](https://img.shields.io/badge/AWS-MLA--C01-orange.svg)](https://aws.amazon.com/certification/certified-machine-learning-associate/)

> Comprehensive, exam-focused study notes for the **AWS Certified Machine Learning - Associate (MLA-C01)** certification exam.

## 📖 Overview

This repository contains concise, exam-focused study notes for the AWS Machine Learning Associate certification. All content is optimized for efficient studying with:
- ✅ **Brief, scannable notes** - No lengthy explanations
- ✅ **Exam-focused content** - Prioritizes exam-relevant information
- ✅ **Visual organization** - Tables, comparisons, and quick references
- ✅ **Comprehensive coverage** - All exam domains covered
- ✅ **Hands-on guidance** - Lab recommendations and practice scenarios

## 🎯 Quick Start

**New to this repository?** Start here:
1. 📝 Read the [Study Guide](./guides/study-guide.md) for exam overview and 10-week study plan
2. ⚡ Bookmark the [Cheat Sheet](./guides/cheat-sheet.md) for quick reference
3. 📚 Work through topics in the [Structure](#structure) section below
4. ✅ Track your progress using the [Study Progress](#study-progress) checklist

## Structure

### Core ML Concepts
- [Machine Learning Fundamentals](./core-ml/ml-fundamentals.md)
- [Model Training & Evaluation](./core-ml/model-training-evaluation.md)
- [Feature Engineering](./core-ml/feature-engineering.md)

### AWS ML/AI Services
- [Amazon SageMaker](./sagemaker/sagemaker.md) - Custom ML model building
  - [Hyperparameters](./sagemaker/sagemaker-hyperparameters.md) - Algorithm hyperparameters in detail
  - [Training & Fine-Tuning](./sagemaker/sagemaker-training.md) - Transfer learning, fine-tuning
  - [JumpStart](./sagemaker/sagemaker-jumpstart.md) - Pre-built models and solutions
- [SageMaker Clarify](./sagemaker/sagemaker-clarify.md) - Bias & Explainability
- [AWS ML Algorithms](./aws-services/aws-ml-algorithms.md) - All 17 SageMaker built-in algorithms
- [AWS AI Services](./aws-services/aws-ai-services.md) - Comprehend, Rekognition, Lex, Textract, Kendra, Personalize
- [AWS Generative AI](./aws-services/aws-generative-ai.md) - Bedrock, Amazon Q (foundation models & LLMs)
- [Data Services](./aws-services/data-services.md) - S3, Glue, Athena, EMR, Kinesis, Lake Formation, Ground Truth
- [MLOps & Deployment](./mlops/mlops-deployment.md) - Deployment strategies, inference optimization
  - [Experiments & Tracking](./mlops/mlops-experiments.md) - SageMaker Experiments, TensorBoard
  - [CI/CD](./mlops/mlops-cicd.md) - Model Registry, Pipelines, Kubernetes
  - [Monitoring](./mlops/mlops-monitoring.md) - Model Monitor, observability, cost optimization
- [Security](./security/security.md) - IAM, core principles, security services, best practices
  - [Encryption](./security/security-encryption.md) - KMS, Secrets Manager, encryption at rest & in transit
  - [Network Security](./security/security-network.md) - VPC, security groups, endpoints, SageMaker VPC config

### Quick References
- [📝 Study Guide](./guides/study-guide.md) - **START HERE!** Exam strategy & roadmap
- [⚡ Cheat Sheet](./guides/cheat-sheet.md) - Quick reference tables
- [📋 Template](./guides/TEMPLATE.md) - Template for creating new notes

## Study Progress

### Core ML Knowledge
- [ ] Machine Learning Fundamentals
- [ ] Model Training & Evaluation
- [ ] Feature Engineering

### AWS ML Services
- [ ] Amazon SageMaker (Custom ML)
  - [ ] Hyperparameters (Algorithm configuration)
  - [ ] Training & Fine-Tuning (Transfer learning)
  - [ ] JumpStart (Pre-built models)
- [ ] SageMaker Clarify (Bias Detection)
- [ ] AWS ML Algorithms (17 built-in algorithms)

### AWS AI Services (Pre-trained)
- [ ] NLP Services (Comprehend, Translate, Transcribe, Polly)
- [ ] Vision Services (Rekognition, Textract)
- [ ] Conversational AI (Lex)
- [ ] Search & Recommendations (Kendra, Personalize)
- [ ] Specialized (A2I, Lookout, Fraud Detector)
- [ ] Generative AI (Bedrock, Amazon Q)

### Data & MLOps
- [ ] Data Services (S3, Glue, Athena, EMR, Kinesis, Lake Formation, Ground Truth)
- [ ] MLOps & Deployment (Deployment strategies, inference optimization)
  - [ ] Experiments & Tracking (SageMaker Experiments, TensorBoard)
  - [ ] CI/CD (Model Registry, Pipelines, Kubernetes)
  - [ ] Monitoring (Model Monitor, observability, cost)
- [ ] Security (IAM, Core Principles, Security Services)
  - [ ] Encryption (KMS, Secrets Manager, at rest & in transit)
  - [ ] Network Security (VPC, Security Groups, VPC Endpoints)

## 📝 About Code Examples

Code blocks in these notes are for:
- **Conceptual understanding** - Illustrate how services work
- **Parameter reference** - Show configuration options you'll see in exam scenarios
- **NOT for memorization** - You won't write code on the exam

**Focus on:** Service names, parameter names, workflow concepts - not syntax.

## Tags
- `#core` - Core exam topic
- `#exam-tip` - Exam-specific insight
- `#hands-on` - Practice/lab required
- `#gotcha` - Common pitfall
- `#important` - High priority

## 📊 Repository Stats

- **Total Notes:** 21 comprehensive markdown files
- **Total Lines:** 9,170 lines of exam-focused content
- **Coverage:** All 4 AWS MLA exam domains (100%)
- **Algorithms Covered:** 17 SageMaker built-in algorithms
  - Supervised: Linear Learner, XGBoost, KNN, Factorization Machines
  - Computer Vision: Image Classification, Object Detection, Semantic Segmentation
  - NLP: BlazingText, Seq2Seq, Object2Vec
  - Time Series: DeepAR
  - Unsupervised: K-Means, PCA, LDA, NTM
  - Anomaly Detection: Random Cut Forest, IP Insights
- **Services Covered:** 25+ AWS ML/AI services
  - Traditional AI Services (Comprehend, Rekognition, Lex, Textract, Kendra, Personalize, etc.)
  - Generative AI (Bedrock: Claude, Titan, Stable Diffusion; Amazon Q family; Agents with aliases)
  - SageMaker ecosystem (Training, Clarify, Ground Truth, Pipelines, Experiments, TensorBoard, JumpStart, Role Manager, Lineage Tracking)
  - Data services (S3, Glue, Athena, EMR, Kinesis, Redshift, Lake Formation)
  - Data Lakes (Lake Formation: column/row security, LF-Tags, permissions)
  - Instance Types (M5, C5, P3, P4d, G4dn, G5, Inf1, Trn1) - Training & inference selection
- **Exam Tips:** 333 `#exam-tip` tags throughout
- **Study Time:** 10-week suggested plan in study guide

## 🤝 Contributing

Contributions are welcome! To maintain consistency:
1. Follow the format in [TEMPLATE.md](./guides/TEMPLATE.md)
2. Keep notes brief and exam-focused
3. Use appropriate tags (`#core`, `#exam-tip`, `#hands-on`, `#gotcha`, `#important`)
4. Update cross-references when adding new content

## 📝 Usage

**For Students:**
- Browse topics by category in the [Structure](#structure) section
- Use search (Ctrl/Cmd + F) to find specific keywords
- Check off items in [Study Progress](#study-progress) as you learn
- Review [Cheat Sheet](./guides/cheat-sheet.md) before exam day

**For AI-Assisted Study:**
- Provide keywords or topics, and AI will organize/update notes accordingly
- Example: "Add notes about AWS Forecast" or "Explain concept drift"
- AI follows guidelines in `CLAUDE.md` for consistency

## 📂 Project Structure

```
aws-mla-certification-notes/
├── CLAUDE.md                          # Repository guidance for AI
│
├── core-ml/                           # Core ML Concepts
│   ├── ml-fundamentals.md
│   ├── model-training-evaluation.md
│   └── feature-engineering.md
│
├── sagemaker/                         # Amazon SageMaker
│   ├── sagemaker.md                   # SageMaker hub
│   ├── sagemaker-hyperparameters.md
│   ├── sagemaker-training.md
│   ├── sagemaker-jumpstart.md
│   └── sagemaker-clarify.md
│
├── aws-services/                      # AWS ML/AI Services
│   ├── aws-ml-algorithms.md
│   ├── aws-ai-services.md
│   ├── aws-generative-ai.md
│   └── data-services.md
│
├── mlops/                             # MLOps & Deployment
│   ├── mlops-deployment.md            # Deployment hub
│   ├── mlops-experiments.md
│   ├── mlops-cicd.md
│   └── mlops-monitoring.md
│
├── security/                          # Security
│   ├── security.md                    # Security hub
│   ├── security-encryption.md
│   └── security-network.md
│
├── guides/                            # Quick References
│   ├── study-guide.md                 # START HERE!
│   ├── cheat-sheet.md                 # Quick reference
│   └── TEMPLATE.md                    # Template
│
└── README.md                           # This file
```

## 🔗 External Resources

- [AWS ML Associate Exam Guide](https://aws.amazon.com/certification/certified-machine-learning-associate/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
- [AWS Skill Builder](https://skillbuilder.aws/) - Official practice exams

## ⚖️ License

MIT License - Feel free to use these notes for your own exam preparation!

---

**Good luck with your certification journey! 🎓**

If you find these notes helpful, please ⭐ star this repository.
