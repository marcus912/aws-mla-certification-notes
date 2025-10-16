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
1. 📝 Read the [Study Guide](./study-guide.md) for exam overview and 10-week study plan
2. ⚡ Bookmark the [Cheat Sheet](./cheat-sheet.md) for quick reference
3. 📚 Work through topics in the [Structure](#structure) section below
4. ✅ Track your progress using the [Study Progress](#study-progress) checklist

## Structure

### Core ML Concepts
- [Machine Learning Fundamentals](./ml-fundamentals.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
- [Feature Engineering](./feature-engineering.md)

### AWS ML/AI Services
- [Amazon SageMaker](./sagemaker.md) - Custom ML model building
- [SageMaker Clarify](./sagemaker-clarify.md) - Bias & Explainability
- [AWS ML Algorithms](./aws-ml-algorithms.md) - All 17 SageMaker built-in algorithms
- [AWS AI Services](./aws-ai-services.md) - Comprehend, Rekognition, Lex, Textract, Kendra, etc.
- [Data Services](./data-services.md) - S3, Glue, Athena, EMR, Kinesis, Lake Formation, Ground Truth
- [MLOps & Deployment](./mlops-deployment.md) - Endpoints, monitoring, CI/CD

### Quick References
- [📝 Study Guide](./study-guide.md) - **START HERE!** Exam strategy & roadmap
- [⚡ Cheat Sheet](./cheat-sheet.md) - Quick reference tables
- [📋 Template](./TEMPLATE.md) - Template for creating new notes

## Study Progress

### Core ML Knowledge
- [ ] Machine Learning Fundamentals
- [ ] Model Training & Evaluation
- [ ] Feature Engineering

### AWS ML Services
- [ ] Amazon SageMaker (Custom ML)
- [ ] SageMaker Clarify (Bias Detection)
- [ ] AWS ML Algorithms (17 built-in algorithms)

### AWS AI Services (Pre-trained)
- [ ] NLP Services (Comprehend, Translate, Transcribe, Polly)
- [ ] Vision Services (Rekognition, Textract)
- [ ] Conversational AI (Lex)
- [ ] Search & Recommendations (Kendra, Personalize)
- [ ] Specialized (A2I, Lookout, Fraud Detector, Q)

### Data & MLOps
- [ ] Data Services (S3, Glue, Athena, EMR, Kinesis, Lake Formation, Ground Truth)
- [ ] MLOps & Deployment (Endpoints, Model Monitor, Pipelines)

## Tags
- `#core` - Core exam topic
- `#exam-tip` - Exam-specific insight
- `#hands-on` - Practice/lab required
- `#gotcha` - Common pitfall
- `#important` - High priority

## 📊 Repository Stats

- **Total Notes:** 13 comprehensive markdown files
- **Total Lines:** 4,577 lines of exam-focused content
- **Coverage:** All 4 AWS MLA exam domains (100%)
- **Algorithms Covered:** 17 SageMaker built-in algorithms
  - Supervised: Linear Learner, XGBoost, KNN, Factorization Machines
  - Computer Vision: Image Classification, Object Detection, Semantic Segmentation
  - NLP: BlazingText, Seq2Seq, Object2Vec
  - Time Series: DeepAR
  - Unsupervised: K-Means, PCA, LDA, NTM
  - Anomaly Detection: Random Cut Forest, IP Insights
- **Services Covered:** 25+ AWS ML/AI services
  - 16 AI Services (Comprehend, Rekognition, Lex, Textract, etc.)
  - SageMaker ecosystem (Training, Clarify, Ground Truth, Pipelines, Experiments)
  - Data services (S3, Glue, Athena, EMR, Kinesis, Redshift, Lake Formation)
  - Data Lakes (Lake Formation: column/row security, LF-Tags, permissions)
  - Instance Types (M5, C5, P3, P4d, G4dn, G5, Inf1, Trn1) - Training & inference selection
- **Exam Tips:** 172 `#exam-tip` tags throughout
- **Study Time:** 10-week suggested plan in study guide

## 🤝 Contributing

Contributions are welcome! To maintain consistency:
1. Follow the format in [TEMPLATE.md](./TEMPLATE.md)
2. Keep notes brief and exam-focused
3. Use appropriate tags (`#core`, `#exam-tip`, `#hands-on`, `#gotcha`, `#important`)
4. Update cross-references when adding new content

**For AI Assistants:** See [`.claude/instructions.md`](./.claude/instructions.md) for detailed guidelines on maintaining these notes.

## 📝 Usage

**For Students:**
- Browse topics by category in the [Structure](#structure) section
- Use search (Ctrl/Cmd + F) to find specific keywords
- Check off items in [Study Progress](#study-progress) as you learn
- Review [Cheat Sheet](./cheat-sheet.md) before exam day

**For AI-Assisted Study:**
- Provide keywords or topics, and AI will organize/update notes accordingly
- Example: "Add notes about AWS Forecast" or "Explain concept drift"
- AI follows guidelines in `.claude/instructions.md` for consistency

## 📂 Project Structure

```
aws-mla-certification-notes/
├── .claude/                    # AI assistant instructions
│   ├── instructions.md         # Rules for maintaining notes
│   ├── prompts.md             # Example prompts
│   └── context.md             # Project context
│
├── Core ML Concepts
│   ├── ml-fundamentals.md
│   ├── model-training-evaluation.md
│   └── feature-engineering.md
│
├── AWS Services
│   ├── sagemaker.md
│   ├── sagemaker-clarify.md
│   ├── aws-ml-algorithms.md
│   ├── aws-ai-services.md
│   ├── data-services.md
│   └── mlops-deployment.md
│
├── Quick References
│   ├── study-guide.md         # START HERE!
│   ├── cheat-sheet.md         # Quick reference
│   └── TEMPLATE.md            # Template for new notes
│
└── README.md                   # This file
```

## 🔗 External Resources

- [AWS ML Associate Exam Guide](https://aws.amazon.com/certification/certified-machine-learning-associate/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
- [AWS Skill Builder](https://skillbuilder.aws/) - Official practice exams

## ⚖️ License

MIT License - Feel free to use these notes for your own exam preparation!

## 🙏 Acknowledgments

Created with assistance from Claude (Anthropic) to help ML practitioners prepare efficiently for the AWS MLA certification.

---

**Good luck with your certification journey! 🎓**

If you find these notes helpful, please ⭐ star this repository!
