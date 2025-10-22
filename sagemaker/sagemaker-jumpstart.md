# SageMaker JumpStart

**Tags:** `#important` `#exam-tip`

## Overview

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

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [AWS Generative AI](../aws-services/aws-generative-ai.md) - Bedrock comparison
- [SageMaker Training & Fine-Tuning](./sagemaker-training.md)
- [Model Training & Evaluation](../core-ml/model-training-evaluation.md)
