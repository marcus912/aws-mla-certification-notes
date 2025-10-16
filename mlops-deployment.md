# MLOps & Deployment

**Tags:** `#core` `#important` `#exam-tip`

## Overview
ML Operations (MLOps) - practices for deploying, monitoring, and maintaining ML models in production.

## Deployment Strategies `#exam-tip`

### Real-time Inference (SageMaker Endpoints)
**Purpose:** Low-latency predictions for individual requests

**Characteristics:**
- Persistent endpoint (always running)
- Millisecond latency
- Auto-scaling supported
- Best for: User-facing applications, APIs

**Types:**
- **Single model endpoint** - One model, one endpoint
- **Multi-model endpoint** - Multiple models on one endpoint (cost savings)
- **Serial inference pipeline** - Chain preprocessing + prediction (2-15 containers)

**Deployment Options:**
- **All-at-once** - Replace old model immediately (downtime risk)
- **Blue/green** - Deploy new, switch traffic, rollback if needed
- **Canary** - **Safety mechanism:** Route small % to new model, gradually increase if safe
  - Purpose: **Risk mitigation** - Test new model on real traffic before full rollout
  - Process: 5% → monitor → 25% → monitor → 50% → monitor → 100%
  - Goal: Ensure new model doesn't break production
- **A/B testing** - **Comparison mechanism:** Split traffic to compare model performance
  - Purpose: **Decide which model is better** - Statistical comparison of variants
  - Process: 50/50 split → collect metrics → choose winner
  - Goal: Determine which model performs better (accuracy, latency, business metrics)

**When to use:** `#exam-tip`
- Real-time user requests (web, mobile apps)
- Low latency required (< 100ms)
- Unpredictable request patterns with auto-scaling
- Interactive applications

### Batch Transform (Offline Inference)
**Purpose:** Process large datasets offline

**Characteristics:**
- No persistent endpoint
- Process entire dataset at once
- Results saved to S3
- Best for: Periodic predictions, large-scale scoring

**When to use:** `#exam-tip`
- Large datasets (millions of records)
- Predictions not needed immediately
- Run on schedule (daily, weekly)
- Cost-sensitive (no always-on endpoint)
- Examples: Daily customer churn scores, monthly credit risk assessment

### SageMaker Serverless Inference `#exam-tip`
**Purpose:** Auto-scaling inference without managing instances

**⚠️ CRITICAL LIMITATIONS:** `#exam-tip`
- **Max payload size: 4MB** - Any request > 4MB will fail
- **Max timeout: 60 seconds** - Any inference > 60s will timeout
- **Cold start latency:** First request after idle period is slower (seconds delay)

**Characteristics:**
- Scales to zero when not in use (no idle cost)
- Pay only for compute time used
- Auto-scales based on traffic

**When to use:** `#exam-tip`
- **Intermittent traffic** (hours/days between requests)
- **Unpredictable usage** with long idle periods
- **Small payloads** (< 4MB) and fast inference (< 60s)
- Development/testing environments
- Cost optimization for low-volume workloads

**When NOT to use:** `#exam-tip`
- High-throughput, consistent traffic (use Endpoint instead)
- Large payloads > 4MB (use Endpoint or Batch Transform)
- Long inference times > 60s (use Endpoint or Batch Transform)
- Latency-sensitive applications that can't tolerate cold starts

**Pricing:** Only pay for inference duration + memory allocated (no idle cost)

### Deployment Comparison Table `#exam-tip`

| Factor | Endpoint | Batch Transform | Serverless |
|--------|----------|----------------|------------|
| **Latency** | Low (ms) | High (minutes/hours) | Medium (cold start adds seconds) |
| **Cost** | $$$ Always running | $ Only during job | $ Only during requests |
| **Use case** | Real-time API | Large datasets | Intermittent traffic |
| **Scaling** | Manual/Auto | N/A (single job) | Automatic (to zero) |
| **Idle cost** | $$$ (full cost) | $0 | $0 |
| **Traffic pattern** | Continuous/predictable | Scheduled batch | Sporadic/unpredictable |
| **Max payload** | 25MB | Unlimited (S3-based) | **4MB (hard limit)** |
| **Max timeout** | No limit | No limit | **60s (hard limit)** |
| **Best for** | Production APIs | Offline scoring | Dev/test, low-volume |

**Decision Framework:** `#exam-tip`

**Choose Endpoint when:**
- Need real-time responses (< 100ms latency)
- Continuous traffic or predictable patterns
- Can justify always-on cost
- Need auto-scaling for variable load

**Choose Batch Transform when:**
- Processing millions of records
- Predictions not needed immediately
- Can run on schedule (daily, weekly)
- Most cost-effective for large batches

**Choose Serverless when:**
- Intermittent traffic (hours/days between requests)
- Can tolerate cold start latency (seconds)
- **Payloads < 4MB AND inference < 60s**
- Don't want to pay for idle time
- Development/testing environments

## Model Registry & Versioning `#important`

### SageMaker Model Registry
**Purpose:** Catalog and manage model versions

**Features:**
- **Model versioning** - Track all model versions
- **Model approval workflow** - Pending → Approved → Rejected
- **Metadata tracking** - Training metrics, lineage, artifacts
- **Model lineage** - Trace data → training → model → endpoint
- **Integration:** CI/CD pipelines, automated deployment

**Model Package Groups:**
- Group related model versions
- Approve/reject for production
- Track model performance over time

**Approval Status:** `#exam-tip`
- **PendingManualApproval** - Awaiting review
- **Approved** - Ready for production deployment
- **Rejected** - Not suitable for production

## CI/CD for ML `#exam-tip`

### SageMaker Pipelines
**Purpose:** Orchestrate end-to-end ML workflows

**Components:**
- **Pipeline steps:**
  - Processing (data prep, feature engineering)
  - Training (model training)
  - Tuning (hyperparameter optimization)
  - Model evaluation
  - Conditional execution
  - Model registration
- **Parameters** - Configurable pipeline inputs
- **Caching** - Skip unchanged steps
- **Execution tracking** - Monitor pipeline runs

**Use Cases:**
- Automate retraining on new data
- A/B testing workflows
- Model validation before deployment
- Reproducible ML workflows

**Integration:**
- Trigger via EventBridge, Lambda, Step Functions
- Git integration for version control
- Model Registry for deployment

### SageMaker Projects `#exam-tip`
**Purpose:** MLOps templates for common workflows

**Pre-built Templates:**
- Model building, training, deployment
- Model deployment with CI/CD
- Multi-account deployment
- Uses CloudFormation, CodePipeline, CodeBuild

**Components:**
- **Source control** - CodeCommit/GitHub
- **Build** - CodeBuild for training/deployment
- **Deploy** - CodePipeline for staging/production
- **Monitoring** - Automatic setup

## Model Monitoring `#important`

### SageMaker Model Monitor `#exam-tip`
**Purpose:** Detect model and data quality issues in production

**Monitoring Types:**

#### 1. Data Quality Monitoring
- **Detects:** Changes in input data distribution
- **Mechanism:** Compares production input features to training baseline (statistical comparison)
- **NO LABELS REQUIRED** - Only looks at input data, not predictions or outcomes
- **Immediate detection** - Identifies issues as soon as data arrives
- **Metrics:** Feature statistics, missing values, data types, distributions
- **Baseline:** Created from training data
- **Alerts:** When production data deviates from baseline (before model even makes predictions)

#### 2. Model Quality Monitoring `#important`
- **Detects:** Model prediction quality degradation
- **Requires:** Ground truth labels (actual outcomes) - **these arrive LATER**
- **Timing mechanism:** Model makes prediction now → Actual outcome known later (hours/days/weeks)
- **Example:**
  - Fraud detection: Predict fraud today → Investigate → Confirm fraud result 3 days later
  - Customer churn: Predict churn in March → Wait 30 days → Know if customer actually churned in April
- **Metrics:** Accuracy, precision, recall, AUC (calculated after ground truth arrives)
- **Use case:** Detect model drift over time (requires patience for ground truth)
- **Key difference from Data Quality:** Data Quality = immediate (no labels needed), Model Quality = delayed (labels required)

#### 3. Bias Drift Monitoring
- **Detects:** Changes in bias metrics over time
- **Integration:** SageMaker Clarify
- **Monitors:** Disparate impact, demographic parity

#### 4. Feature Attribution Drift
- **Detects:** Changes in feature importance
- **Uses:** SHAP values
- **Indicates:** Model behavior changes

### Monitoring Types Comparison `#exam-tip`

| Aspect | Data Quality | Model Quality |
|--------|-------------|---------------|
| **Labels required?** | ❌ No | ✅ Yes (ground truth) |
| **Detection speed** | ⚡ Immediate | ⏳ Delayed (wait for labels) |
| **What it monitors** | Input features only | Prediction accuracy |
| **When to use** | Always (first line of defense) | When ground truth is available |
| **Example alert** | "Age distribution has shifted" | "Accuracy dropped from 95% to 85%" |
| **Cost** | Lower (no labels needed) | Higher (need label collection) |

**Decision Framework:** `#exam-tip`
- **If the question mentions "without labels"** → Data Quality Monitoring
- **If the question asks about "model accuracy degradation"** → Model Quality Monitoring (needs labels)
- **If immediate detection needed** → Data Quality Monitoring
- **Best practice:** Use BOTH - Data Quality alerts you immediately, Model Quality confirms impact

**How It Works:**
1. Create baseline from training data
2. Schedule monitoring jobs (hourly, daily)
3. Compare production data to baseline
4. Generate violations report
5. Alert via CloudWatch/SNS if drift detected

**Exam Scenarios:** `#exam-tip`
- Production data differs from training → **Data Quality Monitor**
- Model accuracy dropping → **Model Quality Monitor** (needs ground truth)
- Check for bias in predictions → **Bias Drift Monitor** (Clarify)

## Inference Optimization `#exam-tip`

### Model Optimization Techniques

#### SageMaker Neo
**Purpose:** Optimize models for edge devices and cloud inference

**Features:**
- Compile models for target hardware
- 2x faster inference, 1/10th memory
- Supports: TensorFlow, PyTorch, MXNet, XGBoost
- Deploy to: Cloud, edge (Greengrass), IoT devices

**Use case:** Deploy models to edge devices, reduce inference cost

#### Elastic Inference
**Purpose:** Attach GPU acceleration to CPU instances

**Features:**
- Fractional GPU (1-8 GB GPU memory)
- Lower cost than full GPU instance
- Good for: TensorFlow, PyTorch, MXNet models

**Use case:** Need some GPU acceleration but not full GPU

#### Multi-Model Endpoints `#exam-tip`
**Purpose:** Host multiple models on single endpoint

**Benefits:**
- Cost savings (share infrastructure)
- Good for: Large number of models, similar resource needs
- Models loaded dynamically (on-demand)
- Max 1000s of models per endpoint

**Use case:** Personalized models per customer, A/B testing many variants

**Trade-off:** Model loading latency (first request to each model)

### Auto-Scaling `#exam-tip`

**Target Tracking Scaling:**
- Scale based on metric (invocations per instance)
- Define target value (e.g., 1000 requests/instance)
- SageMaker adjusts instance count automatically

**Best Practices:**
- Set appropriate min/max instances
- Consider warm-up time for new instances
- Monitor scaling metrics in CloudWatch

## Model Retraining Strategies `#exam-tip`

### When to Retrain

**Triggers:**
- **Scheduled** - Weekly, monthly (calendar-based)
- **Performance-based** - Accuracy drops below threshold
- **Data drift** - Input distribution changes significantly
- **New data available** - Regular data updates

### Retraining Approaches

#### 1. Full Retraining
- Train from scratch on all data
- **Pros:** Fresh start, no concept drift
- **Cons:** Expensive, time-consuming

#### 2. Incremental Training
- Continue training from previous model checkpoint
- **Pros:** Faster, cheaper
- **Cons:** May not adapt to major distribution shifts
- **SageMaker:** Pass previous model as input

#### 3. Online Learning
- Update model continuously with new data
- **Pros:** Always current
- **Cons:** Complex infrastructure, drift risk

**Exam Tip:** Most scenarios use scheduled full retraining (weekly/monthly)

## Infrastructure as Code `#exam-tip`

### AWS CloudFormation
- Define ML infrastructure as code
- Templates for: Endpoints, pipelines, monitoring
- Version control, repeatable deployments

### AWS CDK (Cloud Development Kit)
- Define infrastructure using programming languages
- Higher-level abstractions than CloudFormation
- Good for complex ML workflows

### SageMaker Projects
- Pre-built CloudFormation templates
- Include CI/CD pipelines
- Multi-account deployment

## Security & Compliance `#important`

### Network Isolation
- **VPC mode** - Training and inference in VPC
- **Private subnets** - No internet access
- **VPC endpoints** - Access S3, SageMaker privately
- **Security groups** - Control traffic

### Encryption
- **At rest:** S3 (training data), EBS (training volumes), model artifacts
- **In transit:** TLS for API calls, inter-node communication
- **KMS integration** - Customer-managed keys

### Access Control
- **IAM roles** - Training job roles, endpoint execution roles
- **Resource policies** - Fine-grained permissions
- **Service Control Policies (SCPs)** - Organization-wide guardrails

### Compliance
- **Audit logging** - CloudTrail for all API calls
- **HIPAA eligible** - For healthcare data
- **PCI compliant** - For payment data
- **Model governance** - Model Registry approval workflows

## Monitoring & Observability `#exam-tip`

### CloudWatch Integration
- **Metrics:**
  - Endpoint invocations
  - Model latency (ModelLatency)
  - Instance utilization (CPU, memory, GPU)
  - Error rates (4xx, 5xx)
- **Alarms:** Trigger on threshold violations
- **Dashboards:** Visualize model performance

### CloudWatch Logs
- Training job logs
- Endpoint invocation logs
- Processing job logs
- **Retention:** Configurable (7 days to indefinite)

### AWS X-Ray
- Trace requests through inference pipeline
- Identify bottlenecks
- Debug performance issues

## Cost Optimization `#exam-tip`

### Training Cost Optimization
1. **Managed Spot Training** - 90% savings
2. **Right-size instances** - Don't over-provision
3. **Local mode** - Test on notebook instance first
4. **Checkpoint frequently** - For spot interruptions
5. **Pipe input mode** - Stream data vs downloading

### Inference Cost Optimization
1. **Batch Transform** - For offline predictions
2. **Serverless Inference** - For intermittent traffic
3. **Multi-model endpoints** - Share infrastructure
4. **Auto-scaling** - Scale down during low traffic
5. **Neo optimization** - Reduce instance size needed
6. **Reserved capacity** - For predictable workloads

### Storage Cost Optimization
1. **S3 Intelligent-Tiering** - Auto-move to cheaper tiers
2. **Lifecycle policies** - Delete old model artifacts
3. **Model compression** - Reduce artifact size

## Exam Scenarios Summary `#exam-tip`

| Scenario | Solution |
|----------|----------|
| Real-time API with auto-scaling | SageMaker Endpoint + Auto-scaling |
| Process 10M records monthly | Batch Transform (scheduled) |
| Traffic: 10 requests/day | Serverless Inference |
| Deploy 1000 customer models | Multi-model Endpoint |
| Detect production data drift | Model Monitor (Data Quality) |
| Model accuracy degrading | Model Monitor (Model Quality) + Retrain |
| A/B test 2 model versions | Endpoint with Production Variants |
| Optimize model for edge device | SageMaker Neo |
| Automate ML workflow | SageMaker Pipelines |
| Version and approve models | Model Registry |
| Deploy to dev/staging/prod | SageMaker Projects + CodePipeline |
| Secure training (sensitive data) | VPC mode + Encryption |
| Cost-effective training | Managed Spot Training |
| Monitor for bias in production | Model Monitor + Clarify (Bias Drift) |

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
- [SageMaker Clarify](./sagemaker-clarify.md)
- [Data Services](./data-services.md)
