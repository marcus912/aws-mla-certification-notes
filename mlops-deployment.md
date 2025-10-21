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

## Experiment Tracking & Versioning `#important`

### SageMaker Experiments `#exam-tip`

**Purpose:** Track, organize, and compare machine learning experiments systematically.

**Key Concept:** Helps answer "Which hyperparameters produced the best model?" by organizing all training runs.

#### Hierarchy Structure

```
Experiment
  └── Trial (Training Run 1)
      ├── Trial Component (Training)
      ├── Trial Component (Processing)
      └── Trial Component (Evaluation)
  └── Trial (Training Run 2)
      └── ...
```

**Definitions:**
- **Experiment:** High-level container for related trials (e.g., "fraud-detection-model-v2")
- **Trial:** Single training run with specific hyperparameters (e.g., "learning-rate-0.01-run")
- **Trial Component:** Individual step within a trial (training job, processing job, transform job)

**What Gets Tracked Automatically:** `#important`
- **Hyperparameters:** All training parameters
- **Metrics:** Training/validation metrics over time (loss, accuracy, AUC)
- **Input data:** S3 paths to training/validation datasets
- **Output artifacts:** Model location, checkpoints
- **Code:** Git commit hash, entry point script
- **Instance info:** Instance type, instance count
- **Duration:** Start time, end time, training duration
- **Status:** InProgress, Completed, Failed

#### How It Works

**Automatic Tracking (Recommended):** `#exam-tip`
```python
from sagemaker.experiments import Run

# Create experiment (once)
experiment_name = "fraud-detection-experiment"

# Each training run creates a trial
with Run(
    experiment_name=experiment_name,
    run_name="lr-0.01-batch-128",
    sagemaker_session=session
) as run:
    # Training happens here
    estimator.fit(inputs)

    # Metrics automatically captured
    # Hyperparameters automatically logged
```

**SageMaker automatically creates:**
- Experiment (if doesn't exist)
- Trial for this run
- Trial components for training job
- Links all metadata

**Manual Tracking (Custom Metrics):**
```python
# Log custom metrics during training
run.log_metric(name="custom_f1_score", value=0.923)
run.log_parameter(name="feature_count", value=150)
run.log_artifact(name="confusion_matrix.png", value="s3://bucket/cm.png")
```

#### Key Features

**1. Comparison and Visualization** `#exam-tip`

**SageMaker Studio provides:**
- **Leaderboard view:** Rank all trials by metric (e.g., sort by validation:auc)
- **Parallel coordinates plot:** See hyperparameter effects visually
- **Time series charts:** Compare training curves across trials
- **Scatter plots:** Hyperparameter vs metric relationship

**Use case:** "Which learning rate gave best validation AUC?"
- Filter all trials by metric validation:auc
- Sort descending
- See top trial's hyperparameters instantly

**2. Lineage Tracking**

**Tracks complete data flow:** `#important`
- Input datasets → Processing jobs → Training jobs → Models → Endpoints
- **Forward tracking:** "Which endpoints use this dataset?"
- **Backward tracking:** "Which data created this model?"

**Compliance benefit:** Audit trail for regulatory requirements (GDPR, HIPAA)

**3. Search and Filter**

**Search capabilities:**
- Find trials by hyperparameter values
- Filter by metric thresholds (e.g., "AUC > 0.90")
- Search by date range
- Filter by status (Completed, Failed)

**Example queries:**
- "All trials with learning_rate < 0.01"
- "Trials from last week with F1 > 0.85"
- "Failed trials to debug"

**4. Integration with Other Services** `#exam-tip`

**Works with:**
- **SageMaker Training Jobs** - Automatic trial creation
- **SageMaker Processing Jobs** - Track data preprocessing
- **SageMaker Autopilot** - All AutoML trials captured
- **SageMaker Pipelines** - Link pipeline executions to experiments
- **Model Registry** - Connect best trial to registered model

#### Use Cases `#exam-tip`

**1. Hyperparameter Tuning Analysis**
- **Problem:** Ran 50 HPO jobs, need to understand which parameters matter
- **Solution:** SageMaker Experiments tracks all 50 trials
- **Benefit:** Visualize learning_rate vs accuracy, see patterns

**2. Team Collaboration**
- **Problem:** Multiple data scientists training models, need to share results
- **Solution:** All trials in shared experiment, team sees all results
- **Benefit:** Avoid duplicate work, build on best results

**3. Model Reproducibility**
- **Problem:** Model from 3 months ago, need exact configuration
- **Solution:** Experiments captured code, data, hyperparameters
- **Benefit:** Re-create exact model for compliance/debugging

**4. A/B Testing History**
- **Problem:** Testing multiple model versions in production
- **Solution:** Link production variants to experiment trials
- **Benefit:** Trace production model back to training run

**5. Audit and Compliance**
- **Problem:** Regulatory requirement to show model development process
- **Solution:** Experiments provide complete lineage and audit trail
- **Benefit:** Pass audits, demonstrate due diligence

#### Best Practices `#exam-tip`

**1. Naming Conventions**
```
Experiment: {project}-{model-type}-{version}
Example: "fraud-detection-xgboost-v2"

Trial: {hyperparams-summary}-{date}
Example: "lr-0.01-depth-5-2025-01-15"
```

**2. Metric Logging**
- **Log validation metrics** (not just training) - Use for comparison
- **Log business metrics** if relevant (e.g., cost per prediction)
- **Use consistent naming** across trials (e.g., always "val_auc", not sometimes "validation_auc")

**3. Organization**
- **One experiment per model type/problem**
- **New experiment for major changes** (new algorithm, new features)
- **Group related trials** (e.g., all hyperparameter tuning trials)

**4. Cleanup**
- **Archive old experiments** after model deployment
- **Delete failed trials** after debugging (reduce clutter)
- **Keep production model trials** indefinitely for audit

#### Experiments vs Model Registry `#exam-tip`

**Key Difference:**

| Aspect | SageMaker Experiments | Model Registry |
|--------|----------------------|----------------|
| **Purpose** | Track all training runs | Catalog production-ready models |
| **Scope** | Development phase | Deployment phase |
| **Content** | All trials (good and bad) | Only approved models |
| **When** | During experimentation | After model finalized |
| **Metrics** | Training/validation metrics | Production performance metrics |
| **Approval** | No approval workflow | Approval workflow (Pending/Approved/Rejected) |

**Workflow:** `#important`
1. **Experiments:** Run 100 trials, track all metrics
2. **Select best:** Trial #47 has best validation AUC
3. **Register:** Create model package in Model Registry from Trial #47
4. **Approve:** Model Registry approves for production
5. **Deploy:** Deploy approved model to endpoint
6. **Link back:** Can trace production model back to Trial #47 in Experiments

**Exam Scenario:** `#exam-tip`
- **"Track multiple training runs?"** → SageMaker Experiments
- **"Organize production models?"** → Model Registry
- **"Compare hyperparameter combinations?"** → SageMaker Experiments
- **"Approval workflow before deployment?"** → Model Registry

#### Pricing `#exam-tip`

**SageMaker Experiments is FREE**
- No additional charge beyond training costs
- Storage costs for metrics/metadata (minimal, fractions of a cent)
- **Exam note:** Cost-free way to improve ML workflow

#### Limitations

- **Max 50,000 trials per experiment** (very high, rarely hit)
- **Metrics logged every 5 seconds minimum** (prevent spam)
- **Artifact size limit:** 5GB per trial component

### TensorBoard `#exam-tip`

**Purpose:** Visualize training metrics, model graphs, and debugging information in real-time.

**Key Concept:** Open-source visualization tool (from TensorFlow) that provides rich, interactive dashboards for monitoring ML training.

#### What TensorBoard Provides

**Visualization Types:**
- **Scalars:** Training/validation loss, accuracy, metrics over time (line charts)
- **Histograms:** Weight and bias distributions across training
- **Graphs:** Model architecture visualization
- **Images:** View input images and model outputs (for vision models)
- **Distributions:** Statistical distributions of tensors
- **Embeddings:** 3D visualization of embeddings (t-SNE, PCA)
- **Text/Audio:** Preview text and audio samples (for NLP/audio models)

**Common Uses:** `#exam-tip`
1. **Monitor training progress** - Watch loss decrease in real-time
2. **Compare training runs** - Overlay multiple experiments
3. **Debug issues** - Identify vanishing gradients, exploding weights
4. **Validate model** - Check if model is learning or overfitting
5. **Hyperparameter analysis** - Compare different configurations

#### TensorBoard with SageMaker `#important`

**Integration Methods:**

**1. Training Job Output (Recommended)** `#exam-tip`
```python
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    framework_version='2.12',
    # Enable TensorBoard logging
    tensorboard_output_config={
        'LocalPath': '/opt/ml/output/tensorboard',  # Inside container
        'S3OutputPath': 's3://bucket/tensorboard-logs'  # Upload to S3
    }
)

estimator.fit({'training': s3_data})
```

**In training script (train.py):**
```python
import tensorflow as tf

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='/opt/ml/output/tensorboard',
    histogram_freq=1,
    write_graph=True
)

model.fit(
    x_train, y_train,
    callbacks=[tensorboard_callback],
    epochs=10
)
```

**Result:** Logs automatically uploaded to S3 during/after training

**2. SageMaker Studio Integration** `#exam-tip`
- **View TensorBoard directly in Studio** - No separate installation needed
- **Launch from Studio UI** - Right-click training job → "Open TensorBoard"
- **Automatic S3 sync** - Studio reads logs from S3
- **Multi-job comparison** - Compare multiple training jobs side-by-side

**Steps:**
1. Training job writes TensorBoard logs to S3
2. Open SageMaker Studio
3. Navigate to training job
4. Click "Open TensorBoard"
5. Studio launches TensorBoard visualization

**3. SageMaker Debugger Integration** `#exam-tip`

**SageMaker Debugger + TensorBoard:**
```python
from sagemaker.debugger import TensorBoardOutputConfig

tensorboard_config = TensorBoardOutputConfig(
    s3_output_path='s3://bucket/debugger-tensorboard',
    container_local_output_path='/opt/ml/output/tensorboard'
)

estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    tensorboard_output_config=tensorboard_config,
    # Debugger automatically captures tensors
    debugger_hook_config=debugger_config
)
```

**What Debugger adds to TensorBoard:**
- **Tensor distributions** - Weights, gradients, activations
- **System metrics** - CPU, GPU, memory utilization
- **Debugging insights** - Vanishing gradients, overfitting detection

#### Key Features for Exam `#exam-tip`

**1. Real-time Monitoring**
- **Watch training live** - See metrics update every few seconds
- **Catch issues early** - Stop training if loss explodes
- **No code changes** - Works with existing TensorFlow/PyTorch code

**2. Multi-run Comparison** `#important`
- **Overlay multiple runs** - Compare learning rates visually
- **Color-coded lines** - Each run gets unique color
- **Filter by run** - Show/hide specific experiments
- **Use case:** "Which learning rate converges fastest?"

**3. Histogram Analysis**
- **Weight distributions** - Check if weights updating properly
- **Gradient flow** - Ensure gradients not vanishing/exploding
- **Activation patterns** - Verify neurons firing correctly

**4. Framework Support**
- **TensorFlow:** Native support (built-in)
- **PyTorch:** Via `torch.utils.tensorboard`
- **MXNet:** Via MXBoard
- **XGBoost:** Via custom logging

#### TensorBoard vs SageMaker Experiments `#exam-tip`

**Key Differences:**

| Aspect | TensorBoard | SageMaker Experiments |
|--------|-------------|----------------------|
| **Primary purpose** | Visualize training details | Track trials and organize experiments |
| **Granularity** | Step-level (every batch) | Epoch/trial-level (aggregated) |
| **Best for** | Debugging, real-time monitoring | Comparing hyperparameters, organizing runs |
| **Visualization** | Rich graphs, histograms, embeddings | Tables, leaderboards, basic plots |
| **Storage** | Local files, S3 | SageMaker metadata store |
| **Integration** | Framework-specific (TensorFlow, PyTorch) | Framework-agnostic (works with any) |
| **Use during** | Training (real-time) | After training (analysis) |
| **Typical user** | ML engineer debugging model | Data scientist comparing trials |

**When to use what:** `#important`

**Use TensorBoard when:**
- Need detailed training visualization (loss curves, gradients)
- Debugging training issues (vanishing gradients, poor convergence)
- Want real-time monitoring during training
- Analyzing model architecture and tensor distributions
- Working with TensorFlow/PyTorch

**Use SageMaker Experiments when:**
- Comparing many hyperparameter combinations
- Organizing team experiments
- Need structured trial tracking
- Want searchable experiment catalog
- Building audit trail for compliance

**Use BOTH together:** `#exam-tip`
- **TensorBoard** for detailed debugging and real-time monitoring
- **SageMaker Experiments** for high-level trial organization and comparison
- **Workflow:**
  1. Run training with both enabled
  2. Watch TensorBoard during training (catch issues early)
  3. After training, use Experiments to compare with other trials
  4. Select best trial, register in Model Registry

#### Use Cases `#exam-tip`

**1. Debugging Training Issues**
- **Problem:** Loss not decreasing, suspect vanishing gradients
- **Solution:** TensorBoard histogram view shows gradients → near zero
- **Action:** Increase learning rate or change activation function

**2. Comparing Learning Rates**
- **Problem:** Which learning rate (0.001, 0.01, 0.1) works best?
- **Solution:** Run 3 training jobs, all log to TensorBoard
- **Benefit:** Overlay 3 loss curves, see 0.01 converges fastest

**3. Real-time Training Monitoring**
- **Problem:** Training takes 4 hours, want to check progress
- **Solution:** Open TensorBoard in Studio mid-training
- **Benefit:** See current loss, decide if training is progressing

**4. Model Architecture Validation**
- **Problem:** Complex model, want to verify architecture correct
- **Solution:** TensorBoard Graph view shows model structure
- **Benefit:** Catch errors before long training run

**5. Overfitting Detection**
- **Problem:** Model might be overfitting
- **Solution:** TensorBoard shows training vs validation loss diverging
- **Action:** Add regularization or early stopping

#### Best Practices `#exam-tip`

**1. Always Enable TensorBoard** - Minimal overhead, huge benefit
```python
# Add to all training jobs
tensorboard_output_config={
    'S3OutputPath': 's3://bucket/tensorboard-logs'
}
```

**2. Organize Logs by Experiment**
```
s3://bucket/tensorboard/
  └── fraud-detection/
      ├── lr-0.001/
      ├── lr-0.01/
      └── lr-0.1/
```

**3. Log Both Training and Validation Metrics**
```python
# Log separate scalars for train/val
tf.summary.scalar('loss/train', train_loss, step=epoch)
tf.summary.scalar('loss/validation', val_loss, step=epoch)
```

**4. Use Descriptive Run Names**
- ❌ Bad: "run1", "run2", "run3"
- ✅ Good: "lr-0.01-batch-128", "lr-0.1-batch-64"

**5. Clean Up Old Logs**
- TensorBoard logs accumulate quickly
- Use S3 lifecycle policies to archive/delete old runs
- Keep production model logs indefinitely

#### Limitations `#exam-tip`

- **Framework-specific:** Best with TensorFlow, requires extra setup for PyTorch/MXNet
- **Large files:** Detailed logging (histograms) creates large logs
- **Not a database:** Doesn't support querying like Experiments does
- **Manual comparison:** Must manually select runs to compare

#### Pricing

**TensorBoard itself is FREE** (open-source)
- **Only costs:** S3 storage for logs (minimal, usually cents)
- **Studio integration:** No extra charge
- **Data transfer:** Standard S3 rates if downloading logs

#### Exam Scenarios `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Visualize training loss in real-time?" | **TensorBoard** (with Studio integration) |
| "Debug vanishing gradients?" | **TensorBoard** (histogram view) |
| "Compare 50 hyperparameter trials?" | **SageMaker Experiments** (better for many trials) |
| "View model architecture?" | **TensorBoard** (graph view) |
| "Track which dataset produced best model?" | **SageMaker Experiments** (lineage tracking) |
| "Monitor training from Studio?" | **TensorBoard in Studio** |
| "Overlay 3 learning rate experiments?" | **TensorBoard** (multi-run comparison) |

**Key Exam Distinction:** `#important`
- **"Visualize/debug training details"** → TensorBoard
- **"Organize/compare many experiments"** → SageMaker Experiments
- **"Both together"** → Best practice for comprehensive ML workflow

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

### Data Capture `#exam-tip`
**Purpose:** Capture inference requests and responses for Model Monitor analysis

**Overview:**
- **Prerequisite for Model Monitor** - Must enable data capture on endpoint before monitoring
- Captures input payloads and model predictions in real-time
- Stores captured data in S3 for later analysis
- Required for all Model Monitor types (Data Quality, Model Quality, etc.)

**Configuration:**
```python
DataCaptureConfig = {
    'EnableCapture': True,
    'InitialSamplingPercentage': 100,  # Capture 100% of requests (or lower for high-volume)
    'DestinationS3Uri': 's3://bucket/data-capture',
    'CaptureOptions': [
        {'CaptureMode': 'Input'},   # Capture request payload
        {'CaptureMode': 'Output'}   # Capture model predictions
    ]
}
```

**Key Parameters:**
- **InitialSamplingPercentage** - % of requests to capture (1-100)
  - Use 100% for low-volume endpoints
  - Use 20-50% for high-volume to reduce storage costs
- **CaptureMode** - What to capture:
  - `Input` - Request payload sent to endpoint
  - `Output` - Model predictions/responses
  - Both - Recommended for full monitoring capability
- **DestinationS3Uri** - S3 location for captured data

**Captured Data Format:**
```json
{
  "captureData": {
    "endpointInput": {
      "observedContentType": "text/csv",
      "mode": "INPUT",
      "data": "1.5,2.3,4.1,...",
      "encoding": "CSV"
    },
    "endpointOutput": {
      "observedContentType": "text/csv",
      "mode": "OUTPUT",
      "data": "0.92",
      "encoding": "CSV"
    }
  },
  "eventMetadata": {
    "eventId": "unique-id",
    "inferenceTime": "2025-01-15T10:30:00Z"
  }
}
```

**Storage Structure in S3:**
```
s3://bucket/data-capture/
├── 2025/01/15/10/
│   ├── request-1.jsonl
│   ├── request-2.jsonl
│   └── ...
```
- Organized by year/month/day/hour
- JSONL format (one JSON object per line)

**Exam Scenarios:** `#exam-tip`
- **"Enable monitoring on endpoint"** → Must configure DataCaptureConfig first
- **"High request volume, minimize storage costs"** → Lower InitialSamplingPercentage (20-50%)
- **"Monitor both inputs and predictions"** → Capture both Input and Output modes
- **"Data for baseline creation"** → Data Capture provides the production data

**Common Gotchas:** `#gotcha`
- Cannot enable data capture on existing endpoint - must update endpoint configuration
- Captured data stored in S3 incurs storage costs
- Data capture adds minimal latency (<1ms typically)
- Must have both Input and Output capture for Model Quality monitoring

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

**📖 For comprehensive security coverage, see [Security](./security.md)**

### Quick Security Checklist

**Network Isolation:**
- **VPC mode** - Training and inference in private subnets
- **VPC endpoints** - S3 Gateway Endpoint (free), SageMaker Interface Endpoints
- **Security groups** - Restrict inbound/outbound traffic
- **NAT Gateway** - Outbound internet access (or use VPC Endpoints only)

**Encryption:**
- **At rest:** S3 (SSE-KMS), EBS (KMS), model artifacts (KMS)
- **In transit:** TLS 1.2+ for all communications
- **KMS keys** - Use customer-managed keys for compliance
- **Inter-container encryption** - Enable for distributed training

**Access Control:**
- **IAM roles** - Separate roles for training, inference, users
- **Least privilege** - Minimum permissions required
- **Secrets Manager** - Store credentials, API keys
- **MFA** - Enable for privileged users

**Monitoring & Audit:**
- **CloudTrail** - Log all API calls
- **VPC Flow Logs** - Network traffic monitoring
- **CloudWatch Logs** - Training/inference logs
- **Macie** - PII discovery in S3

**Compliance:**
- **HIPAA eligible** - VPC + Encryption + BAA
- **PCI DSS** - Encryption, audit logging
- **GDPR** - Data anonymization, right to be forgotten
- **Model governance** - Model Registry approval workflows

**See [Security](./security.md) for details on:**
- IAM policies, roles, MFA
- VPC configuration (subnets, endpoints, peering, PrivateLink)
- KMS encryption, key rotation
- Data masking and anonymization
- Macie, WAF, Shield
- Complete security best practices

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
- [Security](./security.md) - Comprehensive security coverage
- [Amazon SageMaker](./sagemaker.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
- [SageMaker Clarify](./sagemaker-clarify.md)
- [Data Services](./data-services.md)
