# MLOps: Experiment Tracking & Versioning

**Tags:** `#core` `#important` `#exam-tip`

## Overview
Experiment tracking helps ML teams organize, compare, and reproduce training runs systematically.

## SageMaker Experiments `#exam-tip`

**Purpose:** Track, organize, and compare machine learning experiments systematically.

**Key Concept:** Helps answer "Which hyperparameters produced the best model?" by organizing all training runs.

### Hierarchy Structure

```
Experiment
  └── Trial (Training Run 1)
      ├── Trial Component (Training)
      ├── Trial Component (Processing)
      └── Trial Component (Evaluation)
  └── Trial (Training Run 2)
      └── ...
```

### SageMaker Experiments Workflow

```mermaid
graph TB
    subgraph "Experiment Setup"
        A[Create Experiment<br/>'fraud-detection-v2'] --> B[Define Search Space<br/>Learning Rate: 0.001-0.1<br/>Batch Size: 32-256]
    end

    subgraph "Trial Execution - Multiple Runs"
        B --> C1[Trial 1: lr=0.001, batch=32]
        B --> C2[Trial 2: lr=0.01, batch=64]
        B --> C3[Trial 3: lr=0.1, batch=128]
        B --> C4[Trial N: ...]

        C1 --> D1[Training Job 1]
        C2 --> D2[Training Job 2]
        C3 --> D3[Training Job 3]
        C4 --> D4[Training Job N]

        D1 --> E1[Auto-Capture:<br/>- Hyperparameters<br/>- Metrics<br/>- Artifacts<br/>- Duration]
        D2 --> E2[Auto-Capture:<br/>- Hyperparameters<br/>- Metrics<br/>- Artifacts<br/>- Duration]
        D3 --> E3[Auto-Capture:<br/>- Hyperparameters<br/>- Metrics<br/>- Artifacts<br/>- Duration]
        D4 --> E4[Auto-Capture:<br/>- Hyperparameters<br/>- Metrics<br/>- Artifacts<br/>- Duration]
    end

    subgraph "Analysis in SageMaker Studio"
        E1 --> F[Leaderboard View]
        E2 --> F
        E3 --> F
        E4 --> F

        F --> G[Sort by Validation AUC<br/>Trial 2: AUC=0.94 ⭐<br/>Trial 3: AUC=0.91<br/>Trial 1: AUC=0.87<br/>Trial N: ...]

        G --> H[Visualization Tools]
        H --> I[Parallel Coordinates<br/>See hyperparameter impact]
        H --> J[Time Series Charts<br/>Compare training curves]
        H --> K[Scatter Plots<br/>LR vs AUC relationship]
    end

    subgraph "Best Model Selection & Deployment"
        G --> L[Select Trial 2<br/>Best AUC=0.94]
        L --> M[View Complete Metadata:<br/>- Code version: commit abc123<br/>- Data: s3://bucket/train.csv<br/>- Instance: ml.p3.2xlarge<br/>- Duration: 45 min]

        M --> N[Register in Model Registry]
        N --> O[Model Package v1.0<br/>Links back to Trial 2]

        O --> P[Deploy to Production]
        P --> Q[Complete Lineage:<br/>Production Model → Trial 2 → Training Data]
    end

    style C2 fill:#d4f4dd
    style E2 fill:#fff4e1
    style L fill:#e1ffe1
    style Q fill:#e1f5ff
```

### Experiments vs TensorBoard vs Model Registry

```mermaid
graph LR
    subgraph "During Training"
        A[Training Job] --> B[TensorBoard<br/>Real-time Visualization]
        B --> C[Monitor:<br/>- Loss curves<br/>- Gradients<br/>- Weights<br/>Step-level detail]

        A --> D[SageMaker Experiments<br/>Trial Tracking]
        D --> E[Capture:<br/>- Hyperparameters<br/>- Final metrics<br/>- Artifacts<br/>Trial-level summary]
    end

    subgraph "After Training"
        E --> F[Compare 100 Trials<br/>Leaderboard]
        F --> G[Best Trial<br/>AUC=0.94]

        G --> H[Model Registry<br/>Production Catalog]
        H --> I[Register Model v1.0<br/>Status: Approved]

        I --> J[Deploy to Endpoint]

        J --> K[Lineage:<br/>Endpoint → Model → Trial → Data]
    end

    subgraph "Use Cases"
        C --> L[Debug Training:<br/>Vanishing gradients?<br/>Overfitting?]

        F --> M[Hyperparameter Analysis:<br/>Which LR is best?<br/>Compare 50 trials]

        K --> N[Compliance:<br/>Reproduce model<br/>Audit trail]
    end

    style B fill:#fff4e1
    style D fill:#e1f5ff
    style H fill:#e1ffe1
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

### How It Works

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

### Key Features

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

### Use Cases `#exam-tip`

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

### Best Practices `#exam-tip`

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

### Experiments vs Model Registry `#exam-tip`

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

### Pricing `#exam-tip`

**SageMaker Experiments is FREE**
- No additional charge beyond training costs
- Storage costs for metrics/metadata (minimal, fractions of a cent)
- **Exam note:** Cost-free way to improve ML workflow

### Limitations

- **Max 50,000 trials per experiment** (very high, rarely hit)
- **Metrics logged every 5 seconds minimum** (prevent spam)
- **Artifact size limit:** 5GB per trial component

## TensorBoard `#exam-tip`

**Purpose:** Visualize training metrics, model graphs, and debugging information in real-time.

**Key Concept:** Open-source visualization tool (from TensorFlow) that provides rich, interactive dashboards for monitoring ML training.

### What TensorBoard Provides

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

### TensorBoard with SageMaker `#important`

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

### Key Features for Exam `#exam-tip`

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

### TensorBoard vs SageMaker Experiments `#exam-tip`

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

### Use Cases `#exam-tip`

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

### Best Practices `#exam-tip`

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

### Limitations `#exam-tip`

- **Framework-specific:** Best with TensorFlow, requires extra setup for PyTorch/MXNet
- **Large files:** Detailed logging (histograms) creates large logs
- **Not a database:** Doesn't support querying like Experiments does
- **Manual comparison:** Must manually select runs to compare

### Pricing

**TensorBoard itself is FREE** (open-source)
- **Only costs:** S3 storage for logs (minimal, usually cents)
- **Studio integration:** No extra charge
- **Data transfer:** Standard S3 rates if downloading logs

### Exam Scenarios `#exam-tip`

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

## Related Topics
- [MLOps & Deployment](./mlops-deployment.md) - Deployment strategies, inference optimization
- [MLOps CI/CD](./mlops-cicd.md) - Model Registry, Pipelines, Kubernetes
- [MLOps Monitoring](./mlops-monitoring.md) - Model Monitor, observability
- [Amazon SageMaker](./sagemaker.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
