# AWS Machine Learning Algorithms

**Tags:** `#core` `#exam-tip` `#important`

## Overview
Deep dive into specific AWS ML algorithms beyond SageMaker built-ins.

## Random Cut Forest (RCF) `#important`

### What Does Random Cut Forest Do?

**Primary Purpose:** Anomaly detection - identifies unusual data points

**How It Works:**
1. Builds ensemble of random decision trees
2. Each tree recursively partitions data with random cuts
3. Anomalies require fewer cuts to isolate (isolated faster)
4. **Anomaly score** = inverse of average path length across trees
   - Low path length → High anomaly score → Unusual point
   - High path length → Low anomaly score → Normal point

**Key Characteristics:**
- **Unsupervised learning** - No labeled data needed
- **Works on numerical features** - Multi-dimensional data
- **Real-time capable** - Can score streaming data
- **Robust to scale** - Handles different feature scales

### Use Cases `#exam-tip`
1. **Fraud detection** - Unusual transactions
2. **Network security** - Detecting intrusions, DDoS attacks
3. **IoT sensor monitoring** - Equipment failure prediction
4. **Quality control** - Manufacturing defects
5. **Time series anomalies** - Sudden spikes or drops

### AWS Implementation

**Amazon SageMaker RCF:**
- Built-in algorithm in SageMaker
- **Input format:** RecordIO-protobuf or CSV
- **Output:** Anomaly score for each data point (higher = more anomalous)
- **Hyperparameters:**
  - `num_trees` - Number of trees in forest (default: 100)
  - `num_samples_per_tree` - Subsample size per tree
  - `feature_dim` - Number of features (auto-detected)

**Amazon Kinesis Data Analytics:**
- Has built-in RCF function for streaming anomaly detection
- SQL function: `RANDOM_CUT_FOREST()`
- Real-time scoring on streaming data

**Amazon QuickSight:**
- ML Insights uses RCF for anomaly detection in visualizations
- Automatic outlier highlighting

### Example Scenario `#exam-tip`
**Question:** You need to detect fraudulent credit card transactions in real-time from streaming data. Which AWS service and algorithm?

**Answer:**
- **Algorithm:** Random Cut Forest (anomaly detection)
- **Service:** Kinesis Data Analytics with RANDOM_CUT_FOREST() function
- **Why:** Real-time streaming, unsupervised, detects unusual patterns

### Training Details
- **Training:** Model learns normal data distribution
- **Inference:** Scores new points against learned distribution
- **No threshold needed during training** - Application sets threshold based on business rules
- **Example:** Score > 3.0 = investigate, Score > 5.0 = block transaction

## Other AWS-Specific Algorithms

### IP Insights `#exam-tip`
- **Purpose:** Detect anomalous IP addresses
- **Use case:** Account takeover, unauthorized access
- **Input:** Entity + IP address pairs
- **Output:** Anomaly score for IP-entity combination

### Object2Vec
- **Purpose:** Learn embeddings for pairs of objects
- **Use case:** Recommendations, similarity search
- **Example:** User-item, document-document similarity

### Neural Topic Model (NTM)
- **Purpose:** Topic modeling (unsupervised)
- **Use case:** Document classification, theme discovery
- **Output:** Topic distributions for documents

## Comparison Table `#exam-tip`

| Algorithm | Type | Primary Use | Supervised? |
|-----------|------|-------------|-------------|
| Random Cut Forest | Anomaly Detection | Find outliers | No |
| XGBoost | Classification/Regression | General purpose | Yes |
| Linear Learner | Classification/Regression | Linear problems | Yes |
| K-Means | Clustering | Group similar items | No |
| PCA | Dimensionality Reduction | Feature reduction | No |
| IP Insights | Anomaly Detection | IP address anomalies | No |
| DeepAR | Forecasting | Time series prediction | Yes |
| Factorization Machines | Recommendations | Sparse data | Yes |

## Exam Tips `#exam-tip`
- **Random Cut Forest:** First choice for anomaly detection questions
- **Streaming anomalies:** Kinesis Analytics + RCF
- **Batch anomalies:** SageMaker RCF
- **RCF is unsupervised:** Don't need labeled anomaly examples
- **Anomaly score interpretation:** Higher = more anomalous
- **Real-time fraud detection:** Almost always RCF

## Gotchas `#gotcha`
- RCF requires numerical features only (encode categoricals first)
- Anomaly scores are relative (no fixed threshold)
- More trees = better accuracy but slower training
- Feature scaling improves RCF performance

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [Data Services](./data-services.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
