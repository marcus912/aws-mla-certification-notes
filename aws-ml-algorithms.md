# AWS Machine Learning Algorithms

**Tags:** `#core` `#exam-tip` `#important`

## Overview
Comprehensive guide to SageMaker built-in algorithms. For detailed hyperparameter tuning (Linear Learner, XGBoost), see [Amazon SageMaker](./sagemaker.md).

## Supervised Learning Algorithms

### Linear Learner `#exam-tip`
**Problem Type:** Regression, Binary/Multi-class Classification

**Purpose:** Linear models with built-in regularization

**Input Format:** RecordIO-protobuf, CSV

**Key Features:**
- Supports L1, L2, Elastic Net regularization
- Handles class imbalance (`balance_multiclass_weights`)
- Unique: `target_precision`, `target_recall` optimization
- Fast training on large datasets

**Use Cases:**
- Linear regression problems
- Binary classification (fraud detection, spam)
- Multi-class classification with balanced or imbalanced data

**When to use:** Simple baseline, interpretable models, fast training needed

📝 **Detailed hyperparameters:** See [SageMaker - Linear Learner](./sagemaker.md#linear-learner-hyperparameters)

---

### XGBoost `#exam-tip`
**Problem Type:** Regression, Binary/Multi-class Classification, Ranking

**Purpose:** Gradient boosted decision trees - most popular SageMaker algorithm

**Input Format:** CSV, LibSVM, Parquet

**Key Features:**
- Handles missing values automatically
- Built-in cross-validation
- Tree pruning (prevents overfitting)
- Parallel processing
- Feature importance scores

**Hyperparameters:**
- `max_depth` - Tree depth (overfitting control)
- `eta` - Learning rate
- `subsample` - Row sampling ratio
- `colsample_bytree` - Column sampling ratio
- `alpha`, `lambda` - L1, L2 regularization
- `scale_pos_weight` - Class imbalance handling

**Use Cases:**
- General-purpose classification/regression
- Kaggle competitions (very popular)
- Structured/tabular data
- When accuracy is more important than interpretability

**When to use:** Default choice for tabular data, high accuracy needed

📝 **Detailed hyperparameters:** See [SageMaker - XGBoost](./sagemaker.md#xgboost-hyperparameters)

---

### K-Nearest Neighbors (KNN) `#exam-tip`
**Problem Type:** Classification, Regression

**Purpose:** Non-parametric algorithm - predictions based on closest training examples

**Input Format:** RecordIO-protobuf, CSV

**Key Features:**
- **Index-based algorithm** (2 steps: train builds index, inference queries index)
- Training = building efficient index (not learning parameters)
- Supports dimension reduction (sign, fjlt)
- Distance metrics: L2 (Euclidean), cosine, inner product

**Hyperparameters:**
- `k` - Number of nearest neighbors (typically 3-10)
- `sample_size` - Number of data points to sample
- `predictor_type` - classifier or regressor
- `dimension_reduction_type` - Reduce dimensions before indexing

**Use Cases:**
- Recommendation systems
- Pattern recognition
- Image classification (smaller datasets)
- When decision boundaries are irregular

**When to use:** Small to medium datasets, non-linear boundaries

**Gotchas:** `#gotcha`
- Slow inference on large datasets (must search neighbors)
- Memory intensive (stores training data)
- Sensitive to feature scaling (normalize first!)

---

### Factorization Machines `#exam-tip`
**Problem Type:** Classification, Regression

**Purpose:** Designed for high-dimensional sparse data (recommendations)

**Input Format:** RecordIO-protobuf

**Key Features:**
- Captures feature interactions efficiently
- Excellent for sparse data (many zeros)
- Linear time complexity
- Works well with click-through rate (CTR) prediction

**Hyperparameters:**
- `num_factors` - Dimension of factorization (typically 64-256)
- `predictor_type` - binary_classifier or regressor

**Use Cases:** `#exam-tip`
- **Recommendation systems** (user-item interactions)
- Click prediction (ads, search results)
- Sparse feature data (one-hot encoded categories)
- Collaborative filtering

**When to use:**
- Sparse datasets with many categorical features
- Recommendation problems
- Need to capture feature interactions

**Example:** User ID (1M users) × Item ID (100K items) = very sparse matrix

---

## Computer Vision Algorithms

### Image Classification `#exam-tip`
**Problem Type:** Multi-class classification

**Purpose:** Assign single label to entire image

**Input Format:** RecordIO (Apache MXNet), Image files (JPG, PNG)

**Architecture:** ResNet CNN (18, 34, 50, 101, 152 layers)

**Key Features:**
- Transfer learning supported (use pre-trained models)
- Full training mode or transfer learning mode
- Multi-GPU training
- Automatic image augmentation

**Hyperparameters:**
- `num_classes` - Number of output classes
- `num_training_samples` - Total training images
- `use_pretrained_model` - 0 (train from scratch) or 1 (transfer learning)
- `learning_rate`, `mini_batch_size`

**Use Cases:**
- Product categorization
- Quality control (defect/no defect)
- Medical image diagnosis
- Wildlife species identification

**When to use:** Need to classify entire image into one category

---

### Object Detection `#exam-tip`
**Problem Type:** Object localization + classification

**Purpose:** Detect multiple objects in image with bounding boxes

**Input Format:** RecordIO, Image files + JSON (annotations)

**Architecture:** Single Shot Detector (SSD)

**Key Features:**
- Detects multiple objects per image
- Returns bounding box coordinates + class labels
- Transfer learning from pre-trained models
- Supports incremental training

**Hyperparameters:**
- `num_classes` - Number of object types
- `num_training_samples` - Total training images
- `base_network` - VGG-16 or ResNet-50
- `mini_batch_size` - GPU memory dependent

**Use Cases:**
- Autonomous vehicles (detect cars, pedestrians, signs)
- Retail (shelf monitoring, inventory)
- Security (person/vehicle detection)
- Manufacturing (defect location)

**When to use:** Need to find and label multiple objects in images

**Output:** `[class, confidence, xmin, ymin, xmax, ymax]` for each object

---

### Semantic Segmentation `#exam-tip`
**Problem Type:** Pixel-level classification

**Purpose:** Classify every pixel in image (dense prediction)

**Input Format:** Image files + PNG masks (annotations)

**Architecture:** Fully Convolutional Network (FCN), Pyramid Scene Parsing (PSP)

**Key Features:**
- Pixel-wise predictions (not bounding boxes)
- Creates segmentation mask
- Transfer learning supported
- Three algorithms: FCN, PSP, DeepLabV3

**Hyperparameters:**
- `num_classes` - Number of classes (including background)
- `backbone` - ResNet-50, ResNet-101
- `algorithm` - fcn, psp, deeplab

**Use Cases:**
- Medical imaging (tumor segmentation)
- Autonomous driving (road, sidewalk, vehicle segmentation)
- Satellite imagery (land use classification)
- Background removal/replacement

**When to use:** Need precise pixel-level boundaries

**Difference from Object Detection:**
- Object Detection: Bounding boxes (rectangles)
- Semantic Segmentation: Exact shape masks (pixel-perfect)

---

## NLP & Text Algorithms

### BlazingText `#exam-tip`
**Problem Type:** Text classification, Word embeddings (Word2Vec)

**Purpose:** Fast text classification and word vector generation

**Input Format:** Text file (one sentence per line)

**Two Modes:**
1. **Word2Vec mode** - Generate word embeddings (unsupervised)
   - Output: Word vectors for semantic similarity
   - Algorithms: Skip-gram, CBOW
2. **Text classification mode** - Supervised text classification
   - Multi-class, multi-label support

**Key Features:**
- Highly optimized (GPU acceleration)
- 20x faster than traditional Word2Vec
- Supports subword embeddings (handles typos, rare words)

**Hyperparameters:**
- `mode` - supervised (classification) or batch_skipgram (Word2Vec)
- `vector_dim` - Embedding dimension (default: 100)
- `learning_rate`, `epochs`

**Use Cases:**
- Sentiment analysis
- Document classification
- Spam detection
- Generate word embeddings for downstream tasks

**When to use:**
- Need fast text classification
- Generate word vectors for transfer learning
- Large text datasets

---

### Sequence-to-Sequence (Seq2Seq) `#exam-tip`
**Problem Type:** Sequence transformation

**Purpose:** Convert input sequence to output sequence

**Input Format:** RecordIO-protobuf, JSON

**Architecture:** Encoder-Decoder with attention mechanism

**Key Features:**
- Handles variable-length inputs and outputs
- Attention mechanism (improves long sequences)
- Beam search for decoding
- Supports multiple layers (LSTM, GRU)

**Hyperparameters:**
- `num_layers_encoder` - Encoder depth
- `num_layers_decoder` - Decoder depth
- `rnn_type` - lstm or gru
- `attention_type` - mlp or dot

**Use Cases:** `#exam-tip`
- **Machine translation** (English → French)
- Text summarization
- Speech recognition (audio → text)
- Chatbots (question → answer)

**When to use:** Input and output are sequences of different lengths

**Example:**
- Input: "Hello, how are you?" (5 words)
- Output: "Bonjour, comment allez-vous ?" (4 words)

---

### Object2Vec `#exam-tip`
**Problem Type:** Embedding generation

**Purpose:** Learn low-dimensional embeddings for pairs of objects

**Input Format:** JSON (pairs of items)

**Key Features:**
- General-purpose neural embedding
- Learns relationships between objects
- Supports sentences, sequences, tokens
- Can compute similarity scores

**Hyperparameters:**
- `enc1_network`, `enc2_network` - Encoder types (pooled_embedding, hcnn, bilstm, attentional_bilstm)
- `output_layer` - mean_squared_error or softmax

**Use Cases:**
- Document similarity
- Recommendation (user-item embeddings)
- Sentence similarity
- Relationship prediction

**When to use:**
- Need to embed pairs of objects
- Measure similarity between items
- More flexible than Word2Vec (handles any object type)

**Example:** Learn embeddings where similar movies are closer together

---

## Time Series Algorithms

### DeepAR `#exam-tip`
**Problem Type:** Time series forecasting

**Purpose:** Probabilistic forecasting using RNN (produces distribution, not just point estimate)

**Input Format:** JSON Lines (one time series per line)

**Key Features:**
- Produces **probabilistic forecasts** (confidence intervals)
- Handles multiple related time series
- Learns across all time series (transfer learning effect)
- Supports missing values
- Incorporates categorical features

**Hyperparameters:**
- `epochs` - Training passes
- `context_length` - How much history to use
- `prediction_length` - How far to forecast
- `time_freq` - Data frequency (D, W, M, H)

**Use Cases:** `#exam-tip`
- **Demand forecasting** (predict future sales)
- Energy consumption prediction
- Stock price forecasting
- Server capacity planning

**When to use:**
- Need probabilistic forecasts (not just single prediction)
- Have multiple related time series
- Time series with trends and seasonality

**Output:** Quantiles (P10, P50, P90) - not just mean prediction

**Example:** "90% confident sales will be between 100-150 units"

---

## Unsupervised Learning Algorithms

### K-Means Clustering `#exam-tip`
**Problem Type:** Clustering (unsupervised)

**Purpose:** Group data into K clusters

**Input Format:** RecordIO-protobuf, CSV

**Algorithm:** Web-scale K-means (optimized for large datasets)

**Key Features:**
- Scalable (distributed training)
- Uses K-means++ initialization
- Multiple restarts to avoid local minima

**Hyperparameters:**
- `k` - Number of clusters (must specify)
- `init_method` - random or k-means++
- `mini_batch_size`

**Use Cases:**
- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection (outliers = far from clusters)

**When to use:**
- Need to group similar items
- Know (or can estimate) number of clusters
- Unsupervised grouping

**How to choose K:** `#exam-tip`
- Elbow method (plot inertia vs K)
- Silhouette analysis
- Business requirements

---

### Principal Component Analysis (PCA) `#exam-tip`
**Problem Type:** Dimensionality reduction (unsupervised)

**Purpose:** Reduce number of features while preserving variance

**Input Format:** RecordIO-protobuf, CSV

**Key Features:**
- **Unsupervised** feature extraction
- Two modes: regular (exact) and randomized (approximate, faster)
- Removes correlated features
- Speeds up training by reducing dimensions

**Hyperparameters:**
- `num_components` - Number of principal components to keep
- `algorithm_mode` - regular or randomized
- `subtract_mean` - Center data (usually true)

**Use Cases:**
- Reduce features from 1000 → 50
- Data visualization (reduce to 2-3 dimensions)
- Speed up training
- Remove multicollinearity

**When to use:** `#exam-tip`
- Too many features (curse of dimensionality)
- Features are highly correlated
- Want to visualize high-dimensional data

**Trade-off:** Lose interpretability (components are combinations of original features)

---

### Latent Dirichlet Allocation (LDA) `#exam-tip`
**Problem Type:** Topic modeling (unsupervised)

**Purpose:** Discover abstract topics in document collection

**Input Format:** RecordIO-protobuf, CSV (bag-of-words)

**Key Features:**
- **Unsupervised** document clustering
- Each document = mixture of topics
- Each topic = mixture of words
- Interpretable output

**Hyperparameters:**
- `num_topics` - Number of topics to discover
- `alpha0` - Document-topic density (higher = documents have more topics)

**Use Cases:**
- Document classification
- Content recommendation
- Research paper organization
- Customer feedback categorization

**When to use:**
- Have collection of text documents
- Want to discover themes/topics
- Need interpretable results

**Output Example:**
- Topic 1: {machine: 0.05, learning: 0.04, data: 0.03...}
- Topic 2: {aws: 0.06, cloud: 0.05, service: 0.04...}

---

### Neural Topic Model (NTM) `#exam-tip`
**Problem Type:** Topic modeling (unsupervised)

**Purpose:** Neural network approach to topic modeling (alternative to LDA)

**Input Format:** RecordIO-protobuf, CSV (bag-of-words or TF-IDF)

**Key Features:**
- Neural network architecture (faster than LDA)
- Uses variational autoencoders
- Scales better to large vocabularies
- GPU acceleration

**Hyperparameters:**
- `num_topics` - Number of topics
- `feature_dim` - Vocabulary size
- `mini_batch_size`, `learning_rate`

**Use Cases:**
- Same as LDA but for larger datasets
- When speed is important
- Large vocabulary sizes

**When to use:**
- Similar to LDA but:
  - Larger datasets
  - Need faster training
  - Have GPU resources

**LDA vs NTM:** `#exam-tip`
- LDA: Traditional, interpretable, smaller datasets
- NTM: Neural, faster, larger datasets, GPU-accelerated

---

## Anomaly Detection Algorithms

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

### IP Insights `#exam-tip`
**Problem Type:** Anomaly detection

**Purpose:** Detect anomalous IP addresses for account security

**Input Format:** CSV (entity, IP address pairs)

**Key Features:**
- Learns normal IP behavior patterns
- Neural network-based
- Unsupervised learning
- Real-time scoring

**Use Cases:**
- Account takeover detection
- Unauthorized access patterns
- Fraudulent logins
- Unusual geographic access

**When to use:**
- Need to detect abnormal IP access patterns
- Account security monitoring
- Fraud prevention for user accounts

**Example:** User always logs in from US, suddenly login from Russia → High anomaly score

---

## Algorithm Comparison Tables `#exam-tip`

### Quick Selection Guide

| Problem Type | Algorithms | Best For |
|--------------|-----------|----------|
| **Classification (Tabular)** | Linear Learner, XGBoost, KNN | XGBoost (accuracy), Linear Learner (speed) |
| **Regression (Tabular)** | Linear Learner, XGBoost, KNN | XGBoost (default choice) |
| **Time Series Forecasting** | DeepAR | Probabilistic forecasts with confidence intervals |
| **Recommendations** | Factorization Machines | Sparse user-item interactions |
| **Anomaly Detection** | Random Cut Forest, IP Insights | RCF (general), IP Insights (IP-specific) |
| **Clustering** | K-Means | Group similar items (know K) |
| **Dimensionality Reduction** | PCA | Too many features, visualization |
| **Topic Modeling** | LDA, NTM | LDA (interpretable), NTM (large/fast) |
| **Image Classification** | Image Classification | Single label per image |
| **Object Detection** | Object Detection | Multiple objects with bounding boxes |
| **Semantic Segmentation** | Semantic Segmentation | Pixel-level classification |
| **Text Classification** | BlazingText | Fast sentiment, spam detection |
| **Machine Translation** | Seq2Seq | Language translation, summarization |
| **Word Embeddings** | BlazingText, Object2Vec | BlazingText (words), Object2Vec (objects) |
| **Embeddings (Pairs)** | Object2Vec | Document/sentence similarity |

### Comprehensive Algorithm Matrix

| Algorithm | Type | Supervised? | Input Format | Key Strength |
|-----------|------|-------------|--------------|--------------|
| **Linear Learner** | Classification/Regression | Yes | CSV, RecordIO | Fast, interpretable, class imbalance handling |
| **XGBoost** | Classification/Regression | Yes | CSV, LibSVM, Parquet | Highest accuracy, handles missing values |
| **KNN** | Classification/Regression | Yes | CSV, RecordIO | Non-linear boundaries, no training |
| **Factorization Machines** | Classification/Regression | Yes | RecordIO | Sparse data, recommendations |
| **DeepAR** | Time Series | Yes | JSON Lines | Probabilistic forecasts, multiple series |
| **Image Classification** | Computer Vision | Yes | RecordIO, Images | Transfer learning, ResNet |
| **Object Detection** | Computer Vision | Yes | RecordIO, Images+JSON | Multiple objects, bounding boxes |
| **Semantic Segmentation** | Computer Vision | Yes | Images+Masks | Pixel-level precision |
| **BlazingText** | NLP | Both | Text | Fast, Word2Vec + classification |
| **Seq2Seq** | NLP | Yes | JSON, RecordIO | Variable length input/output |
| **Object2Vec** | Embeddings | Yes | JSON | General-purpose embeddings |
| **K-Means** | Clustering | No | CSV, RecordIO | Simple, fast, scalable |
| **PCA** | Dim Reduction | No | CSV, RecordIO | Feature reduction, visualization |
| **LDA** | Topic Modeling | No | RecordIO, CSV | Interpretable topics |
| **NTM** | Topic Modeling | No | RecordIO, CSV | Fast, large vocabularies |
| **Random Cut Forest** | Anomaly Detection | No | CSV, RecordIO | Real-time anomalies, streaming |
| **IP Insights** | Anomaly Detection | No | CSV | IP-specific anomalies |

---

## Note on LightGBM `#important`

**LightGBM is NOT a SageMaker built-in algorithm.**

**What is LightGBM?**
- Gradient boosting framework (similar to XGBoost)
- Developed by Microsoft
- Faster training than XGBoost on large datasets
- Lower memory usage

**How to use LightGBM on SageMaker:**
1. **Bring Your Own Container (BYOC)** - Create custom Docker image
2. **SageMaker Processing** - Use with scikit-learn
3. **Use XGBoost instead** - SageMaker's built-in alternative

**When to use LightGBM vs XGBoost:** `#exam-tip`
- **Exam answer:** Use XGBoost (it's built-in)
- **Real world:** LightGBM for very large datasets (>10M rows) with custom container
- **Performance:** Similar accuracy, LightGBM faster on large data

**Important for exam:** If asked about gradient boosting on SageMaker, answer **XGBoost**, not LightGBM.

---

## Exam Tips `#exam-tip`

### Algorithm Selection
- **Default for tabular data:** XGBoost (best accuracy)
- **Fast baseline:** Linear Learner
- **Anomaly detection:** Random Cut Forest
- **Streaming anomalies:** Kinesis Analytics + RCF
- **Time series:** DeepAR (probabilistic forecasts)
- **Recommendations:** Factorization Machines (sparse data)
- **Text classification:** BlazingText (fast)
- **Machine translation:** Seq2Seq
- **Image classification:** Image Classification (ResNet)
- **Object detection:** Object Detection (bounding boxes)
- **Semantic segmentation:** Pixel-level classification
- **Topic modeling:** LDA (small datasets), NTM (large datasets)
- **Clustering:** K-Means
- **Dimensionality reduction:** PCA
- **IP anomalies:** IP Insights

### Common Exam Scenarios
- **"Detect fraud in real-time streaming"** → Kinesis Analytics + RCF
- **"Classify images"** → Image Classification
- **"Find objects in images with locations"** → Object Detection
- **"Pixel-perfect segmentation"** → Semantic Segmentation
- **"Predict future sales with confidence intervals"** → DeepAR
- **"Recommend products for users"** → Factorization Machines
- **"Translate text"** → Seq2Seq
- **"Fast sentiment analysis"** → BlazingText
- **"Too many features (1000+)"** → PCA first, then model
- **"Unsupervised anomaly detection"** → Random Cut Forest
- **"Account takeover detection"** → IP Insights
- **"Gradient boosting on SageMaker"** → XGBoost (NOT LightGBM)

### Input Format Quick Reference
- **RecordIO-protobuf:** Most SageMaker algorithms (efficient)
- **CSV:** Linear Learner, XGBoost, KNN, K-Means, PCA, RCF
- **JSON/JSON Lines:** DeepAR, Seq2Seq, Object2Vec
- **Images:** Computer vision algorithms
- **Text files:** BlazingText
- **LibSVM:** XGBoost (sparse data)
- **Parquet:** XGBoost

## Gotchas `#gotcha`

### General
- **LightGBM:** Not built-in, requires custom container (use XGBoost instead)
- **Feature scaling:** Important for KNN, PCA, RCF (normalize features)
- **Missing values:** XGBoost handles automatically, others need imputation
- **Categorical features:** Encode as numbers (one-hot or label encoding)

### Algorithm-Specific
- **RCF:**
  - Anomaly scores are relative (no fixed threshold)
  - Requires numerical features only
  - More trees = better accuracy but slower
- **KNN:**
  - Slow inference on large datasets (searches all neighbors)
  - Memory intensive (stores all training data)
  - MUST normalize features
- **DeepAR:**
  - Needs multiple related time series (doesn't work well with single series)
  - Requires context_length of historical data
- **Factorization Machines:**
  - Only RecordIO-protobuf format (not CSV)
  - Best for very sparse data
- **Image Classification:**
  - Transfer learning requires use_pretrained_model=1
  - Need sufficient training data (or use transfer learning)
- **Seq2Seq:**
  - Input/output sequences can be different lengths
  - Requires vocabulary files
- **PCA:**
  - Components lose interpretability (linear combinations)
  - Can't invert transformation perfectly

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [Data Services](./data-services.md)
- [Model Training & Evaluation](./model-training-evaluation.md)
