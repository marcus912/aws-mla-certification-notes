# AWS Data Services for ML

**Tags:** `#core` `#important` `#hands-on`

## Overview
AWS services for data storage, processing, and preparation for machine learning.

## Data Storage

### Amazon S3
- **Primary storage** for ML training data
- **Use cases:** Raw data, processed data, model artifacts
- **Best practices:**
  - Use S3 prefixes for organization
  - Enable versioning for datasets
  - Use S3 Intelligent-Tiering for cost optimization
  - Partition data for efficient querying (Athena)

### Amazon EFS (Elastic File System)
- Shared file system for ML training
- **Use case:** Multiple instances need access to same data
- **SageMaker integration:** Attach EFS to training jobs

### Amazon FSx for Lustre
- High-performance file system
- **Use case:** HPC workloads, large-scale training
- **Speed:** Sub-millisecond latencies
- **S3 integration:** Automatically sync with S3

## Data Processing & ETL

### AWS Glue `#important`

**Overview:** Serverless ETL service

**Components:**
- **Glue Crawler** - Auto-discovers schema, populates Data Catalog
- **Glue Data Catalog** - Centralized metadata repository
- **Glue ETL Jobs** - Apache Spark-based transformations
- **Glue DataBrew** - No-code visual data preparation (250+ transformations)

**What is "Presto under the hood"?** `#exam-tip`
- **Context:** AWS Athena uses **Presto** (now called **Trino**) as its query engine
- **Presto/Trino:** Open-source distributed SQL query engine
- **"Under the hood"** means: The underlying technology powering the service
- **What it means for you:**
  - Athena queries use Presto SQL syntax
  - Fast interactive queries on S3 data
  - Supports complex SQL operations (joins, aggregations, window functions)
  - Can query multiple data sources (S3, databases, data lakes)
- **Note:** Glue ETL uses **Spark**, not Presto
- **Athena vs Glue:** Athena (Presto) = ad-hoc queries; Glue (Spark) = batch ETL jobs

**Use Cases:**
- Discover and catalog data for SageMaker
- Transform data before training
- Create training/validation/test splits
- Feature engineering at scale

**Integration with ML:**
- SageMaker can read from Glue Data Catalog
- Data Wrangler can import from Glue

### Amazon Athena
- **Purpose:** Interactive SQL queries on S3 data
- **Serverless:** Pay per query (data scanned)
- **Use cases:**
  - Exploratory data analysis
  - Data validation before training
  - Query training results
- **Best practices:**
  - Partition data to reduce scan costs
  - Use Parquet/ORC formats (columnar, compressed)
  - Create views for common queries

### Amazon EMR (Elastic MapReduce) `#exam-tip`
- **Purpose:** Managed Hadoop/Spark clusters
- **Use cases:**
  - Large-scale data processing
  - Complex ETL workflows
  - Feature engineering on big data
- **Frameworks:** Spark, Hadoop, Hive, Presto, Flink
- **ML integration:**
  - Process data, store in S3, train with SageMaker
  - Spark MLlib for distributed ML
  - SageMaker Spark library for direct integration

### AWS Glue DataBrew
- No-code data preparation
- Visual interface with 250+ transformations
- Profile data quality
- Generate data quality reports
- **Output:** Cleaned data to S3 for ML training

## Data Streaming

### Amazon Kinesis `#important`

**Kinesis Data Streams:**
- Real-time data streaming
- **Use case:** Ingest real-time data for online learning
- **SageMaker integration:** Real-time inference with streaming data

**Kinesis Data Firehose:**
- Load streaming data to S3, Redshift, Elasticsearch
- **Use case:** Batch training data collection
- **Transformations:** Lambda functions for preprocessing

**Kinesis Data Analytics:**
- SQL queries on streaming data
- **ML features:**
  - **RANDOM_CUT_FOREST()** - Real-time anomaly detection
  - Aggregations for feature engineering
- **Use case:** Real-time anomaly detection, streaming feature computation

**Kinesis Video Streams:**
- Streaming video data
- **ML integration:** SageMaker Ground Truth for video labeling

### Amazon Managed Streaming for Apache Kafka (MSK)
- Managed Kafka clusters
- **Use case:** Event streaming, log aggregation
- Alternative to Kinesis for Kafka users

## Data Warehousing

### Amazon Redshift
- **Purpose:** Data warehouse for analytics
- **Use cases:**
  - Aggregate historical data for ML features
  - Store processed features
  - Batch feature computation
- **ML integration:**
  - Redshift ML - Train models using SQL (SageMaker Autopilot under the hood)
  - SageMaker can read from Redshift
  - Data Wrangler integration

### Amazon Redshift Spectrum
- Query S3 data directly from Redshift
- **Use case:** Join warehouse data with S3 data lakes

## Data Lakes `#important`

### Data Lake Concepts `#exam-tip`

**What is a Data Lake?**
- Centralized repository for storing structured, semi-structured, and unstructured data at any scale
- Store raw data in native format until needed
- Schema-on-read (define structure when querying, not when storing)
- Foundation: Amazon S3

**Data Lake vs Data Warehouse:**

| Aspect | Data Lake | Data Warehouse |
|--------|-----------|----------------|
| **Data Type** | Raw, unstructured, semi-structured, structured | Structured, processed |
| **Schema** | Schema-on-read (flexible) | Schema-on-write (rigid) |
| **Users** | Data scientists, ML engineers | Business analysts |
| **Processing** | ELT (Extract-Load-Transform) | ETL (Extract-Transform-Load) |
| **Cost** | Lower (S3 storage) | Higher (Redshift compute) |
| **Use Case** | ML, big data analytics, exploration | Business intelligence, reporting |
| **AWS Service** | S3 + Lake Formation | Amazon Redshift |

**When to use Data Lake:** `#exam-tip`
- Need to store raw data for future unknown uses
- Machine learning on diverse data types (images, logs, JSON)
- Big data analytics
- Cost-effective storage for large volumes

**When to use Data Warehouse:**
- Structured business reporting
- SQL queries on processed data
- Known query patterns
- Fast aggregations on large datasets

---

### AWS Lake Formation `#exam-tip`

**Purpose:** Build, secure, and manage data lakes on S3

**Key Features:**

#### 1. Simplified Data Ingestion
- **Blueprints** for common data sources:
  - JDBC sources (databases)
  - S3 sources
  - Streaming data (Kinesis)
- **Workflows:** Automated ETL jobs to load data into lake
- **Incremental loading:** Only load new/changed data

#### 2. Centralized Security & Governance
- **Lake Formation permissions** - Fine-grained access control
  - Table-level permissions
  - **Column-level security** (hide sensitive columns)
  - **Row-level security** (filter rows based on user)
- **Replaces complex IAM policies** with simpler grant/revoke model
- **Cross-account access** - Share data across AWS accounts

#### 3. Data Catalog (Glue Data Catalog)
- **Shared metadata repository**
- Schemas, tables, partitions discovered by Glue Crawlers
- Used by Athena, EMR, Redshift Spectrum, SageMaker

#### 4. LF-Tags (Attribute-Based Access Control) `#exam-tip`
**Purpose:** Tag-based permissions instead of resource-based

**How it works:**
1. Create LF-Tags (e.g., `Department=Marketing`, `Sensitivity=High`)
2. Assign tags to databases, tables, columns
3. Grant permissions based on tags (e.g., "Marketing team can access all `Department=Marketing` data")

**Benefits:**
- **Scalable:** Don't need to grant permissions for each new table
- **Organize ML data:** Tag by project, team, sensitivity level
- **Dynamic permissions:** New tables with matching tags automatically inherit permissions

**Example Exam Scenario:**
> "An advertising company uses Lake Formation to manage a data lake. ML engineers need access to their campaign data. How to implement?"
>
> **Answer:** Use LF-Tags to tag tables with `Campaign=CampaignID`, grant ML engineers access to their campaign tag.

#### 5. ML Integration
- **SageMaker integration:**
  - SageMaker can query Lake Formation secured data
  - Use Lake Formation permissions instead of IAM
  - Notebook instances respect Lake Formation security
- **Athena integration:** Query lake with column/row-level security
- **EMR integration:** Spark jobs respect Lake Formation permissions

---

### Lake Formation Architecture `#exam-tip`

**Typical Workflow:**
1. **Ingest:** Load data from sources (JDBC, S3, Kinesis) → S3 data lake
2. **Catalog:** Glue Crawlers discover schema → Glue Data Catalog
3. **Secure:** Apply Lake Formation permissions (column/row-level)
4. **Tag:** Apply LF-Tags for organized access control
5. **Query/Analyze:**
   - Athena for SQL queries
   - EMR for Spark processing
   - SageMaker for ML training
6. **Permissions enforced** automatically across all services

**Components:**
- **S3** - Physical storage
- **Glue Data Catalog** - Metadata layer
- **Lake Formation** - Security & governance layer
- **Query services** - Athena, EMR, Redshift Spectrum, SageMaker

---

### Lake Formation Permissions Model `#exam-tip`

**Two Permission Systems:**

1. **IAM Permissions** (Traditional)
   - Control S3 bucket/object access
   - Complex policies for fine-grained access
   - Hard to manage at scale

2. **Lake Formation Permissions** (Recommended)
   - Database/table/column-level access
   - Simpler grant/revoke syntax
   - Automatically enforced by integrated services

**Key Concept:** `#exam-tip`
- When Lake Formation is enabled, **Lake Formation permissions take precedence** over IAM for data lake resources
- **IAM still needed** for S3 bucket access, but Lake Formation controls data access

**Grant Types:**
- **Select** - Read data
- **Insert** - Add data
- **Delete** - Remove data
- **Describe** - View metadata
- **Super** - Full control

**Column-Level Security Example:**
```
Database: customer_data
Table: customers
Columns: name, email, ssn (sensitive)

Grant to Data Scientists:
- Columns: name, email (YES)
- Columns: ssn (NO - hidden)
```

**Row-Level Security Example:**
```
Table: sales_data
Filter: region = 'US-WEST'

Result: Users only see rows where region='US-WEST'
```

---

### Lake Formation vs S3 + Glue `#exam-tip`

| Feature | S3 + Glue (No Lake Formation) | S3 + Glue + Lake Formation |
|---------|-------------------------------|----------------------------|
| **Storage** | S3 | S3 |
| **Metadata** | Glue Data Catalog | Glue Data Catalog |
| **Permissions** | IAM policies (complex) | Lake Formation (simple) |
| **Column-level security** | ❌ Not supported | ✅ Supported |
| **Row-level security** | ❌ Manual (views) | ✅ Built-in |
| **Tag-based permissions** | ❌ Not supported | ✅ LF-Tags |
| **Cross-account sharing** | ❌ Complex | ✅ Easy |
| **When to use** | Simple use cases, small teams | Enterprise, compliance, multi-team |

**Exam Tip:** If the question mentions **column-level or row-level security**, the answer is **Lake Formation**.

---

### Common Exam Scenarios `#exam-tip`

**Scenario 1:** "You need to give ML engineers access only to non-PII columns in customer data"
- **Answer:** AWS Lake Formation with column-level permissions

**Scenario 2:** "Data scientists should only see data for their assigned region"
- **Answer:** AWS Lake Formation with row-level security filters

**Scenario 3:** "Automatically grant permissions to new tables based on tags"
- **Answer:** AWS Lake Formation LF-Tags

**Scenario 4:** "Share data lake with another AWS account securely"
- **Answer:** AWS Lake Formation cross-account data sharing

**Scenario 5:** "SageMaker needs to query S3 data with fine-grained permissions"
- **Answer:** AWS Lake Formation integration with SageMaker (respects Lake Formation permissions)

**Scenario 6:** "Simplify permissions for 100+ tables across 10 teams"
- **Answer:** AWS Lake Formation with LF-Tags (tag-based access control)

---

### Lake Formation Best Practices `#exam-tip`

1. **Use LF-Tags** for scalable permission management
2. **Enable column-level security** for PII/sensitive data
3. **Row-level security** for multi-tenant data
4. **Cross-account sharing** instead of duplicating data
5. **Glue Crawlers** to keep catalog updated
6. **IAM + Lake Formation** together (IAM for S3 access, LF for data access)
7. **Audit with CloudTrail** - Track all Lake Formation API calls

## Data Labeling

### Amazon SageMaker Ground Truth `#exam-tip`
**Purpose:** Data labeling service with human-in-the-loop

**Key Features:**
- **Human labeling workflows** with three workforce options:
  - **Mechanical Turk** - Public crowdsourcing (default, cheapest)
  - **Private workforce** - Your employees (for confidential data)
  - **Vendor workforce** - Third-party managed teams (for domain expertise)
- **Active learning** - ML model auto-labels easy examples, humans label hard ones
- **Built-in workflows:**
  - Image: classification, object detection (bounding boxes), semantic segmentation
  - Text: classification, named entity recognition
  - Video: object tracking, activity recognition
  - 3D point clouds: object detection, semantic segmentation
- **Custom workflows** - Build your own labeling UI

**How Active Learning Works:**
1. Humans label initial small dataset
2. Ground Truth trains ML model on labeled data
3. Model auto-labels confident predictions
4. Humans review only low-confidence or complex items
5. Result: 70% cost reduction vs full manual labeling

**Workforce Selection Guide:** `#exam-tip`

| Data Type | Best Workforce |
|-----------|----------------|
| Public images (cats, dogs) | Mechanical Turk |
| Company documents (invoices) | Private workforce |
| Medical images (X-rays) | Private or Vendor (NOT MTurk) `#gotcha` |
| Product photos (internal) | Private workforce |
| Social media content | Mechanical Turk |
| Legal documents | Private workforce (confidential) |

**Cost Optimization:**
- Active learning reduces labeling by ~70%
- Consensus labeling (multiple workers) improves quality
- Start with MTurk for cost, switch to private if quality issues

**Output:**
- Labeled datasets in S3
- Manifest files (JSON Lines format)
- Ready for SageMaker training

**Exam Tips:** `#exam-tip`
- **Default workforce:** Mechanical Turk (public, cheap, fast)
- **Sensitive data:** Use private workforce (NOT MTurk)
- **Active learning:** Automatic for supported workflows, reduces cost 70%
- **Integration:** Works with A2I for human review workflows
- **Quality control:** Consensus (3-5 workers per item), annotation consolidation

## Data Format Recommendations `#exam-tip`

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| CSV | Small tabular data | Simple, readable | Large file size, slow parsing |
| Parquet | Large tabular data | Columnar, compressed, fast | Binary (not human-readable) |
| RecordIO-Protobuf | SageMaker training | Efficient, Pipe mode support | AWS-specific |
| JSON Lines | Semi-structured | Flexible schema | Larger than Parquet |
| TFRecord | TensorFlow | Optimized for TF | Framework-specific |
| Image files (JPG/PNG) | Computer vision | Standard formats | Need manifest file |

**Best Practice:** Use Parquet for large datasets, CSV for small/simple datasets

## Exam Tips `#exam-tip`
- **Glue Crawler:** Auto-discover schema, populate catalog
- **Athena:** Quick SQL queries on S3 (uses Presto)
- **EMR:** Large-scale Spark/Hadoop processing
- **Kinesis Data Analytics:** Real-time anomaly detection with RCF
- **Ground Truth:** Reduces labeling cost with active learning
- **S3:** Primary storage for all ML data
- **Parquet format:** Best for large tabular datasets
- **"Presto under the hood":** Refers to Athena's query engine
- **Lake Formation:** Use for column/row-level security, LF-Tags for scalable permissions
- **Data Lake vs Warehouse:** Lake for raw/ML data, Warehouse for structured BI

## Data Pipeline Selection

| Scenario | Service |
|----------|---------|
| Simple ETL, auto-schema discovery | AWS Glue |
| Ad-hoc SQL queries on S3 | Amazon Athena |
| Large-scale Spark processing | Amazon EMR |
| Real-time streaming analytics | Kinesis Data Analytics |
| No-code data preparation | Glue DataBrew |
| Data labeling | SageMaker Ground Truth |
| Data warehouse analytics | Amazon Redshift |
| Secure data lake | AWS Lake Formation |

## Related Topics
- [Feature Engineering](../core-ml/feature-engineering.md)
- [Amazon SageMaker](../sagemaker/sagemaker.md)
- [AWS ML Algorithms](./aws-ml-algorithms.md)
