# AWS Security: Encryption

**Tags:** `#core` `#important` `#exam-tip`

**Overview:**
Comprehensive guide to encryption for AWS ML workloads covering encryption at rest, encryption in transit, AWS KMS, and AWS Secrets Manager. Essential for securing training data, model artifacts, and credentials in SageMaker and other AWS ML services.

## Encryption `#important`

### Encryption 101 `#exam-tip`

**Types of Encryption:**

#### 1. Encryption at Rest
**What:** Data stored on disk (S3, EBS, databases)
**Why:** Protect against physical theft, unauthorized access
**AWS Default:** Most services offer encryption at rest

#### 2. Encryption in Transit
**What:** Data moving over network
**Why:** Protect against eavesdropping, man-in-the-middle attacks
**Method:** TLS/SSL (HTTPS)

**Key Concepts:**

**Symmetric Encryption:**
- Same key for encryption and decryption
- **Fast** - Good for large data
- **Example:** AES-256
- **Use:** Encrypt S3 objects, EBS volumes

**Asymmetric Encryption:**
- Public key (encrypt) + Private key (decrypt)
- **Slower** - Good for small data, key exchange
- **Example:** RSA
- **Use:** TLS handshakes, digital signatures

**Envelope Encryption:** `#exam-tip`
**How KMS works:**
1. AWS KMS uses **master key** (Customer Master Key - CMK)
2. CMK encrypts **data key** (envelope key)
3. Data key encrypts your actual data
4. Store encrypted data + encrypted data key together
5. To decrypt: Use CMK to decrypt data key, use data key to decrypt data

**Why envelope encryption:**
- Don't send large data to KMS (network overhead)
- Encrypt data locally with data key
- Only send small data key to KMS for encryption

### SageMaker Encryption `#important` `#exam-tip`

#### Encryption at Rest

**What Gets Encrypted:**

| Resource | Encryption Method | Key Options |
|----------|------------------|-------------|
| **S3 (Training Data)** | S3 Server-Side Encryption | SSE-S3 (AWS-managed), SSE-KMS (customer-managed), SSE-C (customer-provided) |
| **EBS Volumes (Training)** | EBS Encryption | AWS-managed or KMS customer-managed |
| **Model Artifacts (S3)** | S3 Server-Side Encryption | SSE-KMS (default for SageMaker) |
| **SageMaker Studio EFS** | EFS Encryption | KMS customer-managed |
| **Endpoints (EBS)** | EBS Encryption | KMS customer-managed |

**How to Enable:** `#exam-tip`

**Training Job with KMS:**
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='my-algorithm',
    role=role,
    instance_type='ml.m5.xlarge',
    # Encrypt training volume
    volume_kms_key='arn:aws:kms:region:account:key/key-id',
    # Encrypt model artifacts in S3
    output_kms_key='arn:aws:kms:region:account:key/key-id'
)
```

**S3 Bucket with KMS:**
```python
import boto3

s3 = boto3.client('s3')
s3.put_object(
    Bucket='my-training-data',
    Key='data.csv',
    Body=data,
    ServerSideEncryption='aws:kms',
    SSEKMSKeyId='arn:aws:kms:region:account:key/key-id'
)
```

#### Encryption in Transit `#exam-tip`

**What Gets Encrypted:**

| Communication | Encryption Method |
|---------------|------------------|
| **API Calls to SageMaker** | TLS 1.2+ (HTTPS) |
| **S3 Data Transfer** | TLS (HTTPS endpoints) |
| **Inter-node Training** | TLS (distributed training) |
| **Endpoint Invocations** | TLS (HTTPS only) |
| **SageMaker Studio** | TLS |

**Enforce TLS:** `#exam-tip`

**S3 Bucket Policy (Require HTTPS):**
```json
{
  "Effect": "Deny",
  "Principal": "*",
  "Action": "s3:*",
  "Resource": "arn:aws:s3:::my-bucket/*",
  "Condition": {
    "Bool": {
      "aws:SecureTransport": "false"
    }
  }
}
```

**SageMaker Inter-container Encryption:**
```python
estimator = Estimator(
    # ... other params
    enable_inter_container_traffic_encryption=True  # TLS for distributed training
)
```

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Encrypt training data in S3" | **SSE-KMS** with customer-managed key |
| "Encrypt model artifacts" | **SageMaker `output_kms_key`** parameter |
| "Encrypt distributed training communication" | **`enable_inter_container_traffic_encryption=True`** |
| "Comply with regulation requiring customer-managed keys" | **KMS with customer-managed CMK** (not AWS-managed) |
| "Prevent non-HTTPS access to training data" | **S3 bucket policy** requiring `aws:SecureTransport=true` |
| "Encrypt notebook instance storage" | **SageMaker Studio with KMS key for EFS** |

### AWS KMS (Key Management Service) `#important` `#exam-tip`

**Purpose:** Create and control encryption keys used to encrypt data.

**Key Concepts:**

#### Customer Master Key (CMK)
**Types:**

| CMK Type | Who Manages | Rotation | Use Case | Cost |
|----------|-------------|----------|----------|------|
| **AWS Managed** | AWS | Auto (yearly) | Easy, no management | Free |
| **Customer Managed** | You | Manual or auto (yearly) | Full control, compliance | $1/month + usage |
| **AWS Owned** | AWS (invisible to you) | AWS-controlled | S3, DynamoDB (default) | Free |

**Key Policies:** `#exam-tip`
- **Control who can use keys** (like IAM policies for keys)
- **Grant permissions** for encrypt, decrypt, generate data keys
- **Cross-account access** - Allow other AWS accounts to use keys

**Common Operations:**

```python
import boto3

kms = boto3.client('kms')

# Create customer-managed key
response = kms.create_key(
    Description='SageMaker training data encryption key',
    KeyUsage='ENCRYPT_DECRYPT'
)
key_id = response['KeyMetadata']['KeyId']

# Encrypt data (small data only, <4KB)
ciphertext = kms.encrypt(
    KeyId=key_id,
    Plaintext=b'sensitive data'
)

# Decrypt data
plaintext = kms.decrypt(
    CiphertextBlob=ciphertext['CiphertextBlob']
)

# Generate data key (for envelope encryption)
data_key = kms.generate_data_key(
    KeyId=key_id,
    KeySpec='AES_256'
)
# Returns: Plaintext data key + Encrypted data key
```

**Key Rotation:** `#exam-tip`
- **Automatic rotation** - AWS KMS rotates key material yearly
- **Old versions kept** - Can still decrypt old data
- **Transparent** - No code changes needed
- **Best practice** - Enable for all customer-managed keys

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Compliance requires customer control of keys" | **Customer-managed CMK** |
| "Automatic key rotation" | **Enable auto-rotation** on customer-managed CMK |
| "Encrypt large S3 objects" | **Envelope encryption** (generate_data_key) |
| "Allow cross-account access to encrypted data" | **KMS key policy** granting cross-account permissions |
| "Audit key usage" | **CloudTrail logs** (who used which key when) |
| "Encrypt SageMaker training volumes" | Specify **`volume_kms_key`** in training job |

### AWS Secrets Manager `#exam-tip`

**Purpose:** Store, rotate, and manage secrets (passwords, API keys, database credentials).

**Key Features:**
- **Automatic rotation** - Lambda functions rotate secrets
- **Encryption** - KMS encryption at rest
- **Versioning** - Track secret changes
- **Fine-grained access** - IAM policies control who can access secrets
- **Integration** - RDS, Redshift, DocumentDB automatic rotation

**Use Cases in ML:**

| Use Case | How Secrets Manager Helps |
|----------|--------------------------|
| **Database credentials** | SageMaker accesses RDS for training data without hardcoded passwords |
| **API keys** | Store third-party API keys for inference endpoints |
| **Model endpoints** | Store endpoint URLs and access tokens |
| **S3 access** | Store access keys for cross-account S3 access |

**Example: SageMaker Accesses Secret**
```python
import boto3
import json

# Retrieve secret
secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(SecretId='my-db-password')
secret = json.loads(response['SecretString'])

# Use in training script
db_password = secret['password']
db_connection = connect_to_db(password=db_password)
```

**Exam Tip:** `#exam-tip`
- **"Store database password for SageMaker"** → Secrets Manager (NOT hardcode in code)
- **"Rotate API keys automatically"** → Secrets Manager with rotation enabled
- **"Comply with requirement to change passwords every 90 days"** → Secrets Manager automatic rotation

## Related Topics
- [AWS Security](./security.md) - Core security principles and IAM
- [Network Security](./security-network.md) - VPC, security groups, VPC endpoints
- [Amazon SageMaker](./sagemaker.md) - ML service implementation
- [Data Services](./data-services.md) - Data pipeline security
