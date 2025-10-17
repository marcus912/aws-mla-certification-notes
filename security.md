# AWS ML Security

**Tags:** `#core` `#important` `#exam-tip`

## Overview
Security and compliance considerations for machine learning workloads on AWS.

## Core Security Principles `#important`

### Principle of Least Privilege `#exam-tip`

**Definition:** Grant only the minimum permissions required to perform a task.

**Why It Matters:**
- Reduces blast radius of security breaches
- Limits accidental damage
- Meets compliance requirements
- AWS best practice for all services

**Application in ML:** `#exam-tip`

| Scenario | Least Privilege Implementation |
|----------|-------------------------------|
| **Training Job** | IAM role with access ONLY to: specific S3 buckets (training data), ECR (container images), CloudWatch Logs |
| **Data Scientist** | IAM policy allowing SageMaker Studio, read-only S3 access, NO permission to delete production endpoints |
| **Inference Endpoint** | IAM role with access ONLY to: model artifacts in S3, CloudWatch Logs, NO training permissions |
| **Lambda Function** | Permission to invoke SageMaker endpoint ONLY, NO access to S3 or other services |

**Exam Scenarios:** `#exam-tip`
- **"Data scientist accidentally deleted production model"** → Implement least privilege (read-only for non-production roles)
- **"Minimize security risk for training jobs"** → Create IAM role with specific S3 bucket access only
- **"Secure inference endpoint"** → Separate IAM role for endpoints, no training permissions

**Best Practices:**
1. **Create separate IAM roles** for training, inference, and development
2. **Use resource-based policies** to restrict access to specific S3 buckets/paths
3. **Avoid wildcard permissions** (`s3:*` is bad, `s3:GetObject` is good)
4. **Regular audits** - Review IAM policies quarterly
5. **Use IAM Access Analyzer** - Identify overly permissive policies

**Example: Training Job Role** (Good)
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject"
  ],
  "Resource": [
    "arn:aws:s3:::my-training-data/*",
    "arn:aws:s3:::my-models/*"
  ]
}
```

**Example: Too Permissive** (Bad)
```json
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*"
}
```

### Data Masking and Anonymization `#exam-tip`

**Purpose:** Protect sensitive data while preserving utility for ML training.

**Key Concepts:**

#### 1. Data Masking
**Definition:** Replace sensitive data with realistic but fake data

**Techniques:**
- **Substitution:** Replace real names with fake names
- **Shuffling:** Randomize order of values within column
- **Redaction:** Replace with asterisks or blanks (e.g., "***-**-1234")
- **Masking:** Partial masking (e.g., "joh***@email.com")

**Use Case:** Non-production environments (testing, development)

#### 2. Anonymization
**Definition:** Remove personally identifiable information (PII) permanently

**Techniques:**
- **Generalization:** "Age: 32" → "Age: 30-40"
- **Suppression:** Remove entire fields (e.g., delete names column)
- **Pseudonymization:** Replace with consistent pseudonyms (ID-12345)
- **Perturbation:** Add noise to data (differential privacy)

**Use Case:** Production data where re-identification must be impossible

#### 3. Tokenization
**Definition:** Replace sensitive data with tokens, store mapping separately

**Example:**
- Credit card: 4532-1234-5678-9010 → TOKEN-ABC123
- Mapping stored in secure vault

**Use Case:** Payment processing, need to reverse mapping later

**AWS Services for Data Protection:** `#important`

| Service | Purpose | Use Case |
|---------|---------|----------|
| **Amazon Macie** | Discover and classify PII in S3 | Find PII before training |
| **AWS Glue DataBrew** | Visual data masking (250+ transformations) | Mask data before ML training |
| **Lake Formation** | Column-level security (hide PII columns) | Control access to sensitive columns |
| **Comprehend (PII Detection)** | Detect PII in text | Identify sensitive data in documents |
| **SageMaker Processing** | Custom masking scripts | Complex anonymization logic |

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Train model without exposing customer names" | **Pseudonymization** (replace names with IDs) |
| "Find all PII in S3 data lake before training" | **Amazon Macie** |
| "Give data scientists access without showing SSNs" | **Lake Formation column-level security** |
| "Mask credit cards in training data" | **Glue DataBrew** with masking transformations |
| "Comply with GDPR (right to be forgotten)" | **Anonymization** (can't reverse) |
| "Testing environment needs realistic data" | **Data masking** (fake but realistic) |

**Best Practices:**
1. **Identify PII first** - Use Macie to discover sensitive data
2. **Determine reversibility** - Masking (reversible) vs Anonymization (permanent)
3. **Test model impact** - Ensure masking doesn't hurt model performance
4. **Audit access** - CloudTrail logs for sensitive data access
5. **Automate** - Use Glue DataBrew, SageMaker Processing for consistent masking

**K-Anonymity Concept:** `#exam-tip`
- **Definition:** Each record is indistinguishable from at least k-1 other records
- **Example:** k=5 means you can't identify someone from fewer than 5 people
- **Technique:** Generalize data until groups of k people share same attributes
- **Use Case:** Release datasets while protecting privacy

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

## IAM (Identity and Access Management) `#important` `#exam-tip`

### IAM Basics

**Core Components:**

#### 1. Users
- **Individual people** or applications
- **Long-term credentials** (password + access keys)
- **Best practice:** Don't use root user for day-to-day operations

#### 2. Groups
- **Collections of users**
- Assign permissions to groups (not individual users)
- **Example:** "DataScientists" group with SageMaker permissions

#### 3. Roles
- **Temporary credentials** for AWS services or applications
- **No long-term keys** (more secure than access keys)
- **Use for:** EC2, Lambda, SageMaker training jobs, endpoints

**Users vs Roles:** `#exam-tip`

| Aspect | Users | Roles |
|--------|-------|-------|
| **For** | People, applications | AWS services, temp access |
| **Credentials** | Long-term (access keys) | Temporary (STS tokens) |
| **Best for** | Individual identities | SageMaker jobs, EC2, Lambda |
| **Rotation** | Manual | Automatic |

#### 4. Policies
- **JSON documents** defining permissions
- **Effect:** Allow or Deny
- **Action:** What can be done (e.g., `s3:GetObject`)
- **Resource:** Which resources (e.g., specific S3 bucket)

**Policy Example:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-training-bucket/*"
    }
  ]
}
```

### IAM for SageMaker `#exam-tip`

**Common IAM Roles:**

#### 1. SageMaker Execution Role (Training)
**Who uses it:** Training jobs, processing jobs
**Needs access to:**
- S3 (training data, model artifacts)
- ECR (container images)
- CloudWatch Logs
- KMS (if encryption enabled)

**Example Policy:**
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents",
    "ecr:GetAuthorizationToken",
    "ecr:BatchGetImage"
  ],
  "Resource": [...]
}
```

#### 2. SageMaker Endpoint Role
**Who uses it:** Real-time endpoints
**Needs access to:**
- S3 (model artifacts)
- CloudWatch Logs
- ECR (inference container)

#### 3. Data Scientist Role
**Who uses it:** Human users (data scientists, ML engineers)
**Needs access to:**
- SageMaker Studio
- S3 (read training data, read/write experiments)
- Read-only access to production endpoints (NOT delete)

**Exam Scenario:** `#exam-tip`
- **"Training job can't access S3 data"** → Check IAM execution role has `s3:GetObject` permission
- **"Endpoint deployment failed"** → Check endpoint role has access to model artifacts in S3
- **"Prevent data scientist from deleting production models"** → Remove `sagemaker:DeleteModel` from user policy

### IAM MFA (Multi-Factor Authentication) `#exam-tip`

**Purpose:** Add second factor of authentication (something you have + something you know).

**MFA Options:**
- **Virtual MFA** - Google Authenticator, Authy (smartphone app)
- **Hardware MFA** - YubiKey, other FIDO devices
- **SMS** - Not recommended (security risk)

**Best Practices:**
- **Enable MFA for root user** (critical)
- **Enable MFA for privileged users** (admins, production access)
- **Enforce MFA with IAM policy** (deny actions if MFA not present)

**Enforce MFA Policy:**
```json
{
  "Effect": "Deny",
  "Action": "*",
  "Resource": "*",
  "Condition": {
    "BoolIfExists": {
      "aws:MultiFactorAuthPresent": "false"
    }
  }
}
```

**Exam Tip:**
- **"Secure privileged user access"** → Enable MFA
- **"Prevent accidental deletion by admins"** → Require MFA for destructive operations

## Network Security `#important`

### VPC (Virtual Private Cloud) `#exam-tip`

**Purpose:** Isolated network in AWS cloud (like your own data center).

**Core Components:**

#### 1. VPC
- **Private network** with CIDR block (e.g., 10.0.0.0/16)
- **Isolation** - Resources can't be accessed from internet by default
- **Default VPC** - AWS creates one automatically (internet-accessible)

#### 2. Subnets
**Types:**

| Subnet Type | Internet Access | Route | Use Case |
|-------------|----------------|-------|----------|
| **Public** | ✅ Yes (via IGW) | Routes to Internet Gateway | Web servers, bastion hosts |
| **Private** | ❌ No (or via NAT) | Routes to NAT Gateway (outbound only) | Databases, training jobs, endpoints |

**SageMaker Best Practice:** `#exam-tip`
- **Training jobs:** Private subnet (no direct internet)
- **Endpoints:** Private subnet
- **Studio:** Private subnet with VPC endpoints

#### 3. Internet Gateway (IGW)
- **Enables internet access** for public subnets
- **One per VPC**
- **Bidirectional** - Inbound and outbound traffic

#### 4. NAT Gateway `#exam-tip`
**Purpose:** Allow private subnets to access internet (outbound only), but prevent internet from accessing them (no inbound).

**Use Cases:**
- Download packages (pip, yum) during training
- Access public APIs
- Pull Docker images from internet

**Exam Scenario:** `#exam-tip`
- **"SageMaker training job in private subnet needs to download packages"** → **NAT Gateway**
- **"Training job fails: 'Could not resolve host'"** → Add NAT Gateway or VPC Endpoints

**NAT Gateway vs NAT Instance:**

| Aspect | NAT Gateway | NAT Instance |
|--------|-------------|--------------|
| **Managed** | Fully managed by AWS | You manage EC2 instance |
| **Availability** | Highly available (AZ-level) | Single point of failure |
| **Bandwidth** | Up to 45 Gbps | Depends on instance type |
| **Cost** | $$ | $ (but operational overhead) |
| **Exam** | **Preferred choice** | Legacy option |

### NACL (Network Access Control Lists) `#exam-tip`

**Purpose:** Firewall at **subnet level** (stateless).

**Characteristics:**
- **Stateless** - Must configure inbound AND outbound rules separately
- **Rule numbers** - Processed in order (lowest first)
- **Default NACL** - Allows all inbound/outbound traffic
- **Custom NACL** - Denies all by default

**Example NACL Rules:**
```
Rule #  | Type     | Protocol | Port  | Source      | Allow/Deny
100     | Inbound  | TCP      | 443   | 0.0.0.0/0   | ALLOW
200     | Inbound  | TCP      | 80    | 0.0.0.0/0   | ALLOW
*       | Inbound  | All      | All   | 0.0.0.0/0   | DENY

100     | Outbound | TCP      | 1024-65535 | 0.0.0.0/0 | ALLOW (ephemeral)
*       | Outbound | All      | All   | 0.0.0.0/0   | DENY
```

### Security Groups `#exam-tip`

**Purpose:** Firewall at **instance/resource level** (stateful).

**Characteristics:**
- **Stateful** - Return traffic automatically allowed
- **Only ALLOW rules** - Can't create deny rules (implicit deny)
- **Default** - Denies all inbound, allows all outbound
- **Applied to:** EC2, RDS, SageMaker endpoints, EFS

**Example: SageMaker Endpoint Security Group**
```
Inbound Rules:
- Type: HTTPS, Port: 443, Source: Application Security Group
(Only allow HTTPS from application layer)

Outbound Rules:
- Type: All, Destination: 0.0.0.0/0
(Allow all outbound - return traffic)
```

**NACL vs Security Groups:** `#important`

| Aspect | NACL | Security Groups |
|--------|------|----------------|
| **Level** | Subnet | Instance/Resource |
| **State** | Stateless (separate rules) | Stateful (return auto-allowed) |
| **Rules** | Allow + Deny | Allow only |
| **Processing** | In order (rule #) | All rules evaluated |
| **Default** | Allow all | Deny all inbound |
| **Use case** | Subnet-level blocking | Instance-level control |

**Exam Tip:** `#exam-tip`
- **"Block specific IP address"** → NACL (Security Groups can't deny)
- **"Control access to SageMaker endpoint"** → Security Group
- **"Stateful firewall"** → Security Group
- **"Subnet-level protection"** → NACL

### VPC Flow Logs `#exam-tip`

**Purpose:** Capture IP traffic information for VPC, subnet, or network interface.

**What's Logged:**
- Source/destination IP
- Ports
- Protocol
- Bytes/packets
- Accept/reject status

**Use Cases:**
- **Troubleshooting** - "Why can't my training job access S3?"
- **Security monitoring** - Detect unusual traffic patterns
- **Compliance** - Audit network access

**Destinations:**
- CloudWatch Logs
- S3
- Kinesis Data Firehose

**Exam Scenario:**
- **"Debug why SageMaker can't access S3"** → Enable VPC Flow Logs, check for rejected traffic

### VPC Endpoints `#important` `#exam-tip`

**Purpose:** Access AWS services **privately** without going through internet (no IGW or NAT Gateway needed).

**Types:**

#### 1. Gateway Endpoints (Free)
- **For:** S3, DynamoDB only
- **How:** Route table entry
- **Cost:** Free
- **Use:** Most common for S3 access from SageMaker

**Example Route:**
```
Destination: pl-12345 (S3 prefix list)
Target: vpce-abc123 (Gateway Endpoint)
```

#### 2. Interface Endpoints (PrivateLink) (Paid)
- **For:** All other AWS services (SageMaker, CloudWatch, KMS, etc.)
- **How:** Elastic Network Interface (ENI) in subnet
- **Cost:** $0.01/hour + data transfer
- **Use:** Access SageMaker API, CloudWatch, KMS from private subnet

**SageMaker VPC Endpoints:** `#exam-tip`

**Required for SageMaker in VPC:**
```
com.amazonaws.region.sagemaker.api          (SageMaker API)
com.amazonaws.region.sagemaker.runtime      (Invoke endpoints)
com.amazonaws.region.s3                     (S3 - Gateway Endpoint)
com.amazonaws.region.logs                   (CloudWatch Logs)
com.amazonaws.region.ecr.api                (ECR - pull images)
com.amazonaws.region.ecr.dkr                (ECR - Docker registry)
```

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "SageMaker in private subnet, no internet, access S3" | **S3 Gateway Endpoint** (free) |
| "Training job needs to log to CloudWatch, no internet" | **CloudWatch Logs Interface Endpoint** |
| "Invoke endpoint from Lambda in private subnet" | **SageMaker Runtime Interface Endpoint** |
| "Pull Docker images for training, no NAT Gateway" | **ECR Interface Endpoints** (api + dkr) |
| "Reduce NAT Gateway costs for S3 access" | **S3 Gateway Endpoint** (free, no NAT needed) |

**Gateway Endpoint vs Interface Endpoint:**

| Aspect | Gateway Endpoint | Interface Endpoint |
|--------|------------------|-------------------|
| **Services** | S3, DynamoDB only | All other AWS services |
| **Implementation** | Route table entry | ENI in subnet |
| **Cost** | Free | $0.01/hour + transfer |
| **DNS** | AWS public DNS works | Private DNS or endpoint-specific DNS |

### VPC Peering `#exam-tip`

**Purpose:** Connect two VPCs (same or different accounts/regions).

**Use Cases:**
- **Multi-account:** Dev account VPC ↔ Prod account VPC
- **Cross-region:** US-EAST VPC ↔ EU-WEST VPC
- **Shared services:** Central VPC with shared resources

**Characteristics:**
- **Not transitive** - If A↔B and B↔C, then A↔C requires separate peering
- **No overlapping CIDR** - VPCs must have different IP ranges
- **Route table updates** - Both VPCs need routes to each other

**Exam Scenario:**
- **"Share data lake across accounts"** → VPC Peering + S3 cross-account access

### AWS PrivateLink `#exam-tip`

**Purpose:** Expose services across VPCs securely (no internet, no VPC peering).

**How It Works:**
1. **Service provider** creates VPC Endpoint Service (Network Load Balancer)
2. **Service consumer** creates Interface Endpoint in their VPC
3. Traffic flows privately via AWS network

**Use Cases:**
- **SaaS providers** - Expose APIs to customers' VPCs
- **Shared services** - Central inference endpoint accessible from multiple VPCs
- **Microservices** - Internal APIs between VPCs

**PrivateLink vs VPC Peering:**

| Aspect | PrivateLink | VPC Peering |
|--------|-------------|-------------|
| **Scalability** | Many consumers | 1-to-1 connections |
| **IP overlap** | OK (uses DNS) | Not allowed |
| **Transitive** | No (but scalable) | No |
| **Use case** | Service exposure | Network connectivity |

**Exam Scenario:**
- **"Expose SageMaker endpoint to 100 customer VPCs"** → PrivateLink (not 100 VPC peerings)

### VPN & Direct Connect `#exam-tip`

#### AWS VPN
**Purpose:** Encrypted connection from on-premises to AWS over internet.

**Characteristics:**
- **Quick setup** - Minutes to hours
- **Cost:** Low ($)
- **Speed:** Up to 1.25 Gbps per tunnel
- **Use:** Quick connectivity, backup for Direct Connect

#### AWS Direct Connect
**Purpose:** Dedicated network connection from on-premises to AWS.

**Characteristics:**
- **Setup time:** Weeks to months (physical fiber)
- **Cost:** High ($$$$)
- **Speed:** 1 Gbps to 100 Gbps
- **Reliability:** Dedicated fiber (more reliable than internet)
- **Use:** Large data transfers, consistent low latency

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Transfer 10TB training data from on-prem to AWS monthly" | **Direct Connect** (large recurring transfers) |
| "Quick secure connection for testing" | **VPN** (fast setup) |
| "Backup connectivity for Direct Connect" | **VPN as failover** |
| "Transfer 50GB one-time" | **AWS DataSync over internet** (not Direct Connect) |

### SageMaker VPC Configuration `#important` `#exam-tip`

**SageMaker VPC Mode:**

**When to Use VPC Mode:**
- ✅ Sensitive data (HIPAA, PCI compliance)
- ✅ Access private resources (RDS, on-prem data)
- ✅ Network isolation requirements
- ✅ Corporate security policies

**When NOT to Use VPC Mode:**
- ❌ Public datasets from internet (harder to access)
- ❌ Prototyping (adds complexity)

**Configuration:**
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='my-algorithm',
    role=role,
    instance_type='ml.m5.xlarge',
    # VPC Configuration
    subnets=['subnet-abc123', 'subnet-def456'],  # Private subnets
    security_group_ids=['sg-xyz789']
)
```

**Requirements for VPC Mode:** `#exam-tip`

**Must have:**
1. **Private subnets** (2+ for high availability)
2. **Security group** allowing outbound traffic
3. **S3 VPC Endpoint** (Gateway Endpoint - free)
4. **SageMaker API/Runtime VPC Endpoints** (Interface Endpoints)
5. **NAT Gateway** OR **All VPC Endpoints** (for ECR, CloudWatch, KMS)

**Option A: NAT Gateway** (Simpler, costs $$)
- Training job → NAT Gateway → Internet → Pull packages, Docker images

**Option B: VPC Endpoints Only** (More complex, costs $)
- S3 Gateway Endpoint (free)
- ECR Interface Endpoints (api + dkr)
- CloudWatch Logs Interface Endpoint
- SageMaker API Interface Endpoint
- KMS Interface Endpoint (if encryption enabled)

**Exam Decision Tree:** `#exam-tip`
- **"SageMaker in VPC, minimize cost, okay with internet egress"** → NAT Gateway
- **"SageMaker in VPC, no internet access at all"** → All VPC Endpoints (no NAT)
- **"Training fails in VPC: 'Connection timeout'"** → Missing NAT Gateway or VPC Endpoints

**Common VPC Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| "Cannot resolve host" | No internet access | Add NAT Gateway or VPC Endpoints |
| "Access denied to S3" | No S3 VPC Endpoint | Add S3 Gateway Endpoint |
| "Cannot pull Docker image" | No ECR access | Add ECR VPC Endpoints or NAT Gateway |
| "Training job stuck" | Security group blocking | Allow outbound HTTPS (443) in security group |

## Security Services `#exam-tip`

### Amazon Macie `#important`

**Purpose:** Discover and protect sensitive data in S3 using machine learning.

**What Macie Does:**
- **Automatically discover PII** in S3 buckets
- **Classify data** (financial, credentials, personal)
- **Generate findings** when sensitive data found
- **Monitor access patterns** (unusual access alerts)

**PII Types Detected:**
- Names, addresses, SSN, passport numbers
- Credit card numbers, bank accounts
- Driver's license, phone numbers, emails
- IP addresses, AWS credentials

**Use Cases:**
- **Pre-training scan** - Find PII before using data for ML
- **Compliance** - GDPR, HIPAA, PCI-DSS (know where sensitive data is)
- **Data governance** - Inventory of sensitive data across S3

**How It Works:**
1. Enable Macie for AWS account
2. Select S3 buckets to scan
3. Macie runs discovery jobs (uses ML + pattern matching)
4. Review findings in Macie console
5. Remediate: mask, delete, or restrict access

**Exam Scenarios:** `#exam-tip`

| Scenario | Solution |
|----------|----------|
| "Find all PII in S3 before ML training" | **Amazon Macie** discovery job |
| "Comply with GDPR - locate personal data" | **Amazon Macie** |
| "Alert when S3 bucket becomes public with PII" | **Macie** (automatic alerts) |
| "Classify sensitivity of S3 data" | **Macie** |

### AWS WAF (Web Application Firewall) `#exam-tip`

**Purpose:** Protect web applications from common web exploits.

**Where It Works:**
- **CloudFront** (CDN)
- **Application Load Balancer** (ALB)
- **API Gateway**

**Protection Against:**
- **SQL injection** - Malicious SQL in requests
- **Cross-site scripting (XSS)** - Malicious scripts
- **DDoS attacks** - Rate limiting
- **Bot traffic** - Block bad bots
- **Geo-blocking** - Block by country

**Use Cases for ML:**
- **Protect inference API** (API Gateway + Lambda + SageMaker)
- **Rate limiting** - Prevent abuse of inference endpoints
- **Geo-restrictions** - Only allow specific countries

**Example Rules:**
- Block requests with SQL keywords in query string
- Rate limit: Max 1000 requests per IP per 5 minutes
- Block traffic from specific countries
- Allow only specific User-Agent headers

**Exam Scenario:**
- **"Protect SageMaker inference API from DDoS"** → API Gateway + AWS WAF (rate limiting)
- **"Block SQL injection on API"** → WAF with SQL injection rule

### AWS Shield `#exam-tip`

**Purpose:** DDoS (Distributed Denial of Service) protection.

**Tiers:**

| Tier | Protection Level | Cost | Use Case |
|------|-----------------|------|----------|
| **Shield Standard** | Basic DDoS (Layer 3/4) | Free (automatic) | All AWS customers |
| **Shield Advanced** | Enhanced DDoS + 24/7 DDoS Response Team | $3,000/month | High-value applications |

**Shield Standard** (Free):
- Automatic protection against common DDoS (SYN floods, UDP reflection)
- All AWS customers get this

**Shield Advanced** ($$$):
- **DDoS Response Team (DRT)** - 24/7 support during attacks
- **Cost protection** - Refund for DDoS-related scaling costs
- **Advanced detection** - Custom mitigation
- **Integration:** WAF included, CloudFront, Route 53, ALB, ELB

**Exam Scenario:**
- **"Basic DDoS protection for inference endpoint"** → Shield Standard (free, automatic)
- **"Advanced DDoS with 24/7 support"** → Shield Advanced
- **"Protect CloudFront distribution"** → Shield Standard (included)

## Security Best Practices Summary `#exam-tip`

### Data Security
✅ **Encryption at rest** - Enable for S3, EBS, EFS (use KMS)
✅ **Encryption in transit** - TLS for all communications (enforce HTTPS)
✅ **Data masking** - Anonymize PII before training (Glue DataBrew, Macie)
✅ **Key management** - Use customer-managed KMS keys for compliance

### Access Control
✅ **Least privilege** - Minimum permissions for IAM roles
✅ **Separate roles** - Training role ≠ Inference role ≠ User role
✅ **MFA for privileged users** - Admins, production access
✅ **Secrets Manager** - Never hardcode credentials

### Network Security
✅ **VPC mode** - Private subnets for sensitive workloads
✅ **VPC Endpoints** - S3 Gateway Endpoint, SageMaker Interface Endpoints
✅ **Security Groups** - Restrict inbound to necessary ports only
✅ **No public endpoints** - Keep inference endpoints private

### Monitoring & Compliance
✅ **CloudTrail** - Audit all API calls
✅ **VPC Flow Logs** - Network traffic monitoring
✅ **Macie** - Discover and protect PII in S3
✅ **CloudWatch Alarms** - Alert on suspicious activity

### SageMaker-Specific
✅ **Enable inter-container encryption** - Distributed training
✅ **VPC configuration** - Private subnets + VPC Endpoints
✅ **IAM execution roles** - Separate per use case
✅ **Model artifacts encryption** - KMS for S3 storage

## Common Exam Scenarios `#important`

| Scenario | Solution |
|----------|----------|
| **"Secure training on sensitive data"** | VPC mode + Encryption (KMS) + IAM role with least privilege |
| **"Training job can't access S3 in VPC"** | Add S3 VPC Gateway Endpoint |
| **"Find PII before training"** | Amazon Macie |
| **"Encrypt model artifacts"** | SageMaker `output_kms_key` parameter |
| **"Prevent data scientist from deleting prod models"** | IAM policy without Delete permissions |
| **"Access RDS from SageMaker training"** | VPC mode with RDS security group rules |
| **"Store database password securely"** | AWS Secrets Manager |
| **"Rotate encryption keys annually"** | Enable KMS automatic rotation |
| **"Block specific IP from accessing endpoint"** | NACL with Deny rule |
| **"Protect inference API from DDoS"** | API Gateway + WAF + Shield |
| **"Comply with HIPAA"** | VPC + Encryption + Audit logging (CloudTrail) + BAA with AWS |
| **"Multi-account data lake access"** | Lake Formation cross-account + VPC Peering |
| **"Enforce HTTPS for S3 access"** | S3 bucket policy requiring `aws:SecureTransport=true` |

## Related Topics
- [MLOps & Deployment](./mlops-deployment.md)
- [Amazon SageMaker](./sagemaker.md)
- [Data Services](./data-services.md)
- [AWS AI Services](./aws-ai-services.md)
