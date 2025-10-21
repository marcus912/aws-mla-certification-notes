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

**See dedicated guide:** [Encryption](./security-encryption.md)

**Topics covered:**
- Encryption at rest and in transit
- SageMaker encryption (training volumes, model artifacts, inter-container)
- AWS KMS (Customer Master Keys, key rotation, policies)
- AWS Secrets Manager (credential management, rotation)

**Quick Reference:**

| Scenario | Solution |
|----------|----------|
| "Encrypt training data in S3" | **SSE-KMS** with customer-managed key |
| "Encrypt distributed training communication" | **`enable_inter_container_traffic_encryption=True`** |
| "Store database password for SageMaker" | **AWS Secrets Manager** |
| "Comply with regulation requiring customer-managed keys" | **KMS with customer-managed CMK** |

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

**See dedicated guide:** [Network Security](./security-network.md)

**Topics covered:**
- VPC fundamentals (subnets, IGW, NAT Gateway)
- Security Groups and NACLs
- VPC Endpoints (Gateway and Interface)
- VPC Peering and PrivateLink
- VPN and Direct Connect
- SageMaker VPC Configuration

**Quick Reference:**

| Scenario | Solution |
|----------|----------|
| "SageMaker in private subnet, access S3" | **S3 Gateway Endpoint** (free) |
| "Training job needs to download packages in VPC" | **NAT Gateway** or **VPC Endpoints** |
| "Block specific IP address" | **NACL** (Security Groups can't deny) |
| "Invoke endpoint from Lambda in private subnet" | **SageMaker Runtime Interface Endpoint** |

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
- [Encryption](./security-encryption.md) - Encryption at rest, in transit, KMS, Secrets Manager
- [Network Security](./security-network.md) - VPC, security groups, endpoints, SageMaker VPC config
- [MLOps & Deployment](./mlops-deployment.md) - Secure deployment patterns
- [Amazon SageMaker](./sagemaker.md) - ML service security features
- [Data Services](./data-services.md) - Data pipeline security
- [AWS AI Services](./aws-ai-services.md) - AI service security
