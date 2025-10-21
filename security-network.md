# AWS Security: Network Security

**Tags:** `#core` `#important` `#exam-tip`

**Overview:**
Network security for AWS ML workloads covering VPC fundamentals, security groups, NACLs, VPC endpoints, VPC peering, PrivateLink, and SageMaker VPC configuration. Essential for deploying secure, isolated ML training jobs and inference endpoints.

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

## Related Topics
- [AWS Security](./security.md) - Core security principles and IAM
- [Encryption](./security-encryption.md) - Encryption at rest and in transit
- [Amazon SageMaker](./sagemaker.md) - ML service implementation
- [Data Services](./data-services.md) - Data pipeline security
