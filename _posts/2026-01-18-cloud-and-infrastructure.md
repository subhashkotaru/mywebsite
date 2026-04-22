---
title: "Cloud & Infrastructure: Fundamentals to Production"
date: 2026-01-18
description: "Cloud computing from first principles — virtualisation, networking, storage, and IAM — through practical AWS and GCP operations, API gateways, Nginx, load balancing, observability, and the pitfalls that actually bite you in production."
tags: [cloud, aws, gcp, infrastructure, devops, networking, nginx]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#fundamentals">Cloud Fundamentals</a>
      <ul class="post-toc-sublist">
        <li><a href="#why-cloud">Why Cloud Exists</a></li>
        <li><a href="#virtualisation">Virtualisation & Containers</a></li>
        <li><a href="#networking">Networking Primitives</a></li>
        <li><a href="#storage-tiers">Storage Tiers</a></li>
        <li><a href="#iam">Identity & Access Management</a></li>
        <li><a href="#service-models">IaaS vs PaaS vs SaaS vs Serverless</a></li>
        <li><a href="#regions-az">Regions, AZs, and Edge</a></li>
      </ul>
    </li>
    <li><a href="#aws">AWS in Practice</a>
      <ul class="post-toc-sublist">
        <li><a href="#aws-compute">Compute: EC2, ECS, EKS, Lambda</a></li>
        <li><a href="#aws-storage">Storage: S3, EBS, EFS</a></li>
        <li><a href="#aws-networking">Networking: VPC, Subnets, Security Groups</a></li>
        <li><a href="#aws-databases">Databases: RDS, DynamoDB, ElastiCache</a></li>
        <li><a href="#aws-iam-practice">IAM in Practice</a></li>
        <li><a href="#aws-ops">Day-to-Day Operations</a></li>
        <li><a href="#aws-pitfalls">AWS Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#gcp">GCP in Practice</a>
      <ul class="post-toc-sublist">
        <li><a href="#gcp-compute">Compute: GCE, GKE, Cloud Run, Cloud Functions</a></li>
        <li><a href="#gcp-storage">Storage: GCS, Persistent Disk, Filestore</a></li>
        <li><a href="#gcp-networking">Networking: VPC, Firewall Rules, Cloud NAT</a></li>
        <li><a href="#gcp-ml">ML-Specific: Vertex AI, TPUs, Cloud GPUs</a></li>
        <li><a href="#gcp-ops">Day-to-Day Operations</a></li>
        <li><a href="#gcp-pitfalls">GCP Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#api-gateways">API Gateways & Reverse Proxies</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-is-gateway">What an API Gateway Does</a></li>
        <li><a href="#nginx">Nginx: How It Works</a></li>
        <li><a href="#nginx-config">Nginx Configuration in Practice</a></li>
        <li><a href="#load-balancing">Load Balancing Algorithms</a></li>
        <li><a href="#managed-gateways">Managed Gateways: AWS ALB, Kong, Envoy</a></li>
        <li><a href="#gateway-pitfalls">API Gateway Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#observability">Observability</a>
      <ul class="post-toc-sublist">
        <li><a href="#three-pillars">Logs, Metrics, Traces</a></li>
        <li><a href="#practical-obs">Practical Observability Stack</a></li>
      </ul>
    </li>
    <li><a href="#cost">Cost Management</a></li>
    <li><a href="#production-patterns">Production Patterns</a></li>
  </ul>
</nav>

---

## Cloud Fundamentals {#fundamentals}

### Why Cloud Exists {#why-cloud}

Before cloud, running software at scale meant buying servers, racking them in a data centre, managing power and cooling, paying for capacity you needed only during peak load, and waiting weeks for hardware procurement. The utilisation rate of on-premise servers averaged 15–20%.

Cloud solves this with three ideas:

1. **Resource pooling**: the provider's hardware is shared across thousands of tenants, achieving much higher utilisation. Your idle server time is another tenant's burst capacity.
2. **Elasticity**: you rent capacity by the hour or second, scaling up for traffic spikes and releasing after. You match supply to demand continuously.
3. **Managed operations**: the provider handles hardware failure, physical security, network peering, and (for higher-level services) OS patching, backups, and HA.

The trade-off: you lose control and pay a margin above raw hardware cost. For most workloads, the operational savings far outweigh the premium. For sustained, predictable, very high-volume workloads (e.g., Netflix, Cloudflare), hybrid or on-premise often wins on unit economics.

**The billing model in one picture:**

```
On-premise:
 Cost │████████████████████████  ← fixed, paid upfront
      │
      └─────────────────────────── time
         low traffic    peak

Cloud:
 Cost │      ██
      │    ████
      │  ██████  ████
      │████████████████
      └─────────────────────────── time
         tracks actual usage
```

---

### Virtualisation & Containers {#virtualisation}

#### Hardware Virtualisation

A **hypervisor** runs on bare metal and multiplexes physical CPU/memory/disk across multiple **virtual machines (VMs)**. Each VM has its own OS kernel — fully isolated.

```
┌─────────────────────────────┐
│  VM 1       │  VM 2         │
│  Guest OS   │  Guest OS     │
│  App        │  App          │
├─────────────────────────────┤
│        Hypervisor           │  (Type 1: KVM, Xen, Hyper-V)
├─────────────────────────────┤
│        Physical Hardware    │
└─────────────────────────────┘
```

AWS EC2 instances are VMs. The physical host runs many VMs; you share the host but can't see other tenants' memory. AWS uses a custom hypervisor (Nitro) that offloads virtualisation to dedicated hardware, leaving nearly all CPU cycles to the guest.

**Type 1 hypervisor** (bare-metal): KVM (Linux), Xen (AWS original), Hyper-V. Runs directly on hardware.  
**Type 2 hypervisor** (hosted): VMware Workstation, VirtualBox. Runs inside a host OS. Slower.

#### Containers

Containers share the host OS kernel — no guest OS overhead. Isolation uses Linux **namespaces** (pid, net, mnt, uts, ipc, user) and **cgroups** (CPU/memory limits).

```
┌─────────────────────────────────────┐
│  Container A  │  Container B        │
│  App + libs   │  App + libs         │
│               │                     │
├─────────────────────────────────────┤
│           Container Runtime         │  (containerd, runc)
├─────────────────────────────────────┤
│           Host OS Kernel            │
├─────────────────────────────────────┤
│           Physical Hardware         │
└─────────────────────────────────────┘
```

**VM vs Container:**

| | VM | Container |
|---|---|---|
| Startup | 30–60 seconds | <1 second |
| Memory overhead | Hundreds of MB (guest OS) | Tens of MB |
| Isolation | Strong (separate kernel) | Weaker (shared kernel) |
| Portability | Image is large (full OS) | Image is small |
| Use case | Full OS isolation, legacy apps | Microservices, fast scaling |

In practice: containers run *inside* VMs on cloud. ECS/EKS nodes are EC2 instances running container runtimes.

**Kubernetes** orchestrates containers at scale: scheduling (which node to place a pod on), self-healing (restart failed containers), service discovery, rolling updates, and autoscaling. A **pod** is the smallest unit — one or more tightly coupled containers sharing a network namespace and storage volumes.

```
Kubernetes Cluster:
┌────────────────────────────────────────────────┐
│  Control Plane                                 │
│  API Server | etcd | Scheduler | Controller    │
├────────────────────────────────────────────────┤
│  Worker Node 1        │  Worker Node 2         │
│  kubelet + kube-proxy │  kubelet + kube-proxy  │
│  Pod A  Pod B         │  Pod C  Pod D          │
└────────────────────────────────────────────────┘
```

---

### Networking Primitives {#networking}

#### IP Addressing

Every device on a network has an IP address. IPv4: 32-bit, written as four octets (e.g., `10.0.1.45`). IPv6: 128-bit, written as eight hex groups.

**CIDR notation** specifies a range of addresses: `10.0.0.0/16` means the first 16 bits are fixed (`10.0`), leaving 16 bits for hosts — $2^{16} = 65,536$ addresses. `/24` gives 256 addresses; `/32` is a single address.

| CIDR | Addresses | Typical use |
|---|---|---|
| `/8` | 16,777,216 | Large private network |
| `/16` | 65,536 | VPC (AWS default) |
| `/24` | 256 | Subnet |
| `/28` | 16 | Small subnet, NAT gateway |
| `/32` | 1 | Single host, security group rule |

**Private IP ranges** (RFC 1918) — not routable on the public internet:
- `10.0.0.0/8`
- `172.16.0.0/12`
- `192.168.0.0/16`

#### How a Packet Travels

```
Your laptop (192.168.1.5)
    │  DNS lookup: api.example.com → 54.12.34.56
    │
    ▼
Router/Gateway  →  ISP  →  Internet  →  Cloud Load Balancer (54.12.34.56)
                                              │
                                         Target Group
                                         (private IPs)
                                              │
                                         EC2/Container (10.0.1.45:8080)
```

**DNS** translates hostnames to IPs. Lookup order: local cache → `/etc/hosts` → configured DNS resolver (e.g., `8.8.8.8` or your VPC's resolver) → authoritative nameserver.

TTL on DNS records controls how long caches hold the answer. Low TTL (30s) = fast failover but more DNS queries. High TTL (300s) = efficient but slow propagation of changes.

#### TCP vs UDP

**TCP**: connection-oriented, reliable, ordered. Three-way handshake (SYN → SYN-ACK → ACK) before data. Retransmits lost packets. Head-of-line blocking. Used for HTTP, databases, SSH.

**UDP**: connectionless, no reliability guarantee, low overhead. Used for DNS, video streaming, gaming, QUIC (HTTP/3 uses UDP under the hood and reimplements reliability per-stream).

#### Ports

Well-known ports: HTTP 80, HTTPS 443, SSH 22, PostgreSQL 5432, Redis 6379, MySQL 3306. Ports 1024–65535 are ephemeral (used for outbound connections).

---

### Storage Tiers {#storage-tiers}

Cloud storage comes in four fundamentally different types:

```
Block Storage        Object Storage      File Storage      Database
(EBS, Persistent     (S3, GCS)           (EFS, Filestore)  (RDS, DynamoDB)
 Disk)
 
 ┌──────┐            ┌──────────────┐    ┌──────────┐
 │ Raw  │            │ Key → Blob   │    │ /files/  │
 │ disk │            │ any size     │    │ ├── a.py │
 │ ops  │            │ HTTP API     │    │ └── b.py │
 └──────┘            └──────────────┘    └──────────┘
 
 Mounted as          Accessed via        Mounted as        App-level
 filesystem          REST API            NFS/SMB           queries
 
 ~0.1ms latency      ~10–100ms           ~1–10ms           depends
 
 Tied to one VM      Globally            Shared across     Managed
                     accessible          many VMs
```

**Block storage** (EBS, GCP Persistent Disk): raw disk attached to one VM. Best for databases and anything needing low-latency random I/O. Cannot be shared between VMs (except EBS Multi-Attach, limited).

**Object storage** (S3, GCS): key-value store for blobs. Arbitrary size. Accessed via HTTP, not filesystem calls. Eventually consistent (now strongly consistent on S3 since 2020). Cheap at scale. The correct home for model weights, training data, logs, backups, and static assets.

**File storage** (EFS, Filestore): NFS-compatible shared filesystem. Multiple VMs can mount it simultaneously. Useful for shared config, code, or data that must be readable from many nodes at once. More expensive than object storage per GB; lower latency than S3.

**Object storage durability**: S3 guarantees 11 nines (99.999999999%) durability — achieved by automatically replicating across at least 3 AZs. If you delete something, it's gone; if AWS loses hardware, your data isn't.

---

### Identity & Access Management {#iam}

IAM answers: **who can do what to which resource.**

**Principal**: who (user, service account, role)  
**Action**: what (`s3:GetObject`, `ec2:StartInstances`)  
**Resource**: which (`arn:aws:s3:::my-bucket/*`)  
**Condition**: under what circumstances (source IP, MFA required, time of day)

**AWS IAM Policy (JSON)**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::my-model-bucket/*",
      "Condition": {
        "StringEquals": {"s3:prefix": ["models/"]}
      }
    }
  ]
}
```

**Least privilege principle**: grant only the permissions actually needed. Wildcard `"Action": "*"` or `"Resource": "*"` is almost always wrong outside of admin roles.

**IAM Roles vs Users**:
- **Users** have long-lived credentials (access key + secret). Avoid for services — keys get leaked in code.
- **Roles** are assumed temporarily; services get short-lived tokens via instance metadata. EC2, Lambda, ECS tasks, and GKE pods should all use roles, not user keys.

**Instance Metadata Service (IMDS)**: an EC2 instance can get its role credentials at `http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>`. Tokens rotate automatically. IMDS v2 requires a session token (prevents SSRF attacks from exfiltrating credentials).

---

### Service Models {#service-models}

```
You manage →  Everything    App+Data    Just Data    Nothing
              ─────────────────────────────────────────────
              On-Premise     IaaS         PaaS        SaaS

Cloud manages →  Nothing    Hardware   Infra+OS    Everything
              
Examples:      Your servers   EC2         Cloud Run   Gmail
                              GCE         Heroku      Snowflake
                              Raw VMs     App Engine  Databricks
```

**Serverless** (Lambda, Cloud Functions, Cloud Run): you provide code; the platform handles scaling, OS, runtime, and billing per invocation. Cold start latency is the main drawback — the first request after idle can take 100ms–5s.

**When to use what:**
- EC2/GCE: full OS control, stateful apps, long-running GPU training
- ECS/GKE: containerised microservices with orchestration needs
- Lambda/Cloud Functions: event-driven, short-lived, bursty workloads
- Cloud Run: containerised, stateless, HTTP services with zero-to-scale

---

### Regions, AZs, and Edge {#regions-az}

```
AWS Global Infrastructure:

  Region: us-east-1 (N. Virginia)
  ┌─────────────────────────────────────────────┐
  │  AZ: us-east-1a    AZ: us-east-1b           │
  │  ┌──────────┐      ┌──────────┐             │
  │  │ Data     │      │ Data     │             │
  │  │ Center 1 │      │ Center 2 │             │
  │  └──────────┘      └──────────┘             │
  │           \          /                       │
  │            Low-latency                       │
  │            fibre links                       │
  └─────────────────────────────────────────────┘
           │
      Regional services
      (S3, DynamoDB, ELB)
           │
      Edge Locations (~400)
      (CloudFront CDN, Route 53)
```

**Region**: geographically separate cluster of data centres. Choosing a region affects latency (choose close to users), data residency compliance (GDPR requires EU), and service availability (not all services in all regions).

**Availability Zone (AZ)**: one or more physically separate data centres within a region. Independent power, cooling, networking. Deploying across ≥2 AZs protects against single data centre failure. Most production workloads span 2–3 AZs.

**Edge locations**: CDN PoPs (Points of Presence) distributed globally. CloudFront caches content close to users; Route 53 answers DNS queries from the nearest edge.

> **Pitfall — AZ affinity**: data transfer within an AZ is free; across AZs in the same region costs ~$0.01/GB. A poorly architected service that constantly crosses AZ boundaries (e.g., app in AZ-a talking to database in AZ-b every request) can accumulate significant egress costs. Deploy RDS read replicas and cache nodes in each AZ; pin application instances to the same AZ as their data.

---

## AWS in Practice {#aws}

### Compute: EC2, ECS, EKS, Lambda {#aws-compute}

#### EC2

```bash
# Launch an instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \     # Amazon Linux 2
  --instance-type t3.medium \
  --key-name my-keypair \
  --subnet-id subnet-abc123 \
  --security-group-ids sg-xyz789 \
  --iam-instance-profile Name=my-ec2-role \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=api-server}]'

# SSH in
ssh -i ~/.ssh/my-keypair.pem ec2-user@<public-ip>

# Check instance metadata from inside
curl http://169.254.169.254/latest/meta-data/instance-type

# List running instances
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

**Instance families** (choose based on workload):

| Family | Optimised for | Examples |
|---|---|---|
| `t3/t4g` | Burstable general purpose | Dev, low-traffic APIs |
| `m6i/m7i` | Balanced (memory/CPU) | Production web, app servers |
| `c6i/c7i` | Compute (high vCPU/RAM ratio) | CPU-bound ML inference, batch |
| `r6i/r7i` | Memory (high RAM) | In-memory databases, Spark |
| `p3/p4d/p5` | GPU (Nvidia V100/A100/H100) | ML training |
| `g4dn/g5` | GPU (cheaper, inference) | ML inference, graphics |
| `inf1/inf2` | AWS Inferentia (custom ASIC) | Low-cost, high-throughput inference |
| `i3en/is4gen` | NVMe local SSD | High IOPS databases |

**Spot instances**: up to 90% cheaper than On-Demand but can be interrupted with 2-minute warning. Perfect for training jobs and batch processing. Use Spot Fleet or EC2 Auto Scaling with mixed instance types to reduce interruption probability.

**Savings Plans and Reserved Instances**: commit to 1 or 3 years of usage for 30–72% discount. Use for baseline load; use Spot for burst.

#### ECS (Elastic Container Service)

AWS-managed container orchestration. Two launch types:
- **EC2 launch type**: you manage the underlying EC2 instances (instance sizing, patching, scaling)
- **Fargate launch type**: AWS manages the instances; you specify CPU/memory per task

```bash
# Deploy a new task definition revision
aws ecs register-task-definition --cli-input-json file://task-def.json

# Update a service to use new task definition
aws ecs update-service \
  --cluster my-cluster \
  --service my-service \
  --task-definition my-task:42 \
  --desired-count 4

# Watch service events during deployment
aws ecs describe-services \
  --cluster my-cluster \
  --services my-service \
  --query 'services[0].events[:5]'

# Exec into a running container (like kubectl exec)
aws ecs execute-command \
  --cluster my-cluster \
  --task <task-id> \
  --container app \
  --interactive \
  --command "/bin/bash"
```

#### EKS (Elastic Kubernetes Service)

Managed Kubernetes control plane. You still manage worker nodes (or use Fargate for pods). 

```bash
# Update kubeconfig to point at EKS cluster
aws eks update-kubeconfig --name my-cluster --region us-east-1

# Now use kubectl normally
kubectl get nodes
kubectl get pods -n production
kubectl logs -f deploy/api-server -n production
kubectl exec -it <pod-name> -- /bin/bash

# Scale a deployment
kubectl scale deploy api-server --replicas=8 -n production

# Rolling restart (without changing image)
kubectl rollout restart deploy/api-server -n production

# Check rollout status
kubectl rollout status deploy/api-server -n production
```

#### Lambda

```python
# handler.py — Lambda function
import json
import boto3

def lambda_handler(event, context):
    # event: the triggering payload (API Gateway request, S3 event, etc.)
    # context: runtime info (function name, timeout remaining, etc.)
    
    s3 = boto3.client('s3')
    
    # Example: process S3 upload event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    
    # ... process data ...
    
    return {
        'statusCode': 200,
        'body': json.dumps({'processed': key})
    }
```

```bash
# Deploy Lambda from zip
zip function.zip handler.py requirements.txt
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip

# Invoke synchronously
aws lambda invoke \
  --function-name my-function \
  --payload '{"key": "value"}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json

# Tail Lambda logs
aws logs tail /aws/lambda/my-function --follow
```

**Lambda cold start**: when a function hasn't been invoked recently (or scales out to new instances), AWS needs to initialise the execution environment. Mitigation:
- Use Provisioned Concurrency (keeps N instances warm, costs money)
- Keep package size small (Lambda deploys faster)
- Move heavy init (model loading, DB connections) outside the handler, into module-level code — it runs once per execution environment, not per invocation
- Use Graviton (ARM) functions — faster init and cheaper

---

### Storage: S3, EBS, EFS {#aws-storage}

#### S3

```bash
# Upload a file
aws s3 cp model.pt s3://my-bucket/models/v2/model.pt

# Sync a local directory to S3 (only changed files)
aws s3 sync ./checkpoints s3://my-bucket/checkpoints/ \
  --exclude "*.tmp" \
  --storage-class INTELLIGENT_TIERING

# Download with progress
aws s3 cp s3://my-bucket/data/train.tar.gz . \
  --expected-size 10737418240   # 10GB, shows progress bar

# List with human-readable sizes
aws s3 ls s3://my-bucket/models/ --human-readable --summarize

# Presigned URL (temporary public access, no AWS credentials needed)
aws s3 presign s3://my-bucket/models/v2/model.pt --expires-in 3600

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket my-bucket \
  --versioning-configuration Status=Enabled

# Lifecycle rule: move to Glacier after 90 days, delete after 365
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-bucket \
  --lifecycle-configuration file://lifecycle.json
```

**S3 Storage Classes** (cost vs access speed):

| Class | $/GB/month | Retrieval | Use case |
|---|---|---|---|
| Standard | $0.023 | Instant | Frequently accessed |
| Standard-IA | $0.0125 | Instant | Infrequent, but needs fast |
| Intelligent-Tiering | $0.023 + $0.0025/1K | Instant | Unknown access pattern |
| Glacier Instant | $0.004 | Instant | Archives, accessed rarely |
| Glacier Flexible | $0.0036 | 1–12 hours | Long-term archives |
| Glacier Deep Archive | $0.00099 | 12–48 hours | Compliance, 7+ year retention |

**S3 performance**: prefix-based partitioning matters. S3 can handle ~3,500 PUT and ~5,500 GET requests/second per prefix. If you have 100K small files, avoid putting them all under one prefix like `s3://bucket/data/file_0001.txt` — distribute with hashed prefixes: `s3://bucket/a3/data/file_0001.txt`.

#### EBS

```bash
# Check disk usage on EC2
df -h
lsblk

# Resize EBS volume (after resizing in console/CLI)
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1    # ext4
# or: sudo xfs_growfs /      # xfs

# Monitor IOPS
aws cloudwatch get-metric-statistics \
  --namespace AWS/EBS \
  --metric-name VolumeReadOps \
  --dimensions Name=VolumeId,Value=vol-abc123 \
  --start-time 2024-01-01T00:00:00 \
  --end-time 2024-01-01T01:00:00 \
  --period 60 \
  --statistics Sum
```

**EBS Volume Types**:

| Type | Max IOPS | Max Throughput | Use case |
|---|---|---|---|
| gp3 | 16,000 | 1,000 MB/s | General purpose (default) |
| io2 Block Express | 256,000 | 4,000 MB/s | High-perf databases |
| st1 | 500 | 500 MB/s | Sequential (Kafka, Hadoop) |
| sc1 | 250 | 250 MB/s | Cold, infrequent access |

gp3 is almost always the right choice — provisioned IOPS (io2) is needed only for latency-sensitive databases under heavy write load.

---

### Networking: VPC, Subnets, Security Groups {#aws-networking}

```
VPC: 10.0.0.0/16
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Public Subnet (10.0.1.0/24)     Public Subnet (10.0.2.0/24)  │
│  AZ: us-east-1a                  AZ: us-east-1b               │
│  ┌──────────────┐                ┌──────────────┐              │
│  │ Load Balancer│                │ NAT Gateway  │              │
│  │ Bastion Host │                │              │              │
│  └──────────────┘                └──────────────┘              │
│         │                               │                      │
│  Private Subnet (10.0.3.0/24)    Private Subnet (10.0.4.0/24) │
│  AZ: us-east-1a                  AZ: us-east-1b               │
│  ┌──────────────┐                ┌──────────────┐              │
│  │ App servers  │                │ App servers  │              │
│  │ ECS tasks    │                │ ECS tasks    │              │
│  └──────────────┘                └──────────────┘              │
│         │                               │                      │
│  DB Subnet (10.0.5.0/24)         DB Subnet (10.0.6.0/24)     │
│  ┌──────────────┐                ┌──────────────┐              │
│  │ RDS Primary  │                │ RDS Replica  │              │
│  └──────────────┘                └──────────────┘              │
│                                                                 │
│  Internet Gateway ←→ Public subnets only                       │
│  NAT Gateway: private → internet (outbound only)               │
└─────────────────────────────────────────────────────────────────┘
```

**Security Groups** (stateful firewall at the instance level):
```bash
# Allow HTTPS inbound from anywhere
aws ec2 authorize-security-group-ingress \
  --group-id sg-abc123 \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Allow app servers to talk to RDS (source = another security group)
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds \
  --protocol tcp --port 5432 \
  --source-group sg-appservers

# Stateful: if you allow inbound, the return traffic is automatically allowed
# No need to add an outbound rule for response traffic
```

**NACLs vs Security Groups**:

| | Security Group | NACL |
|---|---|---|
| Level | Instance/ENI | Subnet |
| State | Stateful | Stateless |
| Rules | Allow only | Allow + Deny |
| Evaluation | All rules evaluated | Rules evaluated in order, first match wins |
| Use case | Primary firewall | Subnet-level blocklist |

**NAT Gateway**: lets private subnet instances initiate outbound internet connections (for package downloads, API calls) without exposing them to inbound traffic. Costs ~$0.045/hour + $0.045/GB processed — can add up with high egress.

---

### Databases: RDS, DynamoDB, ElastiCache {#aws-databases}

#### RDS

Managed relational database. AWS handles: hardware provisioning, OS patching, automated backups (up to 35-day retention), multi-AZ failover (automatic, ~60 second RTO), read replica creation.

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier prod-postgres \
  --db-instance-class db.r6g.large \
  --engine postgres \
  --engine-version 15.3 \
  --master-username admin \
  --master-user-password <secure-password> \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --db-subnet-group-name my-db-subnet-group \
  --vpc-security-group-ids sg-rds

# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier prod-postgres-replica \
  --source-db-instance-identifier prod-postgres

# Take manual snapshot before risky migration
aws rds create-db-snapshot \
  --db-instance-identifier prod-postgres \
  --db-snapshot-identifier pre-migration-2024-01-15
```

**Multi-AZ vs Read Replica**:
- **Multi-AZ**: synchronous standby in another AZ. Automatic failover. NOT used for reads — for HA only.
- **Read Replica**: asynchronous replication. Used to offload read traffic. Can be in another region (for disaster recovery). Can be promoted to primary if needed (manual).

**RDS Proxy**: connection pooler that sits in front of RDS. Reduces connection overhead for serverless/Lambda workloads that open many short-lived connections. Reuses database connections, improves availability during failover.

#### DynamoDB

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')

# Put item
table.put_item(Item={
    'user_id': 'u123',          # partition key
    'created_at': '2024-01-15', # sort key
    'name': 'Alice',
    'email': 'alice@example.com',
    'age': 30
})

# Get item (O(1))
response = table.get_item(Key={'user_id': 'u123', 'created_at': '2024-01-15'})
item = response.get('Item')

# Query by partition key + sort key range
response = table.query(
    KeyConditionExpression=Key('user_id').eq('u123') & Key('created_at').begins_with('2024')
)

# Conditional write (optimistic locking)
table.update_item(
    Key={'user_id': 'u123', 'created_at': '2024-01-15'},
    UpdateExpression='SET #v = #v + :inc',
    ConditionExpression=Attr('version').eq(5),
    ExpressionAttributeNames={'#v': 'version'},
    ExpressionAttributeValues={':inc': 1}
)
```

**DynamoDB key design is everything.** A bad partition key creates hot partitions:

```
BAD partition key: status ("active"/"inactive")
→ 90% of traffic hits "active" partition → throttled

GOOD partition key: user_id (high cardinality, uniform distribution)
→ traffic spread across all partitions
```

**DynamoDB capacity modes**:
- **On-Demand**: pay per request. No capacity planning. Expensive at high sustained throughput.
- **Provisioned**: set read/write capacity units. Auto-scaling adjusts. Cheaper at sustained load.

#### ElastiCache (Redis)

```python
import redis

r = redis.Redis(host='my-cluster.abc123.cache.amazonaws.com', port=6379)

# Cache-aside pattern
def get_user(user_id):
    cache_key = f"user:{user_id}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    r.setex(cache_key, 300, json.dumps(user))  # TTL: 5 minutes
    return user

# Distributed lock (prevent cache stampede)
lock_key = f"lock:rebuild:{cache_key}"
if r.set(lock_key, "1", nx=True, ex=10):   # NX: only set if not exists
    try:
        data = expensive_computation()
        r.setex(cache_key, 3600, json.dumps(data))
    finally:
        r.delete(lock_key)
```

**Cache invalidation strategies**:
- **TTL-based**: simple, but stale data for up to TTL duration
- **Write-through**: update cache on every write. Consistent but adds write latency
- **Cache-aside (lazy loading)**: populate on miss. Most common. Risk: thundering herd on cache miss
- **Event-driven**: invalidate cache when DB event fires (via CDC or application events)

---

### IAM in Practice {#aws-iam-practice}

```bash
# Create a role for EC2 instances
aws iam create-role \
  --role-name ec2-app-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach a policy
aws iam attach-role-policy \
  --role-name ec2-app-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile (needed to attach role to EC2)
aws iam create-instance-profile --instance-profile-name ec2-app-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name ec2-app-profile \
  --role-name ec2-app-role

# Check what permissions a role has (effective)
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::123456789:role/ec2-app-role \
  --action-names s3:GetObject s3:DeleteObject \
  --resource-arns arn:aws:s3:::my-bucket/*

# Find overly permissive roles
aws iam generate-service-last-accessed-details --arn arn:aws:iam::123456789:role/ec2-app-role
# Check the report to see which services were never accessed — remove those permissions
```

---

### Day-to-Day AWS Operations {#aws-ops}

```bash
# Cost explorer: top services by cost last 30 days
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-02-01 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[0].Groups[*].[Keys[0],Metrics.BlendedCost.Amount]' \
  --output table | sort -t$'\t' -k2 -rn | head -10

# CloudWatch: check CPU on EC2 last hour
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-abc123 \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# CloudWatch Logs: find errors in last 10 minutes
aws logs filter-log-events \
  --log-group-name /ecs/my-service \
  --start-time $(($(date +%s) - 600))000 \
  --filter-pattern "ERROR"

# SSM Session Manager: SSH without opening port 22
aws ssm start-session --target i-abc123

# Parameter Store: store and retrieve secrets
aws ssm put-parameter \
  --name "/prod/db/password" \
  --value "mysecretpassword" \
  --type SecureString \
  --key-id alias/aws/ssm

aws ssm get-parameter \
  --name "/prod/db/password" \
  --with-decryption \
  --query Parameter.Value \
  --output text
```

**Structured workflow for a production incident:**

```
1. aws cloudwatch describe-alarms --state-value ALARM
   → which alarms are firing?

2. aws logs filter-log-events --log-group /ecs/api --filter-pattern ERROR
   → what errors are in logs?

3. aws ecs describe-services --cluster prod --services api
   → are tasks healthy? desired vs running count?

4. aws ec2 describe-instances + check security groups/NACLs
   → is there a network issue?

5. aws rds describe-db-instances → check DB connections, storage
   → is the database overwhelmed?
```

---

### AWS Pitfalls {#aws-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **S3 requester-pays not set** | You pay egress for public dataset | Enable Requester Pays; use VPC endpoints for internal access |
| **NAT Gateway for S3** | All S3 traffic goes through NAT → ~$0.045/GB | Use VPC Gateway Endpoint for S3 (free) |
| **Security group 0.0.0.0/0 on port 22** | SSH exposed to internet → brute force | Use SSM Session Manager; remove port 22 |
| **EBS not in same AZ as EC2** | Cross-AZ EBS doesn't exist — attachment fails | EBS is AZ-specific; create in same AZ |
| **RDS publicly accessible** | Database reachable from internet | Set `PubliclyAccessible=false`; use VPC only |
| **Lambda hitting VPC database** | Lambda cold start + VPC ENI creation = 10+ second latency | Use RDS Proxy; keep Lambda warm |
| **DynamoDB hot partition** | Throttling despite adequate provisioned capacity | Fix partition key design; add random suffix |
| **Not tagging resources** | $50K bill, no idea which team owns what | Enforce tags via Service Control Policies |
| **IAM Access Keys in code** | Keys leaked to GitHub → AWS bill from crypto mining | Use IAM roles; rotate keys; enable GuardDuty |
| **CloudWatch Logs retention not set** | Logs accumulate forever → storage costs compound | Set retention policy (30–90 days) |

---

## GCP in Practice {#gcp}

### Compute: GCE, GKE, Cloud Run, Cloud Functions {#gcp-compute}

GCP's hierarchy: **Organisation → Folders → Projects → Resources**. Billing, IAM, and quotas are managed at project level. A project is the closest analogue to an AWS account.

```bash
# Configure gcloud CLI
gcloud config set project my-project-id
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Create a VM
gcloud compute instances create my-instance \
  --machine-type=n2-standard-4 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --scopes=cloud-platform \   # gives the VM all GCP API access (use specific scopes in prod)
  --service-account=my-sa@my-project.iam.gserviceaccount.com

# SSH via IAP (no public IP needed, no port 22 open)
gcloud compute ssh my-instance --tunnel-through-iap

# List VMs
gcloud compute instances list
```

**Machine types:**

| Series | Optimised for | Examples |
|---|---|---|
| N2/N2D | General purpose | n2-standard-4, n2-highcpu-8 |
| C2/C3 | Compute-optimised | c2-standard-8 |
| M2/M3 | Memory-optimised | m2-ultramem-208 |
| A2/A3 | GPU (Nvidia A100/H100) | a2-highgpu-1g |
| G2 | GPU (L4, inference) | g2-standard-4 |

**Preemptible / Spot VMs**: up to 91% cheaper, can be reclaimed with 30-second warning. Same use cases as AWS Spot.

#### GKE

```bash
# Create a GKE Autopilot cluster (fully managed nodes)
gcloud container clusters create-auto my-cluster \
  --region us-central1

# Or Standard cluster (you manage nodes)
gcloud container clusters create my-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n2-standard-4 \
  --enable-autoscaling --min-nodes 1 --max-nodes 10

# Get credentials
gcloud container clusters get-credentials my-cluster --zone us-central1-a

# Now use kubectl
kubectl get nodes
kubectl apply -f deployment.yaml

# Enable Workload Identity (no JSON key files for pods)
gcloud container clusters update my-cluster \
  --workload-pool=my-project.svc.id.goog
```

**GKE Autopilot vs Standard**: Autopilot manages node provisioning, scaling, and security automatically — you pay per pod CPU/memory rather than per node. Standard gives more control but requires node management. Autopilot is cheaper for variable workloads; Standard for GPU nodes and custom configs.

#### Cloud Run

```bash
# Deploy a container to Cloud Run
gcloud run deploy my-service \
  --image gcr.io/my-project/my-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \          # requests per container instance
  --max-instances 100

# View logs
gcloud run services logs read my-service --region us-central1 --limit 100

# Gradual traffic rollout (canary)
gcloud run services update-traffic my-service \
  --to-revisions my-service-v2=10,my-service-v1=90
```

Cloud Run scales to zero (no cost when idle) and to thousands of instances in seconds. **Concurrency** is the key Cloud Run parameter — how many requests each instance handles simultaneously. Higher concurrency = fewer instances = cheaper, but more memory pressure. Set based on your app's threading model.

---

### Storage: GCS, Persistent Disk, Filestore {#gcp-storage}

```bash
# GCS operations
gsutil cp model.pt gs://my-bucket/models/
gsutil -m cp -r ./data gs://my-bucket/data/   # parallel multi-threaded
gsutil rsync -r ./checkpoints gs://my-bucket/checkpoints/

# Signed URL
gsutil signurl -d 1h service-account-key.json gs://my-bucket/model.pt

# Bucket lifecycle: delete after 90 days
gsutil lifecycle set lifecycle.json gs://my-bucket

# IAM on bucket
gsutil iam ch serviceAccount:my-sa@my-project.iam.gserviceaccount.com:roles/storage.objectViewer gs://my-bucket

# Check bucket size
gsutil du -sh gs://my-bucket/
```

**GCS Storage Classes:**

| Class | $/GB/month | Min duration | Use case |
|---|---|---|---|
| Standard | $0.020 | None | Frequently accessed |
| Nearline | $0.010 | 30 days | Monthly access |
| Coldline | $0.004 | 90 days | Quarterly access |
| Archive | $0.0012 | 365 days | Compliance/backups |

---

### Networking: VPC, Firewall Rules, Cloud NAT {#gcp-networking}

GCP VPCs are **global** — a single VPC spans all regions. Subnets are regional.

```bash
# Create VPC and subnet
gcloud compute networks create my-vpc --subnet-mode=custom
gcloud compute networks subnets create my-subnet \
  --network my-vpc \
  --region us-central1 \
  --range 10.0.0.0/24

# Firewall rules
gcloud compute firewall-rules create allow-http \
  --network my-vpc \
  --allow tcp:80,tcp:443 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server

gcloud compute firewall-rules create allow-internal \
  --network my-vpc \
  --allow tcp,udp,icmp \
  --source-ranges 10.0.0.0/8

# Cloud NAT for private VMs to access internet
gcloud compute routers create my-router \
  --network my-vpc --region us-central1

gcloud compute routers nats create my-nat \
  --router my-router \
  --region us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips
```

**Key GCP networking advantage over AWS**: GCP's **private global network** — traffic between regions travels over Google's backbone, not the public internet. Cross-region latency is lower and more consistent than AWS. Also, GCP's global load balancer is truly anycast — a single IP routes to the nearest healthy backend in any region. AWS ALB is regional.

---

### ML-Specific: Vertex AI, TPUs, Cloud GPUs {#gcp-ml}

```bash
# Submit a training job to Vertex AI
gcloud ai custom-jobs create \
  --region us-central1 \
  --display-name "my-training-job" \
  --config training-config.yaml

# training-config.yaml:
# workerPoolSpecs:
#   - machineSpec:
#       machineType: a2-highgpu-8g
#       acceleratorType: NVIDIA_TESLA_A100
#       acceleratorCount: 8
#     replicaCount: 4   # 4 nodes × 8 GPUs = 32 GPUs total
#     containerSpec:
#       imageUri: gcr.io/my-project/training:latest
#       args: ["--batch_size=512", "--lr=1e-4"]

# TPU VM
gcloud compute tpus tpu-vm create my-tpu \
  --zone us-central1-a \
  --accelerator-type v4-32 \   # v4 TPU, 32 cores
  --version tpu-vm-tf-2.13.0

gcloud compute tpus tpu-vm ssh my-tpu --zone us-central1-a
```

**TPU vs GPU for training**: TPUs are designed for matrix multiplication at scale (transformer training). v4 Pod: 4096 chips, theoretically ~1.1 exaFLOPs. Best for training very large models (>10B parameters) where you can keep the systolic array busy. GPUs are more flexible (custom CUDA kernels, inference, smaller models). Most teams use GPUs; TPUs are worth it at frontier scale.

---

### GCP Pitfalls {#gcp-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **Service account key files in repo** | Keys leaked → GCP bill from crypto mining | Use Workload Identity; no key files |
| **Default service account with editor role** | Every VM can modify all project resources | Create narrow-scoped service accounts per service |
| **GCS egress to internet** | $0.08–0.23/GB depending on destination | Use VPC Service Controls; prefer intra-region access |
| **GKE cluster not private** | Control plane and nodes publicly accessible | Enable private cluster; use master authorised networks |
| **Cloud Run cold start with large image** | 10–30 second cold start | Use small base images; min-instances=1 for latency-sensitive |
| **Firewall rule 0.0.0.0/0 on all ports** | VM fully exposed | Default-deny; allow only needed ports |
| **Not setting budget alerts** | Bill surprise at month end | Set budget with email/PubSub alerts at 50%, 90%, 100% |

---

## API Gateways & Reverse Proxies {#api-gateways}

### What an API Gateway Does {#what-is-gateway}

An API gateway sits between clients and your backend services. It handles cross-cutting concerns so your application code doesn't have to:

```
Client
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    API Gateway / Reverse Proxy           │
│                                                         │
│  TLS termination  →  Authentication  →  Rate limiting   │
│       │                   │                  │          │
│  Request routing  →  Load balancing  →  Logging         │
│       │                   │                  │          │
│  Caching          →  Request/response transformation    │
│                                                         │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Service A    Service B    Service C
  (Users API)  (Orders API) (Products)
```

Without a gateway, each service would need to implement auth, rate limiting, TLS, and logging independently. The gateway enforces these uniformly.

---

### Nginx: How It Works {#nginx}

Nginx (pronounced "engine-x") is an event-driven, non-blocking web server and reverse proxy. It handles tens of thousands of concurrent connections with minimal memory because it does not create a thread per connection (unlike Apache's prefork model).

**Architecture:**

```
                   Master Process
                   (reads config, binds ports)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Worker Process  Worker Process  Worker Process
    (one per CPU    (event loop,    
     core)          handles N connections each)
                          │
                   ┌──────┴──────┐
                   ▼             ▼
            Accept conn    epoll/kqueue
            read request   (OS event
            proxy/serve    notification)
            send response
```

The **event loop** model: each worker uses `epoll` (Linux) to watch thousands of file descriptors. When a socket is readable, the worker reads, processes, and writes without blocking. No thread context switches, no thread per connection overhead.

**Request lifecycle:**

```
1. Client: TCP connect → TLS handshake → HTTP request
2. Nginx: accept connection on port 443
3. Terminate TLS (decrypt request)
4. Match server_name → virtual host
5. Match location block (URI path)
6. Apply directives: auth, rate limit, cache check
7. Proxy_pass to upstream backend
8. Receive response from backend
9. Optionally cache response
10. Send response to client (keep-alive or close)
```

---

### Nginx Configuration in Practice {#nginx-config}

```nginx
# /etc/nginx/nginx.conf

worker_processes auto;          # one per CPU core
worker_connections 4096;        # max connections per worker
                                # total = worker_processes × worker_connections

events {
    use epoll;                  # Linux: use epoll for event notification
    multi_accept on;            # accept all new connections at once
}

http {
    # ── Upstream backends (load balancing) ──────────────────────────────
    upstream api_backend {
        least_conn;             # route to backend with fewest active connections
        server 10.0.1.10:8080;
        server 10.0.1.11:8080;
        server 10.0.1.12:8080;
        keepalive 32;           # keep 32 idle connections to backends
    }

    # ── Rate limiting ────────────────────────────────────────────────────
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    # key: client IP, zone: 10MB shared memory (~160K IPs), rate: 100 req/min

    # ── Caching ──────────────────────────────────────────────────────────
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m
                     max_size=1g inactive=60m use_temp_path=off;

    # ── HTTP → HTTPS redirect ────────────────────────────────────────────
    server {
        listen 80;
        server_name api.example.com;
        return 301 https://$host$request_uri;
    }

    # ── Main HTTPS server ────────────────────────────────────────────────
    server {
        listen 443 ssl http2;
        server_name api.example.com;

        # TLS
        ssl_certificate     /etc/ssl/certs/api.example.com.crt;
        ssl_certificate_key /etc/ssl/private/api.example.com.key;
        ssl_protocols       TLSv1.2 TLSv1.3;
        ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_session_cache   shared:SSL:10m;
        ssl_session_timeout 1d;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000" always;
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;

        # ── Static files (served directly by Nginx) ──────────────────────
        location /static/ {
            root /var/www;
            expires 30d;
            add_header Cache-Control "public, immutable";
            gzip_static on;
        }

        # ── API proxy with rate limiting ─────────────────────────────────
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            # burst: allow up to 20 req above rate limit momentarily
            # nodelay: don't queue, reject excess immediately

            proxy_pass http://api_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";            # enable keepalive to upstream
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 5s;   # time to connect to backend
            proxy_read_timeout    60s;  # time to wait for backend response
            proxy_send_timeout    60s;

            # Retry on failure (idempotent methods only)
            proxy_next_upstream error timeout http_502 http_503;
            proxy_next_upstream_tries 3;
        }

        # ── Cached endpoint ───────────────────────────────────────────────
        location /api/products {
            proxy_cache api_cache;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_valid 200 5m;          # cache 200 responses for 5 minutes
            proxy_cache_valid 404 1m;
            proxy_cache_use_stale error timeout updating; # serve stale on backend error
            add_header X-Cache-Status $upstream_cache_status;  # HIT or MISS in response

            proxy_pass http://api_backend;
        }

        # ── Health check endpoint ─────────────────────────────────────────
        location /health {
            access_log off;
            return 200 "OK\n";
            add_header Content-Type text/plain;
        }
    }
}
```

**Key Nginx commands:**

```bash
nginx -t                          # test config syntax before applying
nginx -s reload                   # reload config without dropping connections (graceful)
nginx -s quit                     # graceful shutdown (finish in-flight requests)

# Check active connections
nginx -V 2>&1 | grep with-http_stub_status_module  # confirm module is compiled in
# Then add to config: location /nginx_status { stub_status; allow 127.0.0.1; deny all; }
curl http://localhost/nginx_status

# Watch access log in real time
tail -f /var/log/nginx/access.log

# Count requests per status code in last 5 minutes
awk -v d="$(date -d '5 minutes ago' +'%d/%b/%Y:%H:%M')" '$4 > "["d' \
  /var/log/nginx/access.log | awk '{print $9}' | sort | uniq -c | sort -rn

# Top 10 client IPs by request count
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head 10

# Slowest requests (if $request_time is logged)
awk '{print $NF, $7}' /var/log/nginx/access.log | sort -rn | head 20
```

**Nginx log format for observability:**

```nginx
log_format detailed '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent" '
                    'rt=$request_time uct=$upstream_connect_time '
                    'uht=$upstream_header_time urt=$upstream_response_time '
                    'cs=$upstream_cache_status';

access_log /var/log/nginx/access.log detailed;
```

`rt` = total request time; `urt` = backend response time. `rt - urt` = time spent in Nginx (mostly SSL handshake + network).

---

### Load Balancing Algorithms {#load-balancing}

```
Incoming requests: ─────────────────────────────────→

Round Robin (default):
  Req 1 → Server A
  Req 2 → Server B
  Req 3 → Server C
  Req 4 → Server A ...
  Problem: ignores response time; slow server gets same traffic

Least Connections:
  → Server with fewest active connections
  Better for variable request durations (some slow, some fast)
  Nginx: `least_conn;`

IP Hash:
  hash(client_ip) % N → always same server
  Use case: session stickiness without session store
  Problem: uneven distribution if traffic comes from few IPs (NAT)
  Nginx: `ip_hash;`

Weighted Round Robin:
  Server A weight=3, Server B weight=1
  → A handles 75%, B handles 25%
  Use when servers have different capacities
  Nginx: `server 10.0.0.1 weight=3; server 10.0.0.2 weight=1;`

Random with two choices (Power of Two):
  Pick 2 servers randomly, send to the one with fewer connections
  Reduces variance vs pure least-conn at scale
  Nginx Plus / Envoy support this natively

Health checks:
  Active:  Nginx periodically sends probe requests to backends
  Passive: Mark backend down when it returns 5xx or times out
  Nginx: `server 10.0.0.1 max_fails=3 fail_timeout=30s;`
```

**Layer 4 vs Layer 7 load balancing:**

```
L4 (TCP/UDP):                    L7 (HTTP):
  ┌──────────────────┐             ┌──────────────────────────────┐
  │ Look at:          │             │ Look at:                      │
  │  - Source IP/port │             │  - URL path                   │
  │  - Dest IP/port   │             │  - HTTP headers               │
  │                   │             │  - Cookie                     │
  │ Cannot:           │             │  - Request body               │
  │  - Inspect HTTP   │             │                               │
  │  - Route by URL   │             │ Can:                          │
  │                   │             │  - Route /api → service A     │
  │ Faster (no TLS    │             │  - Route /ws → WebSocket svc  │
  │ termination)      │             │  - Sticky sessions by cookie  │
  └──────────────────┘             └──────────────────────────────┘
  AWS NLB, HAProxy TCP             AWS ALB, Nginx, Envoy
```

**Consistent hashing** for stateful services (e.g., cache clusters): when a node is added/removed, only $\frac{1}{N}$ of keys move rather than remapping everything. Essential for sharded caches (Memcached, Redis Cluster).

---

### Managed Gateways: AWS ALB, Kong, Envoy {#managed-gateways}

#### AWS ALB (Application Load Balancer)

```
Client → ALB (Layer 7) → Target Groups → EC2 / ECS / Lambda

Routing rules:
  IF host = api.example.com AND path = /v1/*
    → Target group: api-v1-targets (weight 90%)
    → Target group: api-v2-targets (weight 10%)  ← canary

  IF path = /ws/*
    → Target group: websocket-servers (sticky sessions)

  IF header X-Version = beta
    → Target group: beta-backend
```

```bash
# Create ALB target group
aws elbv2 create-target-group \
  --name api-targets \
  --protocol HTTP \
  --port 8080 \
  --vpc-id vpc-abc123 \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Add routing rule for path-based routing
aws elbv2 create-rule \
  --listener-arn arn:aws:elasticloadbalancing:...:listener/... \
  --priority 10 \
  --conditions '[{"Field":"path-pattern","Values":["/api/*"]}]' \
  --actions '[{"Type":"forward","TargetGroupArn":"..."}]'
```

ALB **connection draining** (deregistration delay): when a target is deregistered (e.g., during deploy), ALB waits up to 300 seconds for in-flight requests to complete before forcibly closing connections. Set to 30–60 seconds for most web services.

#### Kong

Open-source API gateway built on Nginx with a plugin system. Adds auth, rate limiting, logging, transformations via plugins — no custom Nginx config needed.

```bash
# Add a service (backend)
curl -X POST http://localhost:8001/services \
  --data name=my-api \
  --data url=http://my-backend:8080

# Add a route (how requests reach the service)
curl -X POST http://localhost:8001/services/my-api/routes \
  --data "paths[]=/api"

# Enable rate limiting plugin on the service
curl -X POST http://localhost:8001/services/my-api/plugins \
  --data name=rate-limiting \
  --data config.minute=1000 \
  --data config.hour=10000 \
  --data config.policy=redis \
  --data config.redis_host=redis

# Enable JWT auth
curl -X POST http://localhost:8001/services/my-api/plugins \
  --data name=jwt
```

#### Envoy

Envoy is a high-performance proxy written in C++ used as the data plane in service meshes (Istio). Key features:

- **xDS APIs**: dynamically reconfigure routing, endpoints, TLS — no restart
- **Circuit breaking**: stop sending requests to a failing upstream; fail fast
- **Retry with backoff**: automatic retries on 5xx with jitter
- **Outlier detection**: automatically eject misbehaving hosts from load balancer pool
- **Distributed tracing**: propagates trace headers (Jaeger, Zipkin, OTLP)

```yaml
# envoy.yaml — simple HTTP proxy config
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address: { address: 0.0.0.0, port_value: 8080 }
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          route_config:
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match: { prefix: "/api" }
                route:
                  cluster: api_service
                  retry_policy:
                    retry_on: "5xx,connect-failure"
                    num_retries: 3

  clusters:
  - name: api_service
    connect_timeout: 0.25s
    type: STRICT_DNS
    load_assignment:
      cluster_name: api_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address: { address: api-backend, port_value: 8080 }
    circuit_breakers:
      thresholds:
      - max_connections: 1000
        max_pending_requests: 500
        max_retries: 3
```

---

### API Gateway Pitfalls {#gateway-pitfalls}

**Timeout mismatches — the most common production issue:**

```
Client timeout: 30s
    └→ ALB timeout: 60s
           └→ Nginx timeout: 30s   ← mismatch! Nginx closes first,
                  └→ Backend: 25s     ALB sees 499, client sees 502
```

Set timeouts so: `client > ALB/Nginx > backend`. Otherwise you get cascading confusing errors. A good default: backend 10s, Nginx 15s, ALB 30s, client 60s.

**Large file upload through gateway:**

Nginx buffers the full request body before proxying by default. A 1GB model upload will buffer to disk, adding latency and disk I/O.

```nginx
# Disable buffering for large uploads
proxy_request_buffering off;
client_max_body_size 0;   # no size limit
```

**Other pitfalls:**

| Pitfall | Symptom | Fix |
|---|---|---|
| **`X-Forwarded-For` not propagated** | Backend logs show gateway IP, not client IP | `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for` |
| **Rate limit by IP behind NAT** | Office of 100 people all rate-limited together | Rate limit by API key or user ID, not IP |
| **No circuit breaker** | One slow backend → all threads blocked → gateway down | Envoy circuit breaker; or `proxy_next_upstream` |
| **Cache poisoning** | Different users see each other's responses | Include `Authorization` or user ID in cache key; never cache authenticated responses without key differentiation |
| **SSL termination without HSTS** | Users downgraded to HTTP | Add `Strict-Transport-Security` header |
| **WebSocket timeout** | Long-lived connections dropped after 60s | Set `proxy_read_timeout 3600s` for WebSocket endpoints; send keep-alive pings |
| **gzip on already-compressed content** | CPU wasted, size increases slightly | `gzip_types` should exclude `image/*`, `video/*`, `.gz` |

---

## Observability {#observability}

### Logs, Metrics, Traces {#three-pillars}

The three pillars answer different questions:

```
LOGS                    METRICS                  TRACES
"What happened?"        "How is the system?"     "Why is this slow?"

2024-01-15 10:23:41     api_latency_p99=234ms    Request ID: abc-123
ERROR [api] DB timeout  error_rate=0.02%         │
user_id=u123            rps=1247                  ├─ nginx: 2ms
request=/api/orders     cpu_usage=67%             ├─ auth-service: 45ms
duration=5001ms                                   │   └─ db query: 40ms
                                                  ├─ api-service: 180ms
Free text,              Time-series numbers.       │   ├─ cache miss: 5ms
unstructured.           Aggregatable.              │   └─ db query: 170ms ← slow
Hard to aggregate.      No context.               └─ Total: 227ms

Tools: CloudWatch Logs  Tools: CloudWatch,       Tools: Jaeger,
       GCP Logging      Prometheus, Datadog       Zipkin, AWS X-Ray
       ELK, Loki                                  GCP Cloud Trace
```

**Structured logging** — always use JSON, never free text:

```python
import structlog
log = structlog.get_logger()

# BAD: "User 123 placed order 456 for $99.99"
# GOOD:
log.info("order_placed",
    user_id="u123",
    order_id="o456",
    amount_cents=9999,
    duration_ms=45,
    region="us-east-1"
)
# → {"event":"order_placed","user_id":"u123","order_id":"o456",...}
```

Structured logs are queryable: `filter event="order_placed" AND duration_ms > 1000` to find slow orders. Free text requires regex parsing.

**Key metrics to instrument for any service:**

| Metric | Why |
|---|---|
| Request rate (RPS) | Traffic baseline; detect spikes |
| Error rate (`5xx / total`) | Service health |
| Latency p50, p95, p99 | User experience; p99 finds tail latency |
| Saturation (CPU, memory, queue depth) | Capacity planning |
| Dependency error rates | Upstream failures |

The **RED method** (for services): Rate, Errors, Duration.  
The **USE method** (for infrastructure): Utilisation, Saturation, Errors.

---

### Practical Observability Stack {#practical-obs}

**AWS stack:**

```
Application → CloudWatch Logs → Log Insights (query)
           → CloudWatch Metrics → Alarms → SNS → PagerDuty
           → AWS X-Ray → Service Map (traces)
           → CloudWatch Container Insights (ECS/EKS auto metrics)
```

**GCP stack:**

```
Application → Cloud Logging → Log Explorer → Log-based metrics
           → Cloud Monitoring → Alerting Policies → PagerDuty
           → Cloud Trace → Latency distribution
```

**Self-managed (common for ML teams):**

```
Application → Prometheus (scrape metrics) → Grafana (dashboards)
           → Loki (logs, like Prometheus for logs) → Grafana
           → Tempo (traces) → Grafana

All in one Grafana dashboard with correlated logs+metrics+traces
```

```bash
# Prometheus query examples (PromQL)

# 5-minute error rate
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])

# p99 latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Memory usage as % of limit
container_memory_usage_bytes / container_spec_memory_limit_bytes * 100

# Alert: error rate > 1% for 5 minutes
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Error rate {{ $value | humanizePercentage }} on {{ $labels.service }}"
```

---

## Cost Management {#cost}

Cloud bills are the easiest place for money to disappear silently.

**The biggest cost drivers for ML/data teams:**

```
Typical monthly bill breakdown:

EC2/GCE (compute)    ████████████████████  40%
  - Idle GPUs                               ← biggest waste
  - Overprovisioned instance sizes

S3/GCS (storage)     ████████              20%
  - Forgotten old checkpoints
  - Logs never deleted

Data Transfer        ██████                15%
  - Cross-AZ traffic
  - Internet egress

RDS/databases        ████                  10%

Everything else      ██████                15%
```

**Practical cost controls:**

```bash
# 1. Find idle EC2 instances (CPU < 5% for 7 days)
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 --metric-name CPUUtilization \
  --period 604800 --statistics Average \
  --start-time 7-days-ago --end-time now

# 2. Find unattached EBS volumes (wasted money)
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --query 'Volumes[*].[VolumeId,Size,CreateTime]' \
  --output table

# 3. Find unused Elastic IPs (charged when not attached)
aws ec2 describe-addresses \
  --query 'Addresses[?!InstanceId].[PublicIp,AllocationId]'

# 4. S3 cost by prefix (identify large directories)
aws s3api list-objects-v2 \
  --bucket my-bucket \
  --query 'sum(Contents[].Size)'

# 5. Enable S3 Intelligent-Tiering on cold data
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-bucket \
  --id entire-bucket \
  --intelligent-tiering-configuration '{"Id":"entire-bucket","Status":"Enabled","Tierings":[{"Days":90,"AccessTier":"ARCHIVE_ACCESS"}]}'
```

**Cost allocation tags**: tag every resource with `team`, `environment`, `project`. Then use Cost Explorer to break down by tag. Without tags, you cannot attribute costs to teams.

**Savings hierarchy (most to least effort):**
1. Delete unused resources (EBS, IPs, old snapshots) — free money
2. Right-size instances — use CloudWatch to check actual CPU/memory utilisation
3. Use Spot/Preemptible for stateless, fault-tolerant workloads
4. Reserved Instances / Committed Use for stable baseline
5. Savings Plans (AWS) / CUDs (GCP) for flexible commitment

---

## Production Patterns {#production-patterns}

**Blue-Green Deployment:**

```
Current (Blue): v1 serving 100% traffic
         ALB
          ├── Blue target group (v1) ← 100%
          └── Green target group (v2) ← 0%

Deploy v2 to Green, run smoke tests
Switch:
         ALB
          ├── Blue target group (v1) ← 0%
          └── Green target group (v2) ← 100%

Rollback: flip back to Blue instantly
```

**Canary Deployment:**

```
Gradually shift traffic:
  1%  → Green (monitor error rate, latency)
  10% → Green
  50% → Green
  100% → Green (retire Blue)

Rollback threshold: if error rate > 0.5%, roll back automatically
AWS CodeDeploy + Lambda / ALB weighted routing supports this natively
```

**Circuit Breaker Pattern:**

```
States:
  CLOSED (normal) → requests pass through
         ↓ N failures in window
  OPEN (tripped)  → requests fail immediately (no backend calls)
         ↓ after timeout
  HALF-OPEN       → let one request through
         ↓ success            ↓ failure
  CLOSED (recover)        OPEN (stay tripped)

Prevents: cascade failure where one slow service exhausts connection pools
          of its callers, taking down the whole system
```

**Graceful shutdown for containers:**

```python
import signal
import sys

class Server:
    def __init__(self):
        self.shutting_down = False
        signal.signal(signal.SIGTERM, self.handle_sigterm)

    def handle_sigterm(self, signum, frame):
        # Kubernetes sends SIGTERM before killing the pod
        self.shutting_down = True
        # Stop accepting new requests
        # Wait for in-flight requests to complete
        # Close DB connections
        sys.exit(0)

    def handle_request(self, request):
        if self.shutting_down:
            # Return 503 so load balancer routes to healthy instances
            return Response(status=503, headers={"Retry-After": "5"})
        # ... normal handling ...
```

**The deployment checklist that actually matters in production:**

```
Before deploy:
  □ Health check endpoint exists and returns 200 only when ready
  □ Graceful shutdown handles SIGTERM
  □ Timeouts set on all external calls (DB, APIs, caches)
  □ Circuit breakers on critical dependencies
  □ Structured logging with request IDs
  □ Metrics instrumented (RED: rate, errors, duration)
  □ Alerts set on error rate and p99 latency

During deploy:
  □ Watch error rate dashboard
  □ Monitor p99 latency
  □ Check pod restart count (kubectl get pods --watch)
  □ Ready to rollback in < 2 minutes

After deploy:
  □ Error rate back to baseline
  □ No memory leak (watch RSS over 30 minutes)
  □ No unexpected cost increase
```
