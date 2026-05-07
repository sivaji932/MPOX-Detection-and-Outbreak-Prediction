# 🚀 MPOX Detection and Outbreak Prediction — CI/CD + AWS Deployment Infrastructure

This repository contains the complete production-grade CI/CD and AWS deployment setup for the MPOX Detection and Outbreak Prediction project.

The infrastructure includes:

- Jenkins CI/CD
- Docker containerization
- AWS ECR
- GitHub Webhooks
- Auto Scaling Group (ASG)
- Application Load Balancer (ALB)
- Redis via ElastiCache
- Secure secret injection using AWS Systems Manager Parameter Store
- AWS Security groups configurations available at last.

---

# 📌 Full Process Flow (INDEX)

This is the overall deployment roadmap.

```text
🧱 PHASE 1 — Base Setup
Create Jenkins EC2
Install Jenkins + Docker
Configure Jenkins (plugins, credentials)

🧱 PHASE 2 — Container Registry
Create ECR repository
Configure AWS CLI + IAM for Jenkins
Test push image → ECR

🧱 PHASE 3 — CI/CD Automation
GitHub Webhook
Jenkins Pipeline
Automatic Docker build + ECR push

🧱 PHASE 4 — Secure Runtime Infrastructure
Parameter Store
Redis Injection
IAM Roles
Launch Template

🧱 PHASE 5 — Production Auto Scaling Deployment
ASG
ALB
Automatic EC2 bootstrap
Container auto deployment
```

---

# 🧱 PHASE 1 — Jenkins Server Setup (Ubuntu 24.04)

We’ll set up:

```text
Ubuntu 24.04 EC2
   ↓
Java 21
   ↓
Jenkins LTS
   ↓
Docker
   ↓
Ready for CI/CD
```

---

# 🧠 Recommended Versions

## ✅ Jenkins

Use:

```text
Jenkins LTS 2.555.x
```

Current LTS line is around 2.555.x in 2026.

---

## ✅ Java

Use:

```text
OpenJDK 21 (LTS)
```

### Why

- Jenkins officially supports Java 21
- Future-proof
- Better long-term support than Java 17 now

---

# 🧱 STEP 0 — EC2 Instance

Create Jenkins EC2.

## Recommended

| Configuration | Value        |
| ------------- | ------------ |
| OS            | Ubuntu 24.04 |
| Type          | t3.medium    |
| Storage       | 25 GB        |

---

## Security Group

Allow:

| Port | Purpose    |
| ---- | ---------- |
| 22   | SSH        |
| 8080 | Jenkins UI |

---

# 🧱 STEP 1 — Update Ubuntu

SSH into EC2:

```bash
sudo apt update && sudo apt upgrade -y
```

---

# 🧱 STEP 2 — Install Java 21

```bash
sudo apt install -y openjdk-21-jdk
```

Verify:

```bash
java -version
```

Expected:

```text
openjdk version "21..."
```

---

# 🧱 STEP 3 — Install Jenkins LTS

```bash
# latest java and jenkins installation for ubuntu lts 24.04

sudo apt update
sudo apt install fontconfig openjdk-21-jre

sudo wget -O /etc/apt/keyrings/jenkins-keyring.asc \
  https://pkg.jenkins.io/debian-stable/jenkins.io-2026.key

echo "deb [signed-by=/etc/apt/keyrings/jenkins-keyring.asc]" \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

sudo apt update
sudo apt install jenkins

sudo systemctl enable jenkins
sudo systemctl start jenkins
sudo systemctl status jenkins
```

---

# 🧱 STEP 5 — Open Jenkins UI

Browser:

```text
http://<JENKINS-EC2-IP>:8080
```

---

# 🧱 STEP 6 — Get Admin Password

```bash
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

Copy → paste into UI.

---

# 🧱 STEP 7 — Initial Jenkins Setup

Choose:

```text
Install Suggested Plugins
```

---

# 🧱 STEP 8 — Install Docker (IMPORTANT)

Jenkins will build Docker images.

## Install Docker

```bash
sudo apt install -y docker.io
```

## Start Docker

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

## Install Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
```

---

# 🧱 STEP 9 — Allow Jenkins to use Docker

```bash
sudo usermod -aG docker jenkins
sudo usermod -aG docker ubuntu
```

Restart services:

```bash
sudo systemctl restart docker
sudo systemctl restart jenkins
```

---

# 🧪 STEP 10 — Verify Docker inside Jenkins Server

```bash
docker --version
docker ps
```

---

# 🧱 STEP 11 — Install AWS CLI

```bash
sudo apt update && sudo apt install -y curl unzip

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Verify:

```bash
aws --version
```

---

# 🧠 PHASE 1 RESULT

You now have:

```text
Jenkins Server
 ├── Java 21
 ├── Jenkins LTS
 ├── Docker
 └── AWS CLI
```

---

# 🚀 PHASE 2 — ECR Setup + Jenkins Integration

## Goal

```text
Jenkins
   ↓
Build Docker image
   ↓
Push image → AWS ECR
```

After this phase:

- Jenkins can push images securely
- EC2 instances can later pull images automatically

---

# 🧠 What is ECR?

Use:

```text
Amazon Elastic Container Registry
```

Think of it as:

```text
AWS version of DockerHub
```

---

# 🧱 STEP 1 — Create ECR Repository

Go to:

```text
AWS → ECR → Repositories → Create
```

Fill:

| Field           | Value    |
| --------------- | -------- |
| Repository name | mpox-app |
| Visibility      | Private  |

Leave others default.

Click:

```text
Create repository
```

---

# 🧱 STEP 2 — Copy Repository URI

Example:

```text
123456789012.dkr.ecr.ap-south-1.amazonaws.com/mpox-app
```

Save carefully.

We’ll call it:

```text
<ECR_URI>
```

---

# 🧱 STEP 3 — Create IAM User for Jenkins

Go to:

```text
AWS → IAM → Users → Create user
```

Create:

```text
User name: jenkins-ecr-user
```

Enable:

```text
Programmatic access ✅
```

---

# 🧱 STEP 4 — Attach Permissions

Attach:

```text
AmazonEC2ContainerRegistryFullAccess
```

---

# 🧱 STEP 5 — Create Access Keys

After creating user:

```text
Security credentials → Create access key
```

Copy:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

---

# 🧱 STEP 6 — Configure AWS CLI on Jenkins Server

SSH into Jenkins EC2:

```bash
aws configure
```

Enter:

- Access key
- Secret key
- Region: ap-south-1

---

# 🧪 STEP 7 — Test ECR Login (VERY IMPORTANT)

```bash
aws ecr get-login-password --region ap-south-1 | \
docker login --username AWS --password-stdin <ECR_URI>
```

Example:

```bash
aws ecr get-login-password --region ap-south-1 | \
docker login --username AWS --password-stdin 123456789012.dkr.ecr.ap-south-1.amazonaws.com
```

Expected:

```text
Login Succeeded
```

---

# 🧱 STEP 8 — Test Docker Build Manually

Go to project:

```bash
cd MPOX-Detection-and-Outbreak-Prediction
```

Build image:

```bash
docker build -t mpox-app .
```

---

# 🧱 STEP 9 — Tag Image for ECR

```bash
docker tag mpox-app:latest <ECR_URI>:latest
```

Example:

```bash
docker tag mpox-app:latest 123456789012.dkr.ecr.ap-south-1.amazonaws.com/mpox-app:latest
```

---

# 🧱 STEP 10 — Push to ECR

```bash
docker push <ECR_URI>:latest
```

Expected:

```text
Image layers upload successfully
```

---

# 🧪 STEP 11 — Verify in AWS

Go to:

```text
ECR → mpox-app
```

You should see:

```text
latest
```

---

# 🧠 What You Achieved

```text
Jenkins server
   ↓
Can build Docker images
   ↓
Push securely to ECR
```

---

# 🚀 PHASE 3 — GitHub Webhook + Jenkins Pipeline (CI/CD)

## Goal

```text
GitHub push
    ↓
Webhook triggers Jenkins
    ↓
Jenkins builds Docker image
    ↓
Pushes image to ECR
```

After this:

```text
Every git push automatically builds and pushes your latest app image.
```

---

# 🧠 Architecture after Phase 3

```text
GitHub
   ↓ (Webhook)
Jenkins
   ↓
Docker Build
   ↓
ECR
```

---

# 🧱 STEP 1 — Install Jenkins Plugins

Go to:

```text
Jenkins → Manage Jenkins → Plugins
```

Install:

| Plugin             | Purpose               |
| ------------------ | --------------------- |
| GitHub Integration | webhook support       |
| Pipeline           | Jenkinsfile pipelines |
| Docker Pipeline    | docker support        |
| AWS Credentials    | AWS auth              |

---

# Add AWS Credentials to Jenkins

Go to:

```text
Jenkins → Manage Jenkins → Credentials
```

Add credentials.

Choose:

```text
Kind: AWS Credentials
```

Fill:

```text
ID: aws-creds
Access Key ID: <your access key>
Secret Access Key: <your secret key>
```

Save.

---

# 🧱 STEP 2 — Create GitHub Personal Access Token

Go to:

```text
GitHub → Settings → Developer Settings → Tokens (classic)
```

Create token with:

```text
repo
admin:repo_hook
```

Copy token.

---

# 🧱 STEP 3 — Add GitHub Token in Jenkins

Go to:

```text
Jenkins → Manage Jenkins → Credentials
```

Add:

| Field    | Value                      |
| -------- | -------------------------- |
| Kind     | Username with password     |
| Username | `<your github username>` |
| Password | `<github token>`         |
| ID       | github-token               |

---

# 🧱 STEP 4 — Create Jenkins Pipeline Job

Go to:

```text
Jenkins → New Item
```

Select:

```text
Pipeline
```

Name:

```text
mpox-pipeline
```

---

# 🧱 STEP 5 — Configure GitHub Repo

Under:

```text
Pipeline → Definition
```

Choose:

```text
Pipeline script from SCM
```

SCM:

```text
Git
```

Repository URL:

```text
https://github.com/<your-username>/<repo>.git
```

Credentials:

```text
github-token
```

Branch:

```text
*/main
```

(or master)

Script Path:

```text
Jenkinsfile
```

---

# 🧱 STEP 6 — Enable Webhook Trigger

Enable:

```text
GitHub hook trigger for GITScm polling
```

---

# 🧱 STEP 9 — Configure GitHub Webhook

Go to:

```text
GitHub Repo → Settings → Webhooks
```

## Payload URL

```text
http://<JENKINS-IP>:8080/github-webhook/
```

⚠️ IMPORTANT

- ending `/` required
- disable SSL verification

## Content Type

```text
application/json
```

## Events

```text
Just the push event
```

---

# 🧪 STEP 10 — TEST

Push any code:

```bash
git push
```

Expected flow:

```text
GitHub push
   ↓
Webhook triggers Jenkins
   ↓
Jenkins builds Docker image
   ↓
Pushes image to ECR
```

---

# 🧠 Verify

## Jenkins

Build should start automatically.

## ECR

New image pushed.

---

# 🚀 PHASE 4 — Secure Redis Injection + Auto EC2 Bootstrap

This is the phase where your setup becomes fully scalable and ASG-ready.

## Goal

```text
New EC2 launches
   ↓
Automatically gets REDIS_URL
   ↓
Pulls Docker image from ECR
   ↓
Runs Flask + Celery containers
```

---

# 🧠 Components

1. AWS Systems Manager Parameter Store
2. IAM Role for EC2
3. Launch Template
4. User Data Script

---

# 🧱 STEP 1 — Store REDIS_URL in Parameter Store

## Create AWS ElastiCache Redis

Go to:

```text
AWS ElastiCache → Redis
```

Configuration:

| Setting           | Value              |
| ----------------- | ------------------ |
| Engine            | Redis OSS          |
| Deployment option | Node based cluster |
| Cluster mode      | Disabled           |
| Same VPC as EC2   | Yes                |
| Access control    | No access control  |
| TLS               | Enabled            |

After creation, copy Primary Endpoint.

---

Go to:

```text
AWS → Systems Manager → Parameter Store
```

Create parameter:

| Field | Value           |
| ----- | --------------- |
| Name  | /mpox/redis_url |
| Type  | SecureString    |

Example value:

```text
rediss://master.mpox-redis.xxxxx.cache.amazonaws.com:6379/0?ssl_cert_reqs=none
```

Encryption:

```text
Default AWS managed key
```

---

# 🧠 Why This Is Important

Now:

- no Redis URL in GitHub
- no Redis URL in Dockerfile
- no Redis URL in compose

Secure runtime injection enabled.

---

# 🧱 STEP 2 — Create IAM Role for EC2

Go to:

```text
AWS → IAM → Roles → Create Role
```

Trusted entity:

```text
AWS Service → EC2
```

---

# 🧱 STEP 3 — Attach Permissions

Attach:

```text
AmazonEC2ContainerRegistryReadOnly
AmazonSSMReadOnlyAccess
```

---

# 🧱 STEP 4 — Name Role

```text
mpox-ec2-role
```

---

# 🧠 Final Production Flow

```text
GitHub push
   ↓
Jenkins builds image
   ↓
Push image → ECR
   ↓
ASG launches EC2
   ↓
User Data runs
   ↓
Fetch REDIS_URL
   ↓
Pull image from ECR
   ↓
Start containers
```

---

# 🚀 PHASE 5 — Launch Template + Auto Scaling Group (ASG)

Infrastructure becomes:

- Self-healing
- Auto-scalable
- Fully automated

---

# 🧠 Goal

When AWS launches a new EC2 instance automatically:

1. EC2 boots
2. User Data script runs
3. Docker installs
4. REDIS_URL fetched securely
5. Image pulled from ECR
6. Flask + Celery containers start
7. ALB sends traffic

---

# 🧱 Architecture after this phase

```text
GitHub
   ↓
Jenkins
   ↓
ECR
   ↓
ASG (EC2 instances)
   ↓
ALB
```

---

# 🧱 STEP 1 — Create EC2 Security Group (APP SG)

Go to:

```text
EC2 → Security Groups → Create
```

Name:

```text
mpox-app-sg
```

## Inbound Rules

| Type       | Port | Source             |
| ---------- | ---- | ------------------ |
| SSH        | 22   | Your IP            |
| Custom TCP | 8000 | ALB Security Group |

⚠️ IMPORTANT

For port 8000:

Use:

```text
alb-sg
```

NOT:

```text
0.0.0.0/0
```

---

# 🧱 STEP 2 — Create Launch Template

Go to:

```text
EC2 → Launch Templates → Create
```

## Basic Info

```text
Name: mpox-launch-template
```

## AMI

```text
Ubuntu 24.04
```

## Instance Type

```text
t3.medium
```

## Security Group

```text
mpox-app-sg
```

## IAM Role

```text
mpox-ec2-role
```

---

# 🧱 STEP 3 — Add User Data Script

Paste:

```bash
#!/bin/bash

exec > /var/log/user-data.log 2>&1
set -x

export DEBIAN_FRONTEND=noninteractive

# update package list only
apt update -y

# install required packages
apt install -y docker.io 

# install aws cli
apt install -y curl unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# install docker compose manually
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
-o /usr/local/bin/docker-compose

chmod +x /usr/local/bin/docker-compose

# start docker
systemctl enable docker
systemctl start docker

sleep 10

# allow ubuntu user docker access
usermod -aG docker ubuntu

# fetch REDIS_URL
REDIS_URL=$(aws ssm get-parameter \
  --name "/mpox/redis_url" \
  --with-decryption \
  --query "Parameter.Value" \
  --output text \
  --region ap-south-1)

# export env
echo "REDIS_URL=$REDIS_URL" >> /etc/environment
export REDIS_URL=$REDIS_URL

# login to ECR
aws ecr get-login-password --region ap-south-1 | \
docker login --username AWS --password-stdin 857263388884.dkr.ecr.ap-south-1.amazonaws.com

# create app dir
mkdir -p /home/ubuntu/mpox-app
cd /home/ubuntu/mpox-app

# download compose file
curl -O https://raw.githubusercontent.com/sivaji932/MPOX-Detection-and-Outbreak-Prediction/master/docker-compose.yml

# pull latest image
docker-compose pull

# run containers
docker-compose up -d

# cleanup
docker system prune -af
```

---

# 🧠 What This Script Does

| Step              | Purpose                 |
| ----------------- | ----------------------- |
| install docker    | runtime                 |
| fetch REDIS_URL   | secure secret injection |
| login to ECR      | image pull              |
| pull compose      | deployment              |
| docker compose up | run containers          |

---

# 🧱 STEP 4 — Create Target Group

Go to:

```text
EC2 → Target Groups → Create
```

Fill:

| Field    | Value     |
| -------- | --------- |
| Type     | Instances |
| Name     | mpox-tg   |
| Protocol | HTTP      |
| Port     | 8000      |

Health check path:

```text
/
```

(or Flask health endpoint)

---

# 🧱 STEP 5 — Create ALB

Go to:

```text
EC2 → Load Balancers → Create Application Load Balancer
```

## Basic

```text
Name: mpox-alb
Scheme: Internet-facing
```

## Network

- At least 2 AZs
- Public subnets

## Security Group

```text
alb-sg
```

## Listener

```text
HTTP : 80 → Forward to mpox-tg
```

---

# 🧱 STEP 6 — Create Auto Scaling Group

Go to:

```text
EC2 → Auto Scaling Groups → Create
```

## Name

```text
mpox-asg
```

## Launch Template

```text
mpox-launch-template
```

## Attach to ALB

```text
Attach to existing target group → mpox-tg
```

## Group Size

| Setting | Value |
| ------- | ----- |
| Desired | 2     |
| Min     | 1     |
| Max     | 3     |

## Scaling Policy

```text
Target tracking → CPU Utilization: 70%
```

---

# 🧪 STEP 7 — TEST

ASG launches instances automatically.

Check EC2 instances.

Expected:

```text
2 running instances
```

---

# 🧪 STEP 8 — Verify Containers

SSH into instance:

```bash
docker ps
```

Expected:

```text
mpox-flask
mpox-celery
```

---

# 🧪 STEP 9 — Test ALB

Open:

```text
http://<ALB-DNS>
```

---

# 🔥 Final Architecture

```text
GitHub
   ↓
Webhook
   ↓
Jenkins
   ↓
Docker image
   ↓
ECR
   ↓
ASG launches EC2
   ↓
User Data auto deploys containers
   ↓
ALB routes traffic
```

---

# 🔐 AWS Security Group Configuration

---

# 🧱 1. ALB Security Group (alb-sg)

## Purpose

Allows public internet traffic to reach the Application Load Balancer.

---

## ✅ Inbound Rules

| Type  | Protocol | Port | Source    |
| ----- | -------- | ---- | --------- |
| HTTP  | TCP      | 80   | 0.0.0.0/0 |
| HTTPS | TCP      | 443  | 0.0.0.0/0 |

---

## ✅ Outbound Rules

| Type        | Protocol | Port | Destination |
| ----------- | -------- | ---- | ----------- |
| All Traffic | All      | All  | 0.0.0.0/0   |

---

## 🧠 Traffic Flow

```text
Internet Users
      ↓
ALB (80 / 443)
      ↓
EC2 Instances (8000)
```

---

# 🧱 2. Application EC2 Security Group (mpox-app-sg)

## Purpose

Allows ALB to access Flask application running on EC2 instances.

---

## ✅ Inbound Rules

| Type       | Protocol | Port | Source |
| ---------- | -------- | ---- | ------ |
| SSH        | TCP      | 22   | My IP  |
| Custom TCP | TCP      | 8000 | alb-sg |

---

⚠️ IMPORTANT

For Port 8000:

DO NOT use:

```text
0.0.0.0/0
```

Use:

```text
Source = ALB Security Group (alb-sg)
```

This ensures:

- only ALB can access Flask app
- application server remains protected

---

## ✅ Outbound Rules

| Type        | Protocol | Port | Destination |
| ----------- | -------- | ---- | ----------- |
| All Traffic | All      | All  | 0.0.0.0/0   |

---

# 🧱 3. Redis / ElastiCache Security Group (redis-sg)

## Purpose

Allows Flask and Celery containers inside EC2 instances to communicate with Redis.

---

## ✅ Inbound Rules

| Type       | Protocol | Port | Source      |
| ---------- | -------- | ---- | ----------- |
| Custom TCP | TCP      | 6379 | mpox-app-sg |

---

⚠️ VERY IMPORTANT

For Redis Port 6379:

DO NOT use:

```text
0.0.0.0/0
```

Use:

```text
Source = Application EC2 Security Group (mpox-app-sg)
```

This ensures:

- only application EC2 instances can access Redis
- Redis remains private and secure

---

## ✅ Outbound Rules

| Type        | Protocol | Port | Destination |
| ----------- | -------- | ---- | ----------- |
| All Traffic | All      | All  | 0.0.0.0/0   |

---

# 🔄 Final Secure Networking Flow

```text
User
 ↓
ALB Security Group
 ↓
EC2 Security Group
 ↓
Redis Security Group
```

---

# 🧠 Security Design Logic

| Layer         | Accessible By                |
| ------------- | ---------------------------- |
| ALB           | Public Internet              |
| EC2 Flask App | Only ALB                     |
| Redis         | Only EC2 Application Servers |

---

# 🔥 Production Security Principle

```text
Least Privilege Access
```

Each layer only allows the minimum required communication.
