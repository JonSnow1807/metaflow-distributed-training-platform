# Deployment Guide

This guide walks you through deploying the Metaflow Distributed Training Platform in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Deployment](#aws-deployment)
3. [GCP Deployment](#gcp-deployment)
4. [On-Premise Deployment](#on-premise-deployment)
5. [Multi-Cloud Deployment](#multi-cloud-deployment)
6. [Production Checklist](#production-checklist)

## Prerequisites

### Required Tools

```bash
# Install required CLI tools
brew install kubectl helm aws-cli terraform

# Python dependencies
pip install metaflow kubernetes boto3 google-cloud-storage

# Verify installations
kubectl version --client
helm version
aws --version
terraform --version
```

### Required Permissions

#### AWS IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "eks:*",
        "iam:*",
        "s3:*",
        "cloudwatch:*",
        "logs:*",
        "efs:*",
        "fsx:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## AWS Deployment

### 1. Create EKS Cluster

```bash
# Set environment variables
export AWS_REGION=us-east-1
export CLUSTER_NAME=ml-training-cluster
export NODE_GROUP_NAME=gpu-nodegroup

# Create EKS cluster
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $AWS_REGION \
  --version 1.27 \
  --nodegroup-name cpu-nodes \
  --node-type m5n.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# Add GPU node group
eksctl create nodegroup \
  --cluster $CLUSTER_NAME \
  --region $AWS_REGION \
  --name $NODE_GROUP_NAME \
  --node-type p4d.24xlarge \
  --nodes 0 \
  --nodes-min 0 \
  --nodes-max 16 \
  --node-labels workload=training,gpu=a100 \
  --taints nvidia.com/gpu=true:NoSchedule \
  --taints spot=true:NoSchedule \
  --spot \
  --instance-types p4d.24xlarge,p4de.24xlarge \
  --managed
```

### 2. Install NVIDIA GPU Operator

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set operator.defaultRuntime=containerd \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set migManager.enabled=false \
  --set mig.strategy=single \
  --set gfd.enabled=true \
  --set dcgmExporter.enabled=true
```

### 3. Set Up Storage

#### EFS for Shared Storage
```bash
# Create EFS filesystem
aws efs create-file-system \
  --creation-token ml-training-efs \
  --performance-mode maxIO \
  --throughput-mode provisioned \
  --provisioned-throughput-in-mibps 1024 \
  --tags "Key=Name,Value=ml-training-efs" \
  --region $AWS_REGION

# Install EFS CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.5"

# Create StorageClass
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-XXXXXXXXX  # Replace with your EFS ID
  directoryPerms: "1000"
EOF
```

#### FSx for Lustre (High Performance)
```bash
# Create FSx filesystem
aws fsx create-file-system \
  --file-system-type LUSTRE \
  --lustre-configuration DeploymentType=PERSISTENT_2,PerUnitStorageThroughput=1000 \
  --storage-capacity 1200 \
  --subnet-ids subnet-XXXXXXXXX \
  --tags "Key=Name,Value=ml-training-fsx" \
  --region $AWS_REGION

# Install FSx CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.2"
```

### 4. Configure S3 for Checkpoints

```bash
# Create S3 bucket
aws s3 mb s3://ml-training-checkpoints-$RANDOM \
  --region $AWS_REGION

# Set lifecycle policy for cost optimization
cat <<EOF > lifecycle.json
{
  "Rules": [
    {
      "Id": "TransitionToIA",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket ml-training-checkpoints-$RANDOM \
  --lifecycle-configuration file://lifecycle.json
```

### 5. Deploy Training Infrastructure

```bash
# Create namespace
kubectl create namespace ml-training

# Create secrets
kubectl create secret generic aws-credentials \
  --from-literal=access_key_id=$AWS_ACCESS_KEY_ID \
  --from-literal=secret_access_key=$AWS_SECRET_ACCESS_KEY \
  -n ml-training

kubectl create secret generic wandb-credentials \
  --from-literal=api_key=$WANDB_API_KEY \
  -n ml-training

# Deploy training components
kubectl apply -f kubernetes/training-job.yaml
kubectl apply -f kubernetes/monitoring-stack.yaml

# Verify deployment
kubectl get pods -n ml-training
kubectl get pods -n monitoring
```

### 6. Configure Spot Instance Interruption Handling

```bash
# Install AWS Node Termination Handler
helm repo add eks https://aws.github.io/eks-charts
helm install aws-node-termination-handler \
  eks/aws-node-termination-handler \
  --namespace kube-system \
  --set enableSpotInterruptionDraining=true \
  --set enableScheduledEventDraining=true \
  --set webhookURL="http://prometheus-pushgateway.monitoring:9091/metrics/job/node-termination"
```

### 7. Set Up Monitoring

```bash
# Get Grafana URL
export GRAFANA_URL=$(kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Grafana URL: http://$GRAFANA_URL:3000"
echo "Default credentials: admin/admin123"

# Import dashboards
curl -X POST http://$GRAFANA_URL:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/fsdp-training.json \
  -u admin:admin123
```

## GCP Deployment

### 1. Create GKE Cluster

```bash
# Set environment variables
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1
export CLUSTER_NAME=ml-training-cluster

# Create cluster
gcloud container clusters create $CLUSTER_NAME \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --machine-type=n1-standard-16 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5 \
  --enable-autorepair \
  --enable-autoupgrade \
  --release-channel=stable

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=$CLUSTER_NAME \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --machine-type=a2-highgpu-8g \
  --accelerator=type=nvidia-tesla-a100,count=8 \
  --num-nodes=0 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=16 \
  --node-taints=nvidia.com/gpu=true:NoSchedule \
  --node-labels=workload=training,gpu=a100 \
  --preemptible
```

### 2. Install NVIDIA Drivers

```bash
# Apply DaemonSet to install drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 3. Set Up GCS for Storage

```bash
# Create GCS bucket
gsutil mb -p $GCP_PROJECT -c STANDARD -l $GCP_REGION gs://ml-training-checkpoints-$RANDOM/

# Set lifecycle rules
cat <<EOF > gcs-lifecycle.json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF

gsutil lifecycle set gcs-lifecycle.json gs://ml-training-checkpoints-$RANDOM/
```

### 4. Configure Workload Identity

```bash
# Create service account
gcloud iam service-accounts create ml-training-sa \
  --project=$GCP_PROJECT \
  --display-name="ML Training Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:ml-training-sa@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Enable workload identity
kubectl annotate serviceaccount training-sa \
  -n ml-training \
  iam.gke.io/gcp-service-account=ml-training-sa@$GCP_PROJECT.iam.gserviceaccount.com
```

## On-Premise Deployment

### 1. Prepare Kubernetes Cluster

```bash
# Install Kubernetes (using kubeadm)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Configure kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install network plugin (Calico)
kubectl apply -f https://docs.projectcalico.org/manifests/tigera-operator.yaml
kubectl apply -f https://docs.projectcalico.org/manifests/custom-resources.yaml
```

### 2. Configure GPU Support

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/libnvidia-container.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### 3. Set Up MinIO for S3-Compatible Storage

```bash
# Deploy MinIO
helm repo add minio https://charts.min.io/
helm install minio minio/minio \
  --namespace minio \
  --create-namespace \
  --set mode=distributed \
  --set replicas=4 \
  --set persistence.size=1Ti \
  --set resources.requests.memory=8Gi \
  --set resources.requests.cpu=4 \
  --set accessKey=minioadmin \
  --set secretKey=minioadmin

# Create bucket for checkpoints
mc alias set minio http://minio.minio:9000 minioadmin minioadmin
mc mb minio/ml-checkpoints
```

### 4. Configure NFS for Shared Storage

```bash
# Install NFS server
sudo apt-get install nfs-kernel-server

# Configure exports
echo "/ml-training *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# Install NFS CSI driver
helm repo add csi-driver-nfs https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/master/charts
helm install csi-driver-nfs csi-driver-nfs/csi-driver-nfs \
  --namespace kube-system
```

## Multi-Cloud Deployment

### 1. Set Up Cloud Interconnect

```bash
# AWS Direct Connect
aws directconnect create-connection \
  --location "EqDC2" \
  --bandwidth 10Gbps \
  --connection-name ml-training-interconnect

# GCP Cloud Interconnect
gcloud compute interconnects attachments create ml-training-attachment \
  --region=$GCP_REGION \
  --interconnect=ml-training-interconnect \
  --vlan-id=100 \
  --bandwidth=10Gbps
```

### 2. Configure Multi-Cloud Kubernetes Federation

```bash
# Install Admiralty for multi-cluster scheduling
helm repo add admiralty https://charts.admiralty.io
helm install admiralty admiralty/admiralty \
  --namespace admiralty \
  --create-namespace \
  --set multicluster.enabled=true

# Register clusters
kubectl config use-context aws-cluster
kubectl apply -f admiralty/aws-cluster-config.yaml

kubectl config use-context gcp-cluster  
kubectl apply -f admiralty/gcp-cluster-config.yaml
```

### 3. Set Up Cross-Cloud Storage Replication

```bash
# Configure AWS DataSync for S3 to GCS
aws datasync create-task \
  --source-location-arn arn:aws:datasync:us-east-1:123456789012:location/loc-12345678 \
  --destination-location-arn arn:aws:datasync:us-east-1:123456789012:location/loc-87654321 \
  --name checkpoint-replication \
  --schedule '{"ScheduleExpression": "rate(1 hour)"}'
```

## Production Checklist

### Pre-Deployment

- [ ] **Capacity Planning**
  - [ ] Calculate required GPU hours
  - [ ] Estimate storage requirements
  - [ ] Plan for network bandwidth

- [ ] **Security Review**
  - [ ] IAM roles configured
  - [ ] Network policies defined
  - [ ] Secrets management setup
  - [ ] Encryption enabled (at-rest and in-transit)

- [ ] **Cost Controls**
  - [ ] Budget alerts configured
  - [ ] Spot instance limits set
  - [ ] Resource quotas defined

### Deployment

- [ ] **Infrastructure**
  - [ ] Kubernetes cluster deployed
  - [ ] GPU drivers installed
  - [ ] Storage configured
  - [ ] Networking optimized

- [ ] **Application**
  - [ ] Docker images built and pushed
  - [ ] Kubernetes manifests applied
  - [ ] Secrets created
  - [ ] Initial smoke test passed

- [ ] **Monitoring**
  - [ ] Prometheus deployed
  - [ ] Grafana dashboards imported
  - [ ] Alerts configured
  - [ ] Log aggregation setup

### Post-Deployment

- [ ] **Validation**
  - [ ] Multi-node training test
  - [ ] Checkpoint recovery test
  - [ ] Spot instance preemption test
  - [ ] Monitoring verification

- [ ] **Documentation**
  - [ ] Runbooks created
  - [ ] Team trained
  - [ ] Support channels established

- [ ] **Optimization**
  - [ ] Performance baseline established
  - [ ] Cost tracking enabled
  - [ ] Autoscaling configured

## Troubleshooting Deployment Issues

### Common Issues

1. **GPU Not Detected**
```bash
# Check NVIDIA drivers
kubectl exec -it <pod-name> -- nvidia-smi

# Check device plugin
kubectl get pods -n kube-system | grep nvidia
kubectl logs -n kube-system <nvidia-device-plugin-pod>
```

2. **NCCL Communication Errors**
```bash
# Check network connectivity
kubectl exec -it <pod-name> -- ping <other-pod-ip>

# Verify NCCL environment
kubectl exec -it <pod-name> -- env | grep NCCL
```

3. **Storage Performance Issues**
```bash
# Test storage throughput
kubectl exec -it <pod-name> -- fio --name=test --size=10G --rw=write --bs=1M --numjobs=1 --time_based --runtime=60
```

## Next Steps

After successful deployment:

1. Run the example training job: `kubectl apply -f examples/sample-training-job.yaml`
2. Access Grafana dashboards: `http://<grafana-url>:3000`
3. Monitor costs: Check the cost tracking dashboard
4. Scale up: Gradually increase the number of nodes

For production support, see the [Troubleshooting Guide](troubleshooting.md).