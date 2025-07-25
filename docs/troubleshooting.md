# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the Metaflow Distributed Training Platform.

## Table of Contents

1. [Distributed Training Issues](#distributed-training-issues)
2. [GPU and CUDA Issues](#gpu-and-cuda-issues)
3. [Networking and Communication](#networking-and-communication)
4. [Storage and Checkpointing](#storage-and-checkpointing)
5. [Memory and Performance](#memory-and-performance)
6. [Cost and Spot Instance Issues](#cost-and-spot-instance-issues)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Common Error Messages](#common-error-messages)

## Distributed Training Issues

### Training Hangs at Initialization

**Symptoms:**
- Training stuck at "Initializing distributed training"
- No progress after several minutes
- Nodes not joining the training cluster

**Diagnosis:**
```bash
# Check if all nodes can communicate
kubectl exec -it <master-pod> -n ml-training -- nslookup distributed-training-master
kubectl exec -it <worker-pod> -n ml-training -- ping distributed-training-master

# Check environment variables
kubectl exec -it <pod-name> -n ml-training -- env | grep -E "RANK|WORLD_SIZE|MASTER"

# Check NCCL debug logs
kubectl logs <pod-name> -n ml-training | grep NCCL
```

**Solutions:**

1. **Environment Variables Not Set:**
```yaml
# Ensure all pods have correct environment variables
env:
  - name: MASTER_ADDR
    value: "distributed-training-master"
  - name: MASTER_PORT
    value: "29500"
  - name: WORLD_SIZE
    value: "4"
  - name: RANK
    value: "0"  # Unique for each pod
```

2. **Network Policy Blocking Communication:**
```bash
# Check network policies
kubectl get networkpolicies -n ml-training

# Temporarily disable to test
kubectl delete networkpolicies --all -n ml-training
```

3. **DNS Resolution Issues:**
```bash
# Add explicit host entries
kubectl exec -it <pod-name> -n ml-training -- bash
echo "10.0.0.1 distributed-training-master" >> /etc/hosts
```

### Uneven GPU Memory Usage

**Symptoms:**
- Some GPUs at 100% memory while others are underutilized
- OOM errors on specific ranks
- Imbalanced training speed

**Diagnosis:**
```bash
# Monitor GPU memory across all nodes
kubectl exec -it <pod-name> -n ml-training -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Check FSDP sharding
kubectl exec -it <pod-name> -n ml-training -- python -c "
import torch.distributed as dist
print(f'Rank {dist.get_rank()} manages parameters: {sum(p.numel() for p in model.parameters())}')"
```

**Solutions:**

1. **Adjust FSDP Configuration:**
```python
# Ensure uniform sharding
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    "cpu_offload": CPUOffload(offload_params=False),  # Disable if causing imbalance
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
}
```

2. **Balance Batch Sizes:**
```python
# Use gradient accumulation to simulate larger batches
effective_batch_size = batch_size_per_gpu * gradient_accumulation_steps * world_size
```

## GPU and CUDA Issues

### CUDA Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes after certain number of steps
- Gradual memory increase

**Diagnosis:**
```python
# Add memory profiling
import torch

# Before training
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# During training
if step % 100 == 0:
    print(f"Step {step} - Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**Solutions:**

1. **Enable Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
# Or for specific layers
for layer in model.transformer.h:
    layer.use_checkpoint = True
```

2. **Reduce Batch Size Dynamically:**
```python
try:
    loss = model(batch).loss
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Reduce batch size
    batch = {k: v[:len(v)//2] for k, v in batch.items()}
    loss = model(batch).loss
```

3. **Memory-Efficient Attention:**
```python
# Use Flash Attention
from flash_attn import flash_attn_func
# Or xFormers
from xformers.ops import memory_efficient_attention
```

### GPU Not Detected

**Symptoms:**
- `nvidia-smi` command not found
- `torch.cuda.is_available()` returns False
- No GPU resources in pod

**Diagnosis:**
```bash
# Check node labels
kubectl get nodes --show-labels | grep gpu

# Check GPU operator status
kubectl get pods -n gpu-operator

# Verify device plugin
kubectl describe daemonset -n kube-system nvidia-device-plugin-daemonset
```

**Solutions:**

1. **Install NVIDIA Drivers:**
```bash
# For Ubuntu
sudo apt-get update
sudo apt-get install -y nvidia-driver-530

# Verify installation
nvidia-smi
```

2. **Fix Pod Spec:**
```yaml
resources:
  limits:
    nvidia.com/gpu: 8  # Request GPUs
  requests:
    nvidia.com/gpu: 8
```

## Networking and Communication

### NCCL Timeout Errors

**Symptoms:**
- `NCCL timeout: 1800000 ms`
- AllReduce operations hanging
- Sporadic communication failures

**Diagnosis:**
```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test network bandwidth
kubectl exec -it <pod1> -- iperf3 -s &
kubectl exec -it <pod2> -- iperf3 -c <pod1-ip>
```

**Solutions:**

1. **Increase NCCL Timeout:**
```python
import os
os.environ["NCCL_TIMEOUT"] = "3600000"  # 1 hour
os.environ["NCCL_BLOCKING_WAIT"] = "1"
```

2. **Optimize Network Configuration:**
```bash
# Enable jumbo frames
sudo ip link set dev eth0 mtu 9000

# Disable reverse path filtering
echo 0 | sudo tee /proc/sys/net/ipv4/conf/all/rp_filter
```

3. **Use NCCL Topology:**
```bash
# Set optimal NCCL topology
export NCCL_TOPOLOGY=FILE
export NCCL_TOPOLOGY_FILE=/opt/nccl-topology.xml
```

### Cross-Region Communication Issues

**Symptoms:**
- High latency between nodes
- Slow AllReduce operations
- Training throughput degradation

**Solutions:**

1. **Hierarchical AllReduce:**
```python
# Configure process groups by region
import torch.distributed as dist

# Create subgroups for intra-region communication
region_groups = {}
for region in ["us-east-1", "us-west-2"]:
    ranks = get_ranks_in_region(region)
    region_groups[region] = dist.new_group(ranks)
```

2. **Gradient Compression:**
```python
# Use PowerSGD compression
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
model.register_comm_hook(state=None, hook=powerSGD_hook)
```

## Storage and Checkpointing

### Checkpoint Save/Load Failures

**Symptoms:**
- Checkpoint corruption errors
- S3 upload timeouts
- Missing checkpoints

**Diagnosis:**
```bash
# Check storage permissions
aws s3 ls s3://your-checkpoint-bucket/

# Verify checkpoint integrity
python -c "import torch; ckpt = torch.load('checkpoint.pt'); print(ckpt.keys())"

# Monitor S3 upload
aws s3api list-multipart-uploads --bucket your-checkpoint-bucket
```

**Solutions:**

1. **Implement Retry Logic:**
```python
import time
from botocore.exceptions import ClientError

def save_checkpoint_with_retry(checkpoint, path, max_retries=3):
    for attempt in range(max_retries):
        try:
            torch.save(checkpoint, path)
            # Verify save
            torch.load(path, map_location='cpu')
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
```

2. **Use Multipart Upload:**
```python
# For large checkpoints
import boto3
from multiprocessing import Pool

def upload_multipart(bucket, key, filename):
    s3 = boto3.client('s3')
    # Implementation of multipart upload
```

### Slow Checkpoint Operations

**Symptoms:**
- Checkpoint saves taking >10 minutes
- Training stalls during checkpointing
- High I/O wait

**Solutions:**

1. **Async Checkpointing:**
```python
import threading

def async_checkpoint_save(checkpoint, path):
    thread = threading.Thread(
        target=lambda: torch.save(checkpoint, path)
    )
    thread.start()
    return thread
```

2. **Checkpoint Sharding:**
```python
# Save model shards separately
def save_sharded_checkpoint(model, optimizer, path):
    for i, param_group in enumerate(optimizer.param_groups):
        shard = {
            'params': param_group['params'],
            'state': optimizer.state,
        }
        torch.save(shard, f"{path}/shard_{i}.pt")
```

## Memory and Performance

### Memory Leak Detection

**Symptoms:**
- Gradual memory increase
- OOM after many iterations
- Degrading performance

**Diagnosis:**
```python
import gc
import psutil
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Training loop
for step in range(num_steps):
    # ... training code ...
    
    if step % 100 == 0:
        # Memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory: {current / 1e9:.2f} GB")
        print(f"Peak memory: {peak / 1e9:.2f} GB")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
```

**Solutions:**

1. **Fix Common Memory Leaks:**
```python
# Clear gradients properly
optimizer.zero_grad(set_to_none=True)

# Detach tensors when storing
metrics['loss'] = loss.detach().item()

# Delete large tensors explicitly
del intermediate_outputs
```

2. **Profile Memory Usage:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    model(batch)
    
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### Poor Training Throughput

**Symptoms:**
- Low GPU utilization
- Slow iterations
- High CPU usage

**Solutions:**

1. **Optimize Data Loading:**
```python
# Increase workers and prefetch
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=16,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
)
```

2. **Profile and Optimize:**
```python
# Find bottlenecks
with torch.profiler.profile() as prof:
    for i in range(10):
        output = model(batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        
prof.export_chrome_trace("trace.json")
# Analyze in chrome://tracing
```

## Cost and Spot Instance Issues

### Frequent Spot Instance Terminations

**Symptoms:**
- Training interrupted multiple times
- Nodes disappearing from cluster
- Cost higher than expected

**Diagnosis:**
```bash
# Check termination notices
curl -s http://169.254.169.254/latest/meta-data/spot/termination-time

# Monitor spot prices
aws ec2 describe-spot-price-history \
    --instance-types p4d.24xlarge \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
    --region us-east-1
```

**Solutions:**

1. **Implement Checkpointing Strategy:**
```python
# Checkpoint more frequently during high-risk periods
if is_spot_price_volatile():
    checkpoint_interval = 100  # More frequent
else:
    checkpoint_interval = 1000  # Normal
```

2. **Use Multiple Availability Zones:**
```yaml
# Spread across AZs
nodeSelector:
  topology.kubernetes.io/zone: us-east-1a
---
nodeSelector:
  topology.kubernetes.io/zone: us-east-1b
```

### Budget Exceeded

**Symptoms:**
- Training costs higher than estimated
- Unexpected charges
- Budget alerts triggered

**Solutions:**

1. **Implement Cost Controls:**
```python
# Stop training if budget exceeded
if cost_tracker.get_accumulated_cost() > budget_limit:
    logger.warning(f"Budget exceeded: ${cost_tracker.get_accumulated_cost()}")
    save_checkpoint()
    sys.exit(0)
```

2. **Optimize Instance Selection:**
```python
# Dynamic instance selection based on price
best_instance = cost_tracker.get_cheapest_suitable_instance()
if best_instance['savings_percent'] < 50:
    logger.info("Waiting for better spot prices...")
    time.sleep(3600)  # Wait 1 hour
```

## Monitoring and Logging

### Missing Metrics in Grafana

**Symptoms:**
- Empty dashboards
- No data points
- Prometheus targets down

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Verify metrics are being pushed
curl http://prometheus-pushgateway:9091/metrics

# Check pod annotations
kubectl describe pod <training-pod> -n ml-training | grep prometheus
```

**Solutions:**

1. **Fix Prometheus Scraping:**
```yaml
# Add annotations to pods
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: "/metrics"
```

2. **Debug Metric Collection:**
```python
# Test metric push manually
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

registry = CollectorRegistry()
g = Gauge('test_metric', 'Test metric', registry=registry)
g.set(42)
push_to_gateway('prometheus-pushgateway:9091', job='test', registry=registry)
```

### Log Aggregation Issues

**Symptoms:**
- Logs scattered across nodes
- Difficult to correlate events
- Missing log entries

**Solutions:**

1. **Centralize Logging:**
```bash
# Deploy Fluentd
kubectl apply -f https://raw.githubusercontent.com/fluent/fluentd-kubernetes-daemonset/master/fluentd-daemonset-elasticsearch-rbac.yaml

# Configure log forwarding
```

2. **Structured Logging:**
```python
import structlog

logger = structlog.get_logger()
logger.info("training_step", 
    step=1000, 
    loss=2.5, 
    rank=dist.get_rank(),
    node=os.environ.get('NODE_NAME'))
```

## Common Error Messages

### "NCCL Error: unhandled cuda error"

**Cause:** CUDA error during collective operation

**Fix:**
```python
# Enable better CUDA error checking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
```

### "RuntimeError: Expected all tensors to be on the same device"

**Cause:** Mixed CPU/GPU tensors

**Fix:**
```python
# Ensure all tensors on same device
device = torch.cuda.current_device()
batch = {k: v.to(device) for k, v in batch.items()}
```

### "OSError: [Errno 28] No space left on device"

**Cause:** Disk full

**Fix:**
```bash
# Clean up old checkpoints
find /checkpoints -name "*.pt" -mtime +7 -delete

# Increase volume size
kubectl patch pvc checkpoint-storage-pvc -n ml-training -p '{"spec":{"resources":{"requests":{"storage":"1Ti"}}}}'
```

### "torch.distributed.DistBackendError: NCCL error in: ../torch/lib/c10d/ProcessGroupNCCL.cpp"

**Cause:** NCCL initialization failure

**Fix:**
```python
# Set proper network interface
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_IB_DISABLE'] = '1'  # If not using InfiniBand
```

## Performance Debugging Checklist

When experiencing performance issues:

1. **Profile GPU Utilization:**
```bash
nvidia-smi dmon -s pucvmet -i 0 -f gpu_profile.log
```

2. **Check Network Bandwidth:**
```bash
# Between nodes
iperf3 -s  # On one node
iperf3 -c <other-node-ip> -t 30  # On another
```

3. **Monitor I/O:**
```bash
iostat -x 1
iotop
```

4. **Analyze Training Profile:**
```python
# Use PyTorch profiler
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    with_stack=True
) as prof:
    train_loop()
```

## Getting Help

If issues persist:

1. Collect diagnostic information:
```bash
kubectl cluster-info dump --namespace ml-training > cluster-dump.tar.gz
```

2. Check logs:
```bash
kubectl logs -n ml-training -l app=distributed-training --tail=1000 > training-logs.txt
```

3. Contact support with:
   - Cluster dump
   - Training logs
   - Configuration files
   - Error messages

For real-time support, join our Slack channel: #ml-platform-support