# Production Dockerfile for FSDP Distributed Training
# Based on NVIDIA PyTorch container for optimal GPU performance

FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    tmux \
    htop \
    iotop \
    nvtop \
    curl \
    wget \
    unzip \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    cmake \
    ninja-build \
    openssh-client \
    openssh-server \
    net-tools \
    iproute2 \
    iputils-ping \
    dnsutils \
    tcpdump \
    iftop \
    sysstat \
    strace \
    lsof \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for distributed training
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    echo "Host *\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile=/dev/null" >> /etc/ssh/ssh_config

# Upgrade pip and install Python build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install APEX for mixed precision training (optional but recommended)
RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd / && rm -rf /tmp/apex

# Install Flash Attention 2 for memory-efficient training
RUN pip install flash-attn --no-build-isolation

# Copy application code
COPY . /app

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data /app/outputs

# Set environment variables for optimized performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=0
ENV NCCL_DEBUG=INFO
ENV NCCL_DEBUG_SUBSYS=ALL
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_DISABLE=0
ENV NCCL_P2P_DISABLE=0
ENV NCCL_TREE_THRESHOLD=0
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi || exit 1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start SSH service if needed for distributed training\n\
if [ "$ENABLE_SSH" = "true" ]; then\n\
    service ssh start\n\
fi\n\
\n\
# Wait for all nodes to be ready\n\
if [ "$DISTRIBUTED" = "true" ]; then\n\
    echo "Waiting for all nodes..."\n\
    sleep 10\n\
fi\n\
\n\
# Execute command\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "examples.train_llama_fsdp"]

# Labels for metadata
LABEL maintainer="your.email@example.com"
LABEL version="1.0.0"
LABEL description="Production-ready FSDP distributed training platform"
LABEL cuda.version="12.2"
LABEL pytorch.version="2.1.0"