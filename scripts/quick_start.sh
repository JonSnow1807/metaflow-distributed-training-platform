#!/bin/bash
# Quick start script for Metaflow Distributed Training Platform
# This script sets up the environment and runs initial checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          Metaflow Distributed Training Platform               ║"
echo "║                    Quick Start Setup                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
REQUIRED_VERSION="3.8"

if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc) -eq 1 ]; then
    print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
else
    print_error "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
    print_success "CUDA $CUDA_VERSION found"
    
    # Show GPU info
    echo -e "\n${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
        echo "  • $line"
    done
else
    print_warning "CUDA not found - GPU training will not be available"
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_success "Docker $DOCKER_VERSION found"
else
    print_warning "Docker not found - containerized deployment will not be available"
fi

# Check kubectl
if command -v kubectl &> /dev/null; then
    KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
    print_success "kubectl $KUBECTL_VERSION found"
else
    print_warning "kubectl not found - Kubernetes deployment will not be available"
fi

# Check AWS CLI
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_success "AWS CLI $AWS_VERSION found"
    
    # Check AWS credentials
    if aws sts get-caller-identity &> /dev/null; then
        AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
        print_success "AWS credentials configured (Account: $AWS_ACCOUNT)"
    else
        print_warning "AWS credentials not configured - run 'aws configure'"
    fi
else
    print_warning "AWS CLI not found - cloud features will be limited"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install package in development mode
print_status "Installing package in development mode..."
pip install -e . --quiet
print_success "Package installed"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data checkpoints logs outputs results monitoring/dashboards
print_success "Directories created"

# Run initial tests
print_status "Running smoke tests..."
if python -m pytest tests/test_distributed_training.py::TestFSDPTrainer::test_trainer_initialization -v --tb=short > /dev/null 2>&1; then
    print_success "Smoke tests passed"
else
    print_warning "Some tests failed - check logs for details"
fi

# Check GPU functionality
if command -v nvidia-smi &> /dev/null; then
    print_status "Testing GPU functionality..."
    python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        print_success "PyTorch can access $GPU_COUNT GPU(s)"
    else
        print_warning "PyTorch cannot access GPUs"
    fi
fi

# Download sample configuration
print_status "Setting up sample configuration..."
if [ ! -f "configs/sample_training.yaml" ]; then
    cat > configs/sample_training.yaml << 'EOF'
# Sample training configuration
model:
  name: "bert-base-uncased"
  type: "masked_lm"

training:
  batch_size_per_gpu: 8
  learning_rate: 3.0e-4
  num_train_epochs: 3
  warmup_steps: 1000

distributed:
  backend: "nccl"
  world_size: 1

optimization:
  gradient_checkpointing: true
  mixed_precision: true
EOF
    print_success "Sample configuration created"
fi

# Set up git hooks if in a git repository
if [ -d ".git" ]; then
    print_status "Setting up git hooks..."
    if command -v pre-commit &> /dev/null; then
        pre-commit install --quiet
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found - install with 'pip install pre-commit'"
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << 'EOF'
# Environment variables for development
WANDB_API_KEY=your_wandb_key_here
AWS_PROFILE=default
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
EOF
    print_success ".env file created (update with your API keys)"
fi

# Show next steps
echo -e "\n${GREEN}✨ Setup complete!${NC}\n"

echo -e "${BLUE}Quick Start Commands:${NC}"
echo "1. Check spot prices:"
echo "   python scripts/check_spot_prices.py"
echo ""
echo "2. Estimate training cost:"
echo "   python scripts/estimate_cost.py -m 7B -d 100"
echo ""
echo "3. Start local training:"
echo "   python examples/train_llama_fsdp.py run --num-nodes 1"
echo ""
echo "4. Deploy to Kubernetes:"
echo "   kubectl apply -f kubernetes/"
echo ""
echo "5. Monitor training:"
echo "   make monitor"

echo -e "\n${BLUE}Useful Resources:${NC}"
echo "• Documentation: docs/"
echo "• Examples: examples/"
echo "• Configuration: configs/"
echo "• Troubleshooting: docs/troubleshooting.md"

echo -e "\n${YELLOW}Tips:${NC}"
echo "• Use 'make help' to see all available commands"
echo "• Run 'make test' to ensure everything is working"
echo "• Check docs/deployment_guide.md for production setup"

# Final checks and recommendations
echo -e "\n${BLUE}System Status:${NC}"

# Memory check
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ $TOTAL_MEM -lt 32 ]; then
    print_warning "System has ${TOTAL_MEM}GB RAM - recommend at least 32GB for training"
else
    print_success "System has ${TOTAL_MEM}GB RAM"
fi

# Disk space check
DISK_FREE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ $DISK_FREE -lt 100 ]; then
    print_warning "Only ${DISK_FREE}GB free disk space - recommend at least 100GB"
else
    print_success "${DISK_FREE}GB free disk space available"
fi

# Network connectivity check
if ping -c 1 google.com &> /dev/null; then
    print_success "Internet connectivity verified"
else
    print_warning "No internet connectivity detected"
fi

echo -e "\n${GREEN}Ready to train! 🚀${NC}"