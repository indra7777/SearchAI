#!/bin/bash

# RunPod RTX 4090 Setup Script for Visual Intelligence Pipeline
# Run this script after connecting to your RunPod instance

set -e  # Exit on any error

echo "🚀 Setting up Visual Intelligence Pipeline on RunPod RTX 4090..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're on RunPod
if [[ ! -d "/workspace" ]]; then
    print_warning "This script is designed for RunPod. Proceeding anyway..."
fi

# Step 1: Update system and install dependencies
print_status "Updating system packages..."
apt update && apt upgrade -y

print_status "Installing system dependencies..."
apt install -y wget curl git vim htop tree unzip p7zip-full

# Step 2: Verify CUDA and PyTorch
print_status "Verifying CUDA and PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    print_error "PyTorch not found. Installing..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if python3 -c "import torch; torch.cuda.is_available()" | grep -q "True"; then
    print_success "CUDA is available"
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "CUDA not available!"
    exit 1
fi

# Step 3: Install Python packages
print_status "Installing Python packages..."

# Core packages
pip install --upgrade pip

# Computer vision packages
pip install opencv-python opencv-contrib-python
pip install albumentations
pip install scikit-image scikit-learn
pip install Pillow

# Dataset handling
pip install pycocotools
pip install h5py

# Graph neural networks (for relationship analysis)
print_status "Installing PyTorch Geometric..."
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Visualization and logging
pip install matplotlib seaborn
pip install wandb
pip install tensorboard

# Utilities
pip install tqdm psutil colorama
pip install pytest pytest-cov

# Verify critical installations
print_status "Verifying package installations..."
python3 -c "
try:
    import cv2, albumentations, torch_geometric, wandb
    print('✅ All critical packages installed successfully')
except ImportError as e:
    print(f'❌ Package import failed: {e}')
    exit(1)
"

# Step 4: Setup project structure
print_status "Setting up project structure..."
cd /workspace

# Create project directory
PROJECT_DIR="/workspace/visual-intelligence-pipeline"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create subdirectories
mkdir -p data/{datasets,processed}
mkdir -p {checkpoints,logs,outputs,config}
mkdir -p models/{exported,quantized}

print_success "Project structure created at $PROJECT_DIR"

# Step 5: Create optimized configuration for RTX 4090
print_status "Creating RTX 4090 optimized configuration..."

cat > config/runpod_training_config.json << 'EOF'
{
  "experiment_name": "visual_intelligence_runpod_4090",
  "data_dir": "data/processed",
  "checkpoint_dir": "checkpoints",
  "output_dir": "outputs",
  "log_dir": "logs",
  
  "model_config": {
    "backbone": "efficientnet-b0",
    "input_size": 640,
    "feature_pyramid": true,
    "mixed_precision": true
  },
  
  "training": {
    "batch_size": 24,
    "num_epochs": 100,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "gradient_clip_norm": 1.0
  },
  
  "data": {
    "num_workers": 8,
    "pin_memory": true,
    "image_size": 640,
    "augmentation": {
      "horizontal_flip": 0.5,
      "vertical_flip": 0.1,
      "rotation": 15,
      "brightness": 0.2,
      "contrast": 0.2,
      "saturation": 0.2,
      "hue": 0.1,
      "gaussian_blur": 0.1,
      "gaussian_noise": 0.1
    }
  },
  
  "num_classes": {
    "objects": 80,
    "relationships": 50,
    "scenes": 365,
    "emotions": 7,
    "materials": 23,
    "textures": 47,
    "lighting_conditions": 8,
    "genders": 2
  },
  
  "task_weights": {
    "object": 1.0,
    "relationship": 0.8,
    "scene": 0.6,
    "emotion": 0.7,
    "age": 0.5,
    "gender": 0.6,
    "material": 0.4,
    "texture": 0.4,
    "lighting": 0.3,
    "shadow": 0.2,
    "occlusion": 0.2
  },
  
  "validation": {
    "frequency": 1,
    "metrics": ["mAP", "accuracy", "f1_score", "mae"]
  },
  
  "checkpointing": {
    "save_frequency": 5,
    "save_best": true,
    "max_checkpoints": 3
  },
  
  "early_stopping": {
    "patience": 15,
    "min_delta": 1e-4,
    "monitor": "val_total_loss"
  },
  
  "logging": {
    "log_frequency": 50,
    "use_wandb": false,
    "wandb_project": "visual-intelligence-runpod",
    "save_images": false,
    "save_predictions": false
  },
  
  "hardware": {
    "device": "cuda",
    "mixed_precision": true,
    "compile_model": false,
    "channels_last": true
  }
}
EOF

cat > config/dataset_paths.json << 'EOF'
{
  "coco": "data/datasets/coco",
  "places365": "data/datasets/places365standard_easyformat",
  "dtd": "data/datasets/dtd"
}
EOF

print_success "Configuration files created"

# Step 6: Create download script for datasets
print_status "Creating dataset download script..."

cat > download_datasets.sh << 'EOF'
#!/bin/bash

# Dataset download script for RunPod
set -e

echo "📥 Downloading datasets..."

cd data/datasets

# COCO Dataset
echo "Downloading COCO dataset (this will take a while)..."
mkdir -p coco/{images,annotations}

# Download COCO images and annotations
wget -c -P coco/ http://images.cocodataset.org/zips/train2017.zip
wget -c -P coco/ http://images.cocodataset.org/zips/val2017.zip
wget -c -P coco/ http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract COCO
echo "Extracting COCO dataset..."
cd coco
unzip -q train2017.zip -d images/ && echo "✅ COCO train images extracted"
unzip -q val2017.zip -d images/ && echo "✅ COCO val images extracted"
unzip -q annotations_trainval2017.zip && echo "✅ COCO annotations extracted"
rm *.zip  # Save space
cd ..

# Places365 Dataset (smaller subset)
echo "Downloading Places365 dataset..."
wget -c http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xf places365standard_easyformat.tar && echo "✅ Places365 extracted"
rm places365standard_easyformat.tar

# DTD Dataset
echo "Downloading DTD dataset..."
wget -c https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz && echo "✅ DTD extracted"
rm dtd-r1.0.1.tar.gz

echo "✅ All datasets downloaded and extracted"
echo "Dataset sizes:"
du -sh */

cd ../..
EOF

chmod +x download_datasets.sh

# Step 7: Create monitoring script
print_status "Creating monitoring script..."

cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3

import time
import subprocess
import json
import psutil
import os
from datetime import datetime

try:
    import GPUtil
except ImportError:
    os.system("pip install gputil")
    import GPUtil

def get_gpu_stats():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'temperature': gpu.temperature
            }
    except:
        pass
    return None

def check_training_process():
    try:
        result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True, text=True)
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        return 0

def monitor_training():
    print("🔍 Starting training monitor...")
    print("Press Ctrl+C to stop monitoring")
    
    start_time = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            print(f"\n{'='*60}")
            print(f"⏰ Time: {current_time.strftime('%H:%M:%S')} | Elapsed: {str(elapsed).split('.')[0]}")
            
            # GPU stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print(f"🎮 GPU Util: {gpu_stats['utilization']:.1f}% | "
                      f"Memory: {gpu_stats['memory_used']}/{gpu_stats['memory_total']}MB "
                      f"({gpu_stats['memory_percent']:.1f}%) | "
                      f"Temp: {gpu_stats['temperature']}°C")
            
            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/workspace')
            
            print(f"💻 CPU: {cpu_percent:.1f}% | "
                  f"RAM: {memory.percent:.1f}% ({memory.used//1024//1024//1024}GB/"
                  f"{memory.total//1024//1024//1024}GB)")
            print(f"💾 Disk: {disk.percent:.1f}% ({disk.used//1024//1024//1024}GB/"
                  f"{disk.total//1024//1024//1024}GB)")
            
            # Training process check
            training_processes = check_training_process()
            if training_processes > 0:
                print(f"🏃 Training processes running: {training_processes}")
                
                # Check latest log
                if os.path.exists('logs/training.log'):
                    try:
                        with open('logs/training.log', 'r') as f:
                            lines = f.readlines()
                            if lines:
                                latest_line = lines[-1].strip()
                                if latest_line:
                                    print(f"📝 Latest log: {latest_line[-100:]}")
                    except:
                        pass
            else:
                print("⚠️  No training process detected")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training()
EOF

chmod +x monitor_training.py

# Step 8: Create quick test script
print_status "Creating test script..."

cat > test_setup.py << 'EOF'
#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import sys
import os

def test_pytorch_gpu():
    print("🔬 Testing PyTorch GPU setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU computation
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.T)
            print("✅ GPU computation test passed")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def test_opencv():
    print("\n🔬 Testing OpenCV...")
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        # Test basic OpenCV operations
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        print("✅ OpenCV test passed")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_packages():
    print("\n🔬 Testing package imports...")
    packages = [
        'albumentations',
        'torch_geometric', 
        'sklearn',
        'matplotlib',
        'tqdm',
        'psutil'
    ]
    
    failed_packages = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    return len(failed_packages) == 0

def main():
    print("🧪 Running setup tests...\n")
    
    tests = [
        ("PyTorch GPU", test_pytorch_gpu),
        ("OpenCV", test_opencv), 
        ("Package Imports", test_packages)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for training.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_setup.py

# Step 9: Create quick start script
print_status "Creating quick start script..."

cat > start_training.sh << 'EOF'
#!/bin/bash

# Quick start training script for RunPod
set -e

echo "🚀 Starting Visual Intelligence Pipeline Training on RunPod RTX 4090"

# Check if datasets exist
if [ ! -d "data/datasets/coco" ] || [ ! -d "data/datasets/places365standard_easyformat" ]; then
    echo "📥 Datasets not found. Downloading..."
    ./download_datasets.sh
fi

# Check if data is preprocessed
if [ ! -f "data/processed/train_annotations.json" ]; then
    echo "🔄 Preprocessing data..."
    python train.py --preprocess --source-datasets config/dataset_paths.json --data-dir data/processed
fi

# Start training
echo "🏋️ Starting training..."
python train.py \
    --config config/runpod_training_config.json \
    --batch-size 24 \
    --epochs 100 \
    --lr 2e-4 \
    --mixed-precision \
    --num-workers 8

echo "✅ Training completed!"
EOF

chmod +x start_training.sh

# Step 10: Final setup
print_status "Running setup tests..."
python3 test_setup.py

# Create helpful README
cat > RUNPOD_README.md << 'EOF'
# Visual Intelligence Pipeline - RunPod RTX 4090 Setup

## Quick Start

1. **Download datasets** (first time only):
   ```bash
   ./download_datasets.sh
   ```

2. **Start training**:
   ```bash
   ./start_training.sh
   ```

3. **Monitor training** (in another terminal):
   ```bash
   python monitor_training.py
   ```

## Manual Steps

1. **Test setup**:
   ```bash
   python test_setup.py
   ```

2. **Download specific datasets**:
   ```bash
   cd data/datasets
   # Download COCO, Places365, DTD as needed
   ```

3. **Preprocess data**:
   ```bash
   python train.py --preprocess --source-datasets config/dataset_paths.json
   ```

4. **Start training with custom settings**:
   ```bash
   python train.py --config config/runpod_training_config.json --batch-size 24 --epochs 100
   ```

## Important Files

- `config/runpod_training_config.json` - RTX 4090 optimized training config
- `download_datasets.sh` - Downloads required datasets
- `monitor_training.py` - Monitors training progress
- `start_training.sh` - One-click training start
- `test_setup.py` - Tests if setup is working

## Monitoring

Watch training logs:
```bash
tail -f logs/training.log
```

Check GPU usage:
```bash
nvidia-smi -l 1
```

## Estimated Training Time

- Full training: ~24-48 hours on RTX 4090
- Quick test (5 epochs): ~2-4 hours
- Dataset download: ~2-4 hours (depends on internet)

## Storage Usage

- Datasets: ~50-100GB
- Models/checkpoints: ~10-20GB
- Total recommended: 200GB+

## Cost Estimation (RunPod)

- RTX 4090: ~$0.50-1.00/hour
- 48 hours training: ~$25-50
- Use spot instances for lower cost
EOF

print_success "Setup completed successfully!"

echo ""
echo "🎉 Visual Intelligence Pipeline is ready for training on RunPod RTX 4090!"
echo ""
echo "📋 Next steps:"
echo "   1. Copy your Python training files to this directory"
echo "   2. Run: ./download_datasets.sh (first time only)"  
echo "   3. Run: ./start_training.sh"
echo "   4. Monitor with: python monitor_training.py"
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo "💾 Available space: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""
print_success "Happy training! 🚀"