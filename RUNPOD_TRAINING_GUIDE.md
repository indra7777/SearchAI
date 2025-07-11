# Training Visual Intelligence Pipeline on RunPod RTX 4090

This guide covers training the visual intelligence pipeline specifically on RunPod with RTX 4090 GPU.

## Step 1: RunPod Setup

### 1.1 Create RunPod Account and Pod
1. Go to [RunPod.io](https://www.runpod.io/)
2. Create account and add payment method
3. Navigate to "Pods" → "GPU Pods"
4. Select RTX 4090 (24GB VRAM)
5. Choose template: **PyTorch 2.1** or **Custom**
6. Set storage: **100GB minimum** (200GB+ recommended)
7. Deploy pod

### 1.2 Recommended Pod Configuration
```
GPU: RTX 4090 (24GB VRAM)
vCPU: 12+ cores
RAM: 32GB+
Storage: 200GB (for datasets + models)
Template: PyTorch 2.1.0 (Python 3.10, CUDA 12.1)
```

### 1.3 Connect to Pod
```bash
# Option 1: Web Terminal (in RunPod interface)
# Option 2: SSH (copy SSH command from pod)
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa

# Option 3: Jupyter Lab (available in pod interface)
```

## Step 2: Initial Pod Setup

### 2.1 Update System and Install Dependencies
```bash
# Update system
apt update && apt upgrade -y

# Install system dependencies
apt install -y wget curl git vim htop tree unzip

# Verify CUDA and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
nvidia-smi
```

### 2.2 Install Additional Python Packages
```bash
# Core packages (if not already installed)
pip install opencv-python opencv-contrib-python
pip install albumentations
pip install scikit-image scikit-learn
pip install pycocotools
pip install wandb

# Graph neural networks
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Utilities
pip install tqdm psutil colorama
pip install matplotlib seaborn

# Verify installations
python -c "import cv2, albumentations, torch_geometric; print('All packages installed successfully')"
```

## Step 3: Clone Repository and Setup

### 3.1 Clone Your Repository
```bash
# If using GitHub
git clone https://github.com/yourusername/visual-intelligence-pipeline.git
cd visual-intelligence-pipeline

# OR upload files directly
mkdir -p /workspace/visual-intelligence-pipeline
cd /workspace/visual-intelligence-pipeline

# Copy all the Python files we created earlier
```

### 3.2 Setup Project Structure
```bash
# Create necessary directories
mkdir -p data/datasets data/processed
mkdir -p checkpoints logs outputs config

# Verify structure
tree -L 2
```

## Step 4: Dataset Download and Preparation

### 4.1 Download Datasets (Optimized for RunPod)
```bash
cd data/datasets

# 1. COCO Dataset (~20GB)
echo "Downloading COCO dataset..."
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip  
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract with progress
mkdir -p coco/images coco/annotations
unzip -q train2017.zip -d coco/images/ && echo "Train images extracted"
unzip -q val2017.zip -d coco/images/ && echo "Val images extracted"  
unzip -q annotations_trainval2017.zip -d coco/ && echo "Annotations extracted"

# Cleanup zip files to save space
rm *.zip

# 2. Places365 Dataset (smaller subset for testing)
echo "Downloading Places365 dataset..."
wget -c http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xf places365standard_easyformat.tar && rm places365standard_easyformat.tar

# 3. DTD Dataset (~600MB)
echo "Downloading DTD dataset..."
wget -c https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz && rm dtd-r1.0.1.tar.gz

# Verify downloads
echo "Dataset sizes:"
du -sh */
```

### 4.2 Verify Dataset Structure
```bash
# Check structure
ls -la coco/
ls -la places365standard_easyformat/
ls -la dtd/

echo "COCO images: $(find coco/images -name "*.jpg" | wc -l)"
echo "Places365 images: $(find places365standard_easyformat -name "*.jpg" | wc -l)"
echo "DTD images: $(find dtd/images -name "*.jpg" | wc -l)"
```

## Step 5: Optimized Configuration for RTX 4090

### 5.1 Create RunPod-Optimized Config
```bash
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
    "use_wandb": true,
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
```

### 5.2 Update Dataset Paths
```bash
cat > config/dataset_paths.json << 'EOF'
{
  "coco": "data/datasets/coco",
  "places365": "data/datasets/places365standard_easyformat", 
  "dtd": "data/datasets/dtd"
}
EOF
```

## Step 6: Data Preprocessing

### 6.1 Run Preprocessing with Progress Monitoring
```bash
# Start preprocessing
echo "Starting data preprocessing..."
python train.py \
    --preprocess \
    --source-datasets config/dataset_paths.json \
    --data-dir data/processed \
    --num-workers 8

# Monitor preprocessing progress
echo "Preprocessing completed. Checking results..."
ls -la data/processed/
echo "Training samples: $(python -c "import json; print(len(json.load(open('data/processed/train_annotations.json'))))")"
echo "Validation samples: $(python -c "import json; print(len(json.load(open('data/processed/val_annotations.json'))))")"
echo "Test samples: $(python -c "import json; print(len(json.load(open('data/processed/test_annotations.json'))))")"
```

## Step 7: Setup Monitoring and Logging

### 7.1 Configure Weights & Biases (Optional but Recommended)
```bash
# Install and login to WandB
pip install wandb
wandb login  # Enter your API key from wandb.ai

# OR run without WandB
# Edit config to set "use_wandb": false
```

### 7.2 Setup Monitoring Script
```bash
cat > monitor_training.py << 'EOF'
import time
import subprocess
import json
import psutil
import GPUtil

def monitor_training():
    while True:
        try:
            # GPU stats
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU Util: {gpu.load*100:.1f}% | Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            
            # CPU and RAM
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            print(f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")
            
            # Check if training is still running
            result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True)
            if result.returncode != 0:
                print("Training process not found. Stopping monitor.")
                break
                
            time.sleep(30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training()
EOF
```

## Step 8: Start Training

### 8.1 Pre-Training Checks
```bash
# Check GPU status
nvidia-smi

# Test data loading
python -c "
from torch.utils.data import DataLoader
from training_framework import MultiTaskDataset
import torch

dataset = MultiTaskDataset('data/processed', 'train')
loader = DataLoader(dataset, batch_size=2, num_workers=2)
print(f'Dataset size: {len(dataset)}')

# Test one batch
for batch in loader:
    print('Batch loaded successfully!')
    print(f'Image shape: {batch[0].shape}')
    break
"

# Check available disk space
df -h /
```

### 8.2 Start Training with Optimal Settings
```bash
# Start training in background with logging
nohup python train.py \
    --config config/runpod_training_config.json \
    --batch-size 24 \
    --epochs 100 \
    --lr 2e-4 \
    --num-workers 8 \
    --mixed-precision \
    --use-wandb \
    --project-name visual-intelligence-runpod \
    > training.out 2>&1 &

# Get process ID
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"

# Start monitoring in another terminal
python monitor_training.py &
```

### 8.3 Monitor Training Progress
```bash
# Watch training logs
tail -f training.out

# Watch system resources
watch -n 2 'nvidia-smi && echo "" && df -h'

# Check training metrics
tail -f logs/training.log | grep -E "(Epoch|Loss|Accuracy)"
```

## Step 9: Training Management

### 9.1 Useful Training Commands
```bash
# Check training status
ps aux | grep train.py

# Stop training gracefully
kill -TERM $(cat training.pid)

# Resume from checkpoint (if training stops)
python train.py \
    --config config/runpod_training_config.json \
    --resume checkpoints/best_model.pth \
    --batch-size 24

# Quick validation run (test setup)
python train.py \
    --config config/runpod_training_config.json \
    --batch-size 8 \
    --epochs 2 \
    --lr 1e-4
```

### 9.2 Save Training Progress
```bash
# Create backup script
cat > backup_training.sh << 'EOF'
#!/bin/bash
# Backup training progress
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_${DATE}"

mkdir -p ${BACKUP_DIR}
cp -r checkpoints ${BACKUP_DIR}/
cp -r logs ${BACKUP_DIR}/
cp -r config ${BACKUP_DIR}/
cp training.out ${BACKUP_DIR}/

echo "Backup created: ${BACKUP_DIR}"
echo "Size: $(du -sh ${BACKUP_DIR})"
EOF

chmod +x backup_training.sh

# Run backup
./backup_training.sh
```

## Step 10: Optimization for RTX 4090

### 10.1 Memory Optimization
```python
# Add to training script for better memory usage
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.95)

# Enable cudnn benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### 10.2 Performance Tuning
```bash
# Optimal batch sizes for RTX 4090 (24GB):
# Input 224x224: batch_size = 64-128
# Input 640x640: batch_size = 16-32
# Our multi-task: batch_size = 20-24

# Edit config for best performance
python -c "
import json
with open('config/runpod_training_config.json', 'r') as f:
    config = json.load(f)

# RTX 4090 optimized settings
config['training']['batch_size'] = 24
config['data']['num_workers'] = 8
config['hardware']['channels_last'] = True
config['training']['gradient_clip_norm'] = 1.0

with open('config/runpod_training_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Config updated for RTX 4090')
"
```

## Step 11: Cost Optimization on RunPod

### 11.1 Save Costs
```bash
# Stop pod when not training
# RunPod charges per second, so stop when done

# Use spot instances if available (cheaper but can be interrupted)

# Estimate training time
echo "Estimated training time calculation:"
python -c "
# Rough estimates for RTX 4090
samples_per_epoch = 10000  # Adjust based on your dataset
batch_size = 24
epochs = 100
seconds_per_batch = 2  # Estimate

batches_per_epoch = samples_per_epoch / batch_size
total_batches = batches_per_epoch * epochs
total_seconds = total_batches * seconds_per_batch
total_hours = total_seconds / 3600

print(f'Estimated training time: {total_hours:.1f} hours')
print(f'At $0.50/hour: ~${total_hours * 0.5:.2f}')
"
```

### 11.2 Efficient Data Management
```bash
# Compress checkpoints to save space
python -c "
import torch
import os

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')

# Remove optimizer states to reduce size (if not resuming)
if 'optimizers' in checkpoint:
    del checkpoint['optimizers']
if 'schedulers' in checkpoint:
    del checkpoint['schedulers']

# Save compressed
torch.save(checkpoint, 'checkpoints/best_model_compressed.pth')
print('Checkpoint compressed')
"

# Clean up unnecessary files
find data/datasets -name "*.zip" -delete
find data/datasets -name "*.tar*" -delete
```

## Step 12: Troubleshooting RunPod Issues

### 12.1 Common RunPod Issues
```bash
# Out of storage space
df -h
du -sh * | sort -hr  # Find large directories
rm -rf data/datasets/*.zip  # Remove zip files

# Network issues
ping google.com
wget --spider http://google.com

# CUDA issues
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Pod connection issues
# Use web terminal if SSH fails
# Check firewall settings in RunPod
```

### 12.2 Performance Issues
```bash
# Check GPU utilization
nvidia-smi -l 1

# If GPU utilization is low:
# 1. Increase batch size
# 2. Increase num_workers
# 3. Check data loading bottlenecks

# Memory issues
# Reduce batch size
# Enable gradient checkpointing
# Use mixed precision training
```

## Step 13: Downloading Results

### 13.1 Download Trained Models
```bash
# Compress results for download
tar -czf training_results.tar.gz \
    checkpoints/ \
    logs/ \
    outputs/ \
    config/

# Check size
ls -lh training_results.tar.gz

# Download via RunPod file manager or SCP
# scp root@<pod-ip>:/workspace/visual-intelligence-pipeline/training_results.tar.gz ./
```

### 13.2 Quick Model Test
```bash
# Test trained model quickly
python -c "
from visual_intelligence_pipeline import VisualIntelligencePipeline
import numpy as np
import time

print('Loading pipeline...')
pipeline = VisualIntelligencePipeline()
pipeline.initialize_models()

# Test with random image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print('Running inference...')
start_time = time.time()
result = pipeline.process_image(test_image)
inference_time = time.time() - start_time

print(f'Inference time: {inference_time:.3f}s')
print(f'Objects detected: {len(result.objects)}')
print(f'Faces detected: {len(result.faces)}')
print(f'Scene category: {result.scene.scene_category}')
print('Model working correctly!')
"
```

## Complete RunPod Training Command

```bash
# Single command to start everything
python train.py \
    --config config/runpod_training_config.json \
    --data-dir data/processed \
    --batch-size 24 \
    --epochs 100 \
    --lr 2e-4 \
    --weight-decay 1e-5 \
    --mixed-precision \
    --num-workers 8 \
    --use-wandb \
    --project-name visual-intelligence-runpod \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

This guide should get you training successfully on RunPod RTX 4090! The RTX 4090's 24GB VRAM allows for larger batch sizes and faster training compared to smaller GPUs.