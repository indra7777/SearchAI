# Step-by-Step Training Guide for Visual Intelligence Pipeline

This comprehensive guide will walk you through training the visual intelligence pipeline from scratch.

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080/4080 or better recommended)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 500GB+ available space for datasets and models
- **OS**: Linux (Ubuntu 18.04+), macOS, or Windows 10+

### Software Requirements
```bash
# Python 3.8+
python --version

# CUDA 11.8+ (for GPU training)
nvcc --version

# Required Python packages (install below)
```

## Step 1: Environment Setup

### 1.1 Create Virtual Environment
```bash
# Create conda environment
conda create -n visual_intelligence python=3.9
conda activate visual_intelligence

# OR create venv
python -m venv visual_intelligence
source visual_intelligence/bin/activate  # Linux/Mac
# visual_intelligence\Scripts\activate  # Windows
```

### 1.2 Install Dependencies
```bash
# Core ML packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Computer vision
pip install opencv-python opencv-contrib-python
pip install albumentations
pip install scikit-image

# Data processing
pip install numpy scipy pandas
pip install scikit-learn
pip install Pillow

# Graph neural networks
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Visualization and logging
pip install matplotlib seaborn
pip install wandb  # Optional for experiment tracking
pip install tensorboard

# Dataset handling
pip install pycocotools
pip install h5py

# Testing
pip install pytest pytest-cov

# Utilities
pip install tqdm
pip install psutil
pip install colorama
```

### 1.3 Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Step 2: Dataset Preparation

### 2.1 Download Required Datasets

#### Core Datasets (Required)
```bash
# Create dataset directory
mkdir -p data/datasets
cd data/datasets

# 1. COCO Dataset (Object Detection)
# Download from: https://cocodataset.org/#download
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip -d coco/images/
unzip val2017.zip -d coco/images/
unzip annotations_trainval2017.zip -d coco/

# 2. Places365 Dataset (Scene Classification)
# Download from: http://places2.csail.mit.edu/download.html
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xf places365standard_easyformat.tar

# 3. DTD Dataset (Texture Classification)
# Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
```

#### Optional Datasets (Enhanced Performance)
```bash
# FER2013 (Emotion Recognition)
# Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

# CelebA (Facial Attributes)
# Download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# Materials in Context (Material Classification)
# Download from: https://opensurfaces.cs.cornell.edu/publications/minc/

# Visual Genome (Relationships)
# Download from: https://visualgenome.org/
```

### 2.2 Verify Dataset Structure
```bash
# Check your dataset structure
ls -la data/datasets/

# Should look like:
# coco/
#   images/
#     train2017/
#     val2017/
#   annotations/
#     instances_train2017.json
#     instances_val2017.json
# places365standard_easyformat/
#   data_256/
#   places365_train_standard.txt
#   places365_val_standard.txt
# dtd/
#   images/
#   labels/
```

## Step 3: Data Preprocessing

### 3.1 Configure Dataset Paths
```bash
# Edit config/dataset_paths.json with your actual paths
nano config/dataset_paths.json
```

Update the paths to match your dataset locations:
```json
{
  "coco": "data/datasets/coco",
  "places365": "data/datasets/places365standard_easyformat",
  "dtd": "data/datasets/dtd"
}
```

### 3.2 Run Data Preprocessing
```bash
# Run preprocessing to convert all datasets to unified format
python train.py --preprocess --source-datasets config/dataset_paths.json --data-dir data/processed

# This will:
# - Convert COCO annotations to unified format
# - Process Places365 scene labels
# - Extract texture labels from DTD
# - Generate synthetic relationships
# - Create train/val/test splits
```

### 3.3 Verify Processed Data
```bash
# Check processed data structure
ls -la data/processed/

# Should contain:
# images/
#   train/
#   val/  
#   test/
# train_annotations.json
# val_annotations.json
# test_annotations.json

# Check annotation format
python -c "
import json
with open('data/processed/train_annotations.json', 'r') as f:
    data = json.load(f)
print(f'Training samples: {len(data)}')
print('Sample annotation:', json.dumps(data[0], indent=2)[:500])
"
```

## Step 4: Configuration

### 4.1 Training Configuration
Edit `config/training_config.json` for your setup:

```json
{
  "training": {
    "batch_size": 8,      // Reduce if GPU memory is limited
    "num_epochs": 50,     // Start with fewer epochs for testing
    "learning_rate": 1e-4
  },
  "hardware": {
    "device": "cuda",     // Set to "cpu" if no GPU
    "mixed_precision": true
  },
  "logging": {
    "use_wandb": false    // Set to true for advanced tracking
  }
}
```

### 4.2 Adjust for Your Hardware
```bash
# Check GPU memory
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Recommend batch size based on GPU memory
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if memory_gb >= 24:
        print('Recommended batch_size: 16-32')
    elif memory_gb >= 12:
        print('Recommended batch_size: 8-16')
    elif memory_gb >= 8:
        print('Recommended batch_size: 4-8')
    else:
        print('Recommended batch_size: 2-4')
else:
    print('No GPU detected - use CPU training with batch_size: 1-2')
"
```

## Step 5: Training

### 5.1 Start Training
```bash
# Basic training command
python train.py

# With custom configuration
python train.py --config config/training_config.json --batch-size 8 --epochs 50

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth

# Enable advanced logging
python train.py --use-wandb --project-name my-visual-ai
```

### 5.2 Monitor Training
```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check tensorboard (if enabled)
tensorboard --logdir logs/
```

### 5.3 Training Output Structure
```
checkpoints/
├── best_model.pth         # Best model based on validation loss
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── ...

logs/
├── training.log           # Training logs
├── tensorboard/          # Tensorboard logs
└── ...

outputs/
├── evaluation_report.json
├── test_summary.png
└── ...
```

## Step 6: Monitoring and Troubleshooting

### 6.1 Common Issues and Solutions

#### Out of Memory Error
```bash
# Reduce batch size
python train.py --batch-size 4

# Enable gradient checkpointing (edit training_framework.py)
# Use CPU offloading for large models
```

#### Slow Training
```bash
# Use mixed precision
python train.py --mixed-precision

# Increase number of workers
python train.py --num-workers 8

# Use smaller input size (edit config)
```

#### Poor Convergence
```bash
# Adjust learning rate
python train.py --lr 5e-5

# Change task weights in config/training_config.json
# Enable warm-up epochs
```

### 6.2 Performance Monitoring
```python
# Check training progress
import json
with open('logs/training_metrics.json', 'r') as f:
    metrics = json.load(f)

import matplotlib.pyplot as plt
epochs = list(range(len(metrics['train_loss'])))
plt.plot(epochs, metrics['train_loss'], label='Train Loss')
plt.plot(epochs, metrics['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

## Step 7: Evaluation and Testing

### 7.1 Run Evaluation
```bash
# Evaluate trained model
python integration_testing.py

# Generate comprehensive report
python -c "
from training_framework import EvaluationFramework
evaluator = EvaluationFramework({'model_paths': {'object_detector': 'checkpoints/best_model.pth'}})
# evaluator.evaluate_comprehensive(test_loader)
"
```

### 7.2 Test Individual Components
```bash
# Test object detection only
python -c "
from object_detection import ObjectDetector
import cv2
import numpy as np

detector = ObjectDetector()
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
results = detector.detect(test_image)
print(f'Detected {len(results)} objects')
"

# Test complete pipeline
python -c "
from visual_intelligence_pipeline import VisualIntelligencePipeline
import cv2

pipeline = VisualIntelligencePipeline()
pipeline.initialize_models()

# Load sample image
image = cv2.imread('sample_image.jpg')
if image is not None:
    result = pipeline.process_image(image)
    print(f'Objects: {len(result.objects)}')
    print(f'Faces: {len(result.faces)}')
    print(f'Scene: {result.scene.scene_category}')
"
```

## Step 8: Model Deployment

### 8.1 Export Models
```python
# Convert to production format
import torch
from visual_intelligence_pipeline import VisualIntelligencePipeline

# Load trained pipeline
pipeline = VisualIntelligencePipeline()
pipeline.initialize_models()

# Export to TorchScript
scripted_model = torch.jit.script(pipeline.object_detector)
scripted_model.save('models/object_detector_scripted.pt')

# Export to ONNX
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    pipeline.object_detector,
    dummy_input,
    'models/object_detector.onnx',
    export_params=True,
    opset_version=11
)
```

### 8.2 Optimization for Inference
```python
# Quantization for faster inference
import torch.quantization

model = pipeline.object_detector
model.eval()

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'models/quantized_model.pth')
```

## Step 9: Advanced Training Techniques

### 9.1 Multi-GPU Training
```bash
# Data parallel training
python -m torch.distributed.launch --nproc_per_node=2 train.py --batch-size 32

# Distributed training setup (modify training_framework.py)
```

### 9.2 Hyperparameter Tuning
```python
# Use Optuna for hyperparameter optimization
pip install optuna

# Create tuning script
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Run training with these parameters
    # Return validation accuracy
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 9.3 Custom Dataset Integration
```python
# Add your own dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        # Load your annotations
        pass
    
    def __getitem__(self, idx):
        # Return image and targets in unified format
        pass

# Integrate into training pipeline
```

## Step 10: Production Checklist

### 10.1 Model Validation
- [ ] Training converged successfully
- [ ] Validation metrics meet requirements
- [ ] Test set evaluation completed
- [ ] No overfitting detected
- [ ] All components tested individually

### 10.2 Performance Benchmarks
- [ ] Inference speed measured
- [ ] Memory usage profiled
- [ ] Batch processing tested
- [ ] Edge cases handled

### 10.3 Deployment Ready
- [ ] Models exported to production format
- [ ] Optimization applied (quantization, etc.)
- [ ] Integration tests passed
- [ ] Documentation updated

## Troubleshooting Common Issues

### Dataset Issues
```bash
# Corrupted images
find data/processed/images -name "*.jpg" -exec file {} \; | grep -v JPEG

# Missing annotations
python -c "
import json
import os
with open('data/processed/train_annotations.json') as f:
    data = json.load(f)
missing = [d['image_file'] for d in data if not os.path.exists(f'data/processed/images/train/{d[\"image_file\"]}')]
print(f'Missing images: {len(missing)}')
"
```

### Training Issues
```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop

# Debug data loading
python -c "
from torch.utils.data import DataLoader
from training_framework import MultiTaskDataset
dataset = MultiTaskDataset('data/processed', 'train')
loader = DataLoader(dataset, batch_size=1)
for batch in loader:
    print('Batch loaded successfully')
    break
"
```

### Memory Issues
```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce precision
# Edit config to use fp16 instead of fp32

# Use gradient accumulation
# Modify training loop to accumulate gradients
```

## Getting Help

1. **Check Logs**: Always start with `logs/training.log`
2. **GPU Memory**: Use `nvidia-smi` to monitor usage
3. **Dataset**: Validate with `python data_preprocessing.py --validate`
4. **Community**: Post issues with logs and system specs

## Next Steps

After successful training:
1. Fine-tune on domain-specific data
2. Experiment with different architectures
3. Deploy to production environment
4. Set up monitoring and retraining pipeline
5. Collect feedback and iterate

This completes the comprehensive training guide. The pipeline should now be ready for your specific computer vision tasks!