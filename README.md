# Visual Intelligence Pipeline

A comprehensive deep learning system for rich image analysis using computer vision models (LLM-free).

## Overview

This project implements an end-to-end visual intelligence pipeline that performs multiple computer vision tasks:

- **Object Detection** - Detect, localize, and classify objects using custom YOLOv8-based architecture
- **Object Relationship Understanding** - Model spatial and semantic relationships using scene graphs
- **Background & Scene Analysis** - Extract scene information, lighting, colors, and environment types
- **Facial Detection & Expression Analysis** - Comprehensive facial analysis including emotions, age, gender, and gaze
- **Fine-Grained Visual Details** - Material classification, texture analysis, shadows, occlusions, and affordances

## Architecture

### Core Components

1. **`visual_intelligence_pipeline.py`** - Main pipeline orchestrator
2. **`object_detection.py`** - Custom YOLOv8 object detection with FPN
3. **`relationship_analysis.py`** - Scene graph-based relationship modeling with GNNs
4. **`scene_analysis.py`** - Scene classification and environmental analysis
5. **`facial_analysis.py`** - MTCNN-based face detection and multi-task facial analysis
6. **`fine_detail_extraction.py`** - Material, texture, lighting, and affordance analysis
7. **`training_framework.py`** - Multi-task training pipeline with mixed precision
8. **`integration_testing.py`** - Comprehensive testing and validation framework

### Model Architectures

#### Object Detection
- **Backbone**: EfficientNet-B0 with Feature Pyramid Network
- **Detection Heads**: Multi-scale detection with anchor-free design
- **Loss Functions**: Focal Loss for classification, Smooth L1 for regression

#### Relationship Analysis
- **Scene Graph Construction**: Object-centric graph representation
- **Spatial Encoding**: Relative position and geometric features
- **Graph Neural Networks**: Graph Attention Networks for relationship refinement

#### Scene Analysis
- **Scene Classification**: ResNet-50 based scene categorization (Places365)
- **Color Analysis**: K-means clustering for dominant color extraction
- **Lighting Analysis**: Multi-branch network for direction, intensity, and type estimation

#### Facial Analysis
- **Face Detection**: Multi-task CNN (MTCNN) with P-Net, R-Net, O-Net
- **Expression Analysis**: ResNet-18 for 7-emotion classification
- **Age/Gender**: Dual-branch architecture for regression and classification
- **Gaze Estimation**: 3D gaze direction prediction

#### Fine Detail Extraction
- **Material Classification**: 23-class material recognition
- **Texture Analysis**: 47-class texture classification with statistical features
- **Shadow/Occlusion**: U-Net based segmentation models
- **Affordance Detection**: Multi-label affordance prediction

## Installation

```bash
# Clone repository
git clone <repository-url>
cd visual-intelligence-pipeline

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python numpy scipy scikit-learn
pip install albumentations matplotlib seaborn
pip install torch-geometric  # For graph neural networks
pip install wandb  # For experiment tracking (optional)
pip install pytest  # For testing
```

## Quick Start

### Basic Usage

```python
from visual_intelligence_pipeline import VisualIntelligencePipeline
import cv2

# Initialize pipeline
pipeline = VisualIntelligencePipeline(device='cuda')
pipeline.initialize_models()

# Load and process image
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run complete analysis
result = pipeline.process_image(image)

# Access results
print(f"Detected {len(result.objects)} objects")
print(f"Scene category: {result.scene.scene_category}")
print(f"Found {len(result.faces)} faces")
print(f"Relationships: {len(result.relationships)}")
```

### Training Custom Models

```python
from training_framework import TrainingFramework

# Configure training
config = {
    'data_dir': 'path/to/dataset',
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'num_classes': {
        'objects': 80,
        'scenes': 365,
        'emotions': 7,
        'materials': 23,
        'textures': 47
    }
}

# Initialize and start training
trainer = TrainingFramework(config)
trainer.train()
```

## Dataset Requirements

### Data Structure
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
```

### Annotation Format
```json
{
  "image_file": "image_001.jpg",
  "objects": [
    {
      "bbox": [x1, y1, x2, y2],
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "relationships": [
    {
      "subject_id": 0,
      "object_id": 1,
      "relationship": "sitting_on"
    }
  ],
  "scene": {
    "category": "living_room",
    "indoor_outdoor": "indoor"
  },
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "emotion": "happy",
      "age": 25,
      "gender": "female"
    }
  ]
}
```

## Training Pipeline

### Multi-Task Learning
The framework supports simultaneous training of all components with:
- **Shared Backbone**: EfficientNet/ResNet features shared across tasks
- **Task-Specific Heads**: Specialized architectures for each task
- **Weighted Loss**: Configurable task weights for balanced training
- **Mixed Precision**: Automatic mixed precision for faster training

### Training Features
- **Data Augmentation**: Comprehensive augmentation pipeline with Albumentations
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Clipping**: Stable training for complex multi-task objectives
- **Early Stopping**: Automatic stopping based on validation metrics
- **Checkpoint Management**: Automatic saving and loading of best models

### Evaluation Metrics
- **Object Detection**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Scene Classification**: Top-1/Top-5 Accuracy, F1-Score
- **Facial Analysis**: Accuracy (emotion, gender), MAE (age), Angular Error (gaze)
- **Material/Texture**: Multi-class accuracy, Confusion Matrix analysis
- **Relationships**: Accuracy for subject-object-predicate triplets

## Performance Optimization

### Inference Optimization
```python
# Model optimization for deployment
import torch

# Convert to TorchScript
model_scripted = torch.jit.script(pipeline.object_detector)

# Quantization for mobile deployment
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX export for cross-platform deployment
torch.onnx.export(model, input_tensor, "model.onnx")
```

### Batch Processing
```python
# Process multiple images efficiently
images = [cv2.imread(f'image_{i}.jpg') for i in range(10)]
results = []

for image in images:
    result = pipeline.process_image(image)
    results.append(result)
```

## Testing and Validation

### Comprehensive Testing Suite
```bash
# Run all integration tests
python integration_testing.py

# Run specific test categories
python -m pytest integration_testing.py::IntegrationTester::test_object_detection_unit
```

### Test Categories
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Speed and memory benchmarks
- **Stress Tests**: Large images and continuous processing
- **Edge Cases**: Extreme conditions and error handling

### Performance Benchmarks
- **Inference Speed**: ~0.5-2.0 seconds per 640x480 image (GPU)
- **Memory Usage**: ~4-8GB GPU memory for full pipeline
- **Accuracy**: 80%+ on standard benchmarks (task-dependent)

## Model Architecture Details

### Object Detection (Custom YOLOv8)
```python
# Architecture components
- Backbone: EfficientNet-B0 (pretrained)
- Neck: Feature Pyramid Network (FPN)
- Head: Anchor-free detection with objectness, classification, regression
- Input: 640x640 RGB images
- Output: Bounding boxes, class probabilities, confidence scores
```

### Relationship Analysis (Scene Graphs + GNN)
```python
# Graph construction
- Nodes: Detected objects with visual features
- Edges: Spatial and semantic relationships
- Features: Object embeddings + spatial encoding
- GNN: Graph Attention Network for relationship refinement
```

### Scene Analysis (Multi-Branch)
```python
# Architecture branches
- Scene Classification: ResNet-50 → 365 scene categories
- Indoor/Outdoor: Binary classifier
- Lighting Analysis: Direction, intensity, type estimation
- Color Analysis: K-means clustering + color harmony
```

### Facial Analysis (MTCNN + Multi-Task)
```python
# Pipeline stages
1. Face Detection: MTCNN (P-Net → R-Net → O-Net)
2. Face Alignment: Similarity transform using landmarks
3. Expression Analysis: ResNet-18 → 7 emotions
4. Age/Gender: Dual-branch → regression + classification
5. Gaze Estimation: 3D direction vector prediction
```

## Configuration

### Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'weight_decay': 1e-5,
    'mixed_precision': True,
    'task_weights': {
        'object': 1.0,
        'relationship': 0.8,
        'scene': 0.6,
        'emotion': 0.7,
        'material': 0.4
    }
}
```

### Inference Configuration
```python
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.5,
    'max_detections': 100,
    'input_size': 640,
    'batch_size': 1
}
```

## API Reference

### Main Pipeline Class
```python
class VisualIntelligencePipeline:
    def __init__(self, device='cuda')
    def initialize_models()
    def process_image(image: np.ndarray) -> VisualAnalysisResult
```

### Result Classes
```python
@dataclass
class VisualAnalysisResult:
    objects: List[DetectionResult]
    faces: List[FaceResult] 
    scene: SceneResult
    relationships: List[RelationshipResult]
    fine_details: Dict[str, Any]
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@software{visual_intelligence_pipeline,
  title={Visual Intelligence Pipeline: LLM-Free Computer Vision System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/visual-intelligence-pipeline}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV community for computer vision utilities  
- Papers and researchers whose work influenced this implementation
- Open source datasets used for training and evaluation