# E-commerce Datasets Guide for Visual Intelligence Pipeline

## 🛒 Available E-commerce Datasets

### **Fashion & Clothing Datasets**

#### 1. **DeepFashion2** (Recommended for Fashion)
- **Size**: 12GB, 491K images
- **Categories**: 13 fashion categories, 294 attributes
- **Annotations**: Bounding boxes, landmarks, categories, style attributes
- **Use Cases**: Fashion detection, attribute recognition, style analysis
- **Download**: https://github.com/switchablenorms/DeepFashion2
- **License**: Academic use
- **Best For**: Fashion-specific training

#### 2. **Fashion-IQ**
- **Size**: 8GB, 77K images
- **Categories**: Dress, shirt, toptee
- **Annotations**: Natural language descriptions, attributes
- **Use Cases**: Fashion search, attribute extraction
- **Download**: https://github.com/XiaoxiaoGuo/fashion-iq
- **License**: Research use
- **Best For**: Fashion search and retrieval

#### 3. **In-shop Clothes Retrieval**
- **Size**: 10GB, 52K shop images
- **Categories**: Fashion items with detailed attributes
- **Annotations**: Bounding boxes, landmarks, 1000 attributes
- **Use Cases**: Fashion retrieval, attribute classification
- **Download**: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- **License**: Academic use
- **Best For**: Fashion attribute learning

#### 4. **Polyvore Dataset**
- **Size**: 6GB, 365K items
- **Categories**: Fashion compatibility, outfit sets
- **Annotations**: Outfit compatibility, item relationships
- **Use Cases**: Outfit recommendation, compatibility analysis
- **Download**: https://github.com/xthan/polyvore-dataset
- **License**: Research use
- **Best For**: Fashion compatibility

### **General E-commerce Datasets**

#### 5. **Shopee Product Matching** (Recommended for General)
- **Size**: 15GB, 34M product titles, 70K training images
- **Categories**: Electronics, fashion, home, beauty, sports
- **Annotations**: Product IDs, categories, titles, matching groups
- **Use Cases**: Product matching, category classification
- **Download**: https://www.kaggle.com/competitions/shopee-product-matching
- **License**: Competition use
- **Best For**: Product similarity and matching

#### 6. **Amazon Berkeley Objects Dataset**
- **Size**: 20GB, 147K product images
- **Categories**: Household items, toys, electronics
- **Annotations**: 3D poses, materials, dimensions, 398 objects
- **Use Cases**: 3D understanding, material classification
- **Download**: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- **License**: Research use
- **Best For**: 3D product analysis

#### 7. **Flipkart Grid Challenge**
- **Size**: 12GB, 100K+ products
- **Categories**: Electronics, books, fashion, home
- **Annotations**: Product attributes, categories, specifications
- **Use Cases**: Attribute extraction, category classification
- **Download**: https://www.kaggle.com/competitions/flipkart-grid-challenge
- **License**: Competition use
- **Best For**: Multi-category classification

### **Specialized Datasets**

#### 8. **Fashion-MNIST** (Great for Testing)
- **Size**: 0.5GB, 70K images
- **Categories**: 10 fashion categories
- **Annotations**: Simple category labels
- **Use Cases**: Quick prototyping, algorithm testing
- **Download**: https://github.com/zalandoresearch/fashion-mnist
- **License**: MIT
- **Best For**: Fast experimentation

## 📊 Dataset Comparison Table

| Dataset | Size | Images | Categories | Annotations | Best For | Difficulty |
|---------|------|--------|------------|-------------|----------|------------|
| **DeepFashion2** | 12GB | 491K | 13 fashion | Rich | Fashion AI | Hard |
| **Fashion-IQ** | 8GB | 77K | 3 fashion | Text+Visual | Fashion Search | Medium |
| **Shopee** | 15GB | 70K | Multi-category | Product matching | General E-commerce | Medium |
| **Amazon Berkeley** | 20GB | 147K | Household | 3D poses | Product 3D | Hard |
| **Polyvore** | 6GB | 365K | Fashion | Compatibility | Outfit recommendation | Medium |
| **Fashion-MNIST** | 0.5GB | 70K | 10 fashion | Simple | Quick testing | Easy |

## 🎯 Recommended Dataset Combinations

### **For Fashion-focused Training**
```bash
Primary: DeepFashion2 (12GB)
Secondary: Fashion-IQ (8GB) 
Testing: Fashion-MNIST (0.5GB)
Total: ~20GB
```

### **For General E-commerce**
```bash
Primary: Shopee (15GB)
Secondary: Amazon Berkeley (20GB)
Testing: Fashion-MNIST (0.5GB)
Total: ~35GB
```

### **For Budget Training (<$10)**
```bash
Primary: Fashion-MNIST (0.5GB)
Secondary: Subset of Shopee (2GB)
Total: ~2.5GB
```

### **For Research/Production**
```bash
All datasets combined: ~70GB
Comprehensive training for best results
```

## 📥 Dataset Download Scripts

### Quick Start - Fashion MNIST (Testing)
```bash
#!/bin/bash
mkdir -p data/ecommerce/fashion_mnist
cd data/ecommerce/fashion_mnist

wget https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz

echo "✅ Fashion-MNIST downloaded (0.5GB)"
```

### Shopee Dataset (Kaggle)
```bash
#!/bin/bash
# Requires Kaggle API setup
pip install kaggle

# Setup Kaggle credentials first
# Place kaggle.json in ~/.kaggle/

mkdir -p data/ecommerce/shopee
cd data/ecommerce/shopee

kaggle competitions download -c shopee-product-matching
unzip shopee-product-matching.zip

echo "✅ Shopee dataset downloaded (15GB)"
```

### DeepFashion2 (Manual Download Required)
```bash
# Manual steps required:
# 1. Visit: https://github.com/switchablenorms/DeepFashion2
# 2. Fill out request form
# 3. Download links will be provided
# 4. Extract to data/ecommerce/deepfashion2/
```

## 🔧 Data Preprocessing Pipeline

### 1. **Convert to Unified Format**
```python
from ecommerce_adaptation import EcommerceDataPreprocessor

preprocessor = EcommerceDataPreprocessor("data/ecommerce")

# Process different datasets
preprocessor.process_deepfashion2("data/ecommerce/deepfashion2")
preprocessor.process_shopee_dataset("data/ecommerce/shopee")
```

### 2. **Create Training Splits**
```python
# Automatically creates train/val/test splits
# Saves unified annotations in JSON format
# Handles class balancing and data cleaning
```

### 3. **Quality Control**
```python
# Remove corrupted images
# Filter low-quality annotations
# Balance class distributions
# Generate statistics
```

## 🚀 Training Configurations

### Fashion-Specific Training
```bash
python train.py --config config/ecommerce_fashion_config.json
```

### General E-commerce Training
```bash
python train.py --config config/ecommerce_general_config.json
```

### Budget Training (Under $10)
```bash
python budget_training.py --config config/ecommerce_budget_config.json
```

## 📈 Expected Performance

### Fashion Datasets
- **Category Classification**: 85-92% accuracy
- **Attribute Recognition**: 75-85% accuracy
- **Style Classification**: 70-80% accuracy
- **Color Detection**: 90-95% accuracy

### General E-commerce
- **Product Detection**: 75-85% mAP
- **Category Classification**: 80-90% accuracy
- **Brand Recognition**: 60-75% accuracy
- **Quality Assessment**: 70-80% correlation

## 💰 Cost Analysis for Different Approaches

### Budget Approach ($5-10)
- **Datasets**: Fashion-MNIST + Small Shopee subset (2GB)
- **Training Time**: 4-8 hours
- **Expected Performance**: 70-80% accuracy
- **Use Case**: Prototyping, learning

### Standard Approach ($20-50)
- **Datasets**: DeepFashion2 + Shopee (27GB)
- **Training Time**: 24-48 hours
- **Expected Performance**: 85-90% accuracy
- **Use Case**: Production MVP

### Research Approach ($100+)
- **Datasets**: All datasets (70GB+)
- **Training Time**: 1-2 weeks
- **Expected Performance**: 90-95% accuracy
- **Use Case**: State-of-the-art results

## 🛠️ Tools and Utilities

### Dataset Verification
```bash
python ecommerce_adaptation.py --verify-datasets
```

### Statistics Generation
```bash
python ecommerce_adaptation.py --generate-stats
```

### Sample Visualization
```bash
python ecommerce_adaptation.py --visualize-samples
```

## ⚠️ Important Considerations

### **Legal and Licensing**
- Most datasets require academic/research use only
- Commercial use may require additional licensing
- Always check dataset terms before use

### **Data Quality**
- E-commerce datasets can have noisy labels
- Product images vary greatly in quality
- Background removal may be needed

### **Computational Requirements**
- Fashion datasets require more GPU memory (landmarks, attributes)
- General e-commerce datasets are more varied (harder to train)
- Budget accordingly for training time

### **Evaluation Challenges**
- E-commerce metrics differ from standard vision tasks
- User experience metrics matter more than pure accuracy
- A/B testing recommended for real deployment

## 🎯 Quick Start Recommendations

### **New to E-commerce AI**
1. Start with Fashion-MNIST (quick setup)
2. Move to DeepFashion2 subset
3. Gradually add more datasets

### **Production Timeline**
1. **Week 1**: Setup + Fashion-MNIST testing
2. **Week 2**: DeepFashion2 training
3. **Week 3**: Add Shopee dataset
4. **Week 4**: Fine-tuning and evaluation

### **Research Project**
1. **Month 1**: All fashion datasets
2. **Month 2**: General e-commerce datasets
3. **Month 3**: Multi-task training
4. **Month 4**: Evaluation and optimization

This guide provides everything needed to get started with e-commerce visual intelligence training!