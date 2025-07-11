#!/usr/bin/env python3
"""
E-commerce Dataset Adaptation for Visual Intelligence Pipeline
Specialized for product analysis, fashion, retail, and shopping applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import requests
from urllib.parse import urlparse

@dataclass
class EcommerceProduct:
    """E-commerce product analysis result"""
    product_id: str
    category: str
    subcategory: str
    attributes: Dict[str, Any]
    colors: List[str]
    materials: List[str]
    price_range: str
    style_tags: List[str]
    quality_score: float
    bbox: Tuple[float, float, float, float]

class EcommerceDatasetManager:
    """Manager for popular e-commerce datasets"""
    
    def __init__(self):
        self.datasets = {
            'deepfashion2': {
                'name': 'DeepFashion2',
                'url': 'https://github.com/switchablenorms/DeepFashion2',
                'description': 'Large-scale fashion dataset with 491K images',
                'categories': ['fashion', 'clothing', 'accessories'],
                'annotations': ['bbox', 'landmarks', 'categories', 'attributes'],
                'size_gb': 12
            },
            'fashioniq': {
                'name': 'Fashion-IQ',
                'url': 'https://github.com/XiaoxiaoGuo/fashion-iq',
                'description': 'Fashion image retrieval with natural language',
                'categories': ['dress', 'shirt', 'toptee'],
                'annotations': ['categories', 'captions', 'attributes'],
                'size_gb': 8
            },
            'shopee': {
                'name': 'Shopee Product Detection',
                'url': 'https://www.kaggle.com/competitions/shopee-product-matching',
                'description': 'Product matching for e-commerce',
                'categories': ['electronics', 'fashion', 'home', 'beauty'],
                'annotations': ['product_id', 'categories', 'titles'],
                'size_gb': 15
            },
            'amazon_berkeley': {
                'name': 'Amazon Berkeley Objects Dataset',
                'url': 'https://amazon-berkeley-objects.s3.amazonaws.com/index.html',
                'description': '147K product images with 3D poses',
                'categories': ['household', 'toys', 'electronics'],
                'annotations': ['bbox', '3d_pose', 'materials', 'dimensions'],
                'size_gb': 20
            },
            'fashion_mnist': {
                'name': 'Fashion-MNIST',
                'url': 'https://github.com/zalandoresearch/fashion-mnist',
                'description': 'Fashion product classification dataset',
                'categories': ['clothing', 'shoes', 'bags'],
                'annotations': ['categories'],
                'size_gb': 0.5
            },
            'inshop_clothes': {
                'name': 'In-shop Clothes Retrieval',
                'url': 'http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html',
                'description': 'Fashion retrieval with 52K shop images',
                'categories': ['fashion', 'clothing'],
                'annotations': ['bbox', 'landmarks', 'attributes'],
                'size_gb': 10
            },
            'polyvore': {
                'name': 'Polyvore Dataset',
                'url': 'https://github.com/xthan/polyvore-dataset',
                'description': 'Fashion compatibility and outfit recommendation',
                'categories': ['fashion', 'outfits', 'compatibility'],
                'annotations': ['outfit_id', 'categories', 'compatibility'],
                'size_gb': 6
            },
            'flipkart_grid': {
                'name': 'Flipkart Grid Challenge',
                'url': 'https://www.kaggle.com/competitions/flipkart-grid-challenge',
                'description': 'Product information extraction',
                'categories': ['electronics', 'books', 'fashion'],
                'annotations': ['attributes', 'categories', 'text'],
                'size_gb': 12
            }
        }
    
    def list_datasets(self):
        """List available e-commerce datasets"""
        print("🛒 Available E-commerce Datasets:")
        print("=" * 50)
        
        for key, dataset in self.datasets.items():
            print(f"\n📦 {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Categories: {', '.join(dataset['categories'])}")
            print(f"   Size: {dataset['size_gb']}GB")
            print(f"   URL: {dataset['url']}")
    
    def get_dataset_info(self, dataset_name: str):
        """Get detailed information about a specific dataset"""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        else:
            print(f"Dataset '{dataset_name}' not found.")
            return None
    
    def recommend_datasets(self, use_case: str, budget_gb: int = 50):
        """Recommend datasets based on use case and storage budget"""
        recommendations = []
        
        use_case_mapping = {
            'fashion': ['deepfashion2', 'fashioniq', 'inshop_clothes', 'polyvore'],
            'electronics': ['shopee', 'amazon_berkeley', 'flipkart_grid'],
            'general_products': ['shopee', 'amazon_berkeley', 'flipkart_grid'],
            'clothing': ['deepfashion2', 'fashioniq', 'fashion_mnist'],
            'product_matching': ['shopee', 'flipkart_grid'],
            'outfit_recommendation': ['polyvore', 'deepfashion2'],
            'budget_training': ['fashion_mnist', 'polyvore']
        }
        
        if use_case in use_case_mapping:
            candidate_datasets = use_case_mapping[use_case]
            total_size = 0
            
            for dataset_name in candidate_datasets:
                dataset = self.datasets[dataset_name]
                if total_size + dataset['size_gb'] <= budget_gb:
                    recommendations.append(dataset_name)
                    total_size += dataset['size_gb']
        
        return recommendations

class EcommerceDataPreprocessor:
    """Preprocess e-commerce datasets for training"""
    
    def __init__(self, output_dir: str = "data/ecommerce"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # E-commerce specific categories
        self.product_categories = {
            'fashion': {
                'clothing': ['dress', 'shirt', 'pants', 'skirt', 'jacket', 'coat'],
                'shoes': ['sneakers', 'boots', 'sandals', 'heels', 'flats'],
                'accessories': ['bag', 'hat', 'jewelry', 'watch', 'belt', 'scarf']
            },
            'electronics': {
                'mobile': ['smartphone', 'tablet', 'smartwatch'],
                'computers': ['laptop', 'desktop', 'monitor', 'keyboard'],
                'audio': ['headphones', 'speakers', 'earbuds']
            },
            'home': {
                'furniture': ['chair', 'table', 'sofa', 'bed', 'cabinet'],
                'decor': ['lamp', 'vase', 'artwork', 'mirror', 'plant'],
                'kitchen': ['cookware', 'appliances', 'utensils']
            }
        }
        
        # Color mapping for products
        self.color_palette = {
            'red': [220, 20, 60], 'blue': [30, 144, 255], 'green': [50, 205, 50],
            'yellow': [255, 215, 0], 'orange': [255, 165, 0], 'purple': [138, 43, 226],
            'pink': [255, 192, 203], 'brown': [139, 69, 19], 'black': [0, 0, 0],
            'white': [255, 255, 255], 'gray': [128, 128, 128], 'navy': [0, 0, 128],
            'beige': [245, 245, 220], 'gold': [255, 215, 0], 'silver': [192, 192, 192]
        }
    
    def process_deepfashion2(self, data_dir: str):
        """Process DeepFashion2 dataset"""
        print("📦 Processing DeepFashion2 dataset...")
        
        data_path = Path(data_dir)
        annotations_dir = data_path / 'annos'
        images_dir = data_path / 'image'
        
        unified_annotations = []
        
        for split in ['train', 'validation', 'test']:
            split_dir = annotations_dir / split
            if not split_dir.exists():
                continue
                
            print(f"Processing {split} split...")
            
            for anno_file in split_dir.glob('*.json'):
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to unified format
                unified_ann = self.convert_deepfashion2_annotation(data, split)
                if unified_ann:
                    unified_annotations.append(unified_ann)
        
        # Save unified annotations
        self.save_unified_annotations(unified_annotations, 'deepfashion2')
        print(f"✅ Processed {len(unified_annotations)} DeepFashion2 samples")
    
    def convert_deepfashion2_annotation(self, data: Dict, split: str) -> Dict:
        """Convert DeepFashion2 annotation to unified format"""
        try:
            # Extract image info
            image_name = data.get('source', '')
            if not image_name:
                return None
            
            # Extract items (clothing pieces)
            items = data.get('item', {})
            objects = []
            
            for item_id, item_data in items.items():
                if 'bounding_box' in item_data:
                    bbox = item_data['bounding_box']
                    category_id = item_data.get('category_id', 0)
                    category_name = item_data.get('category_name', 'unknown')
                    
                    # Extract attributes
                    attributes = {}
                    if 'style' in item_data:
                        attributes['style'] = item_data['style']
                    if 'color' in item_data:
                        attributes['color'] = item_data['color']
                    
                    objects.append({
                        'bbox': bbox,
                        'class_id': category_id,
                        'class_name': category_name,
                        'attributes': attributes,
                        'landmarks': item_data.get('landmarks', [])
                    })
            
            # Create unified annotation
            unified_ann = {
                'image_file': image_name,
                'dataset': 'deepfashion2',
                'split': split,
                'objects': objects,
                'scene': {
                    'category': 'fashion',
                    'indoor_outdoor': 'indoor',
                    'context': 'fashion_photography'
                },
                'ecommerce_attributes': {
                    'product_type': 'fashion',
                    'image_type': 'product_photo',
                    'background': 'clean'
                }
            }
            
            return unified_ann
            
        except Exception as e:
            print(f"Error processing annotation: {e}")
            return None
    
    def process_shopee_dataset(self, data_dir: str):
        """Process Shopee product matching dataset"""
        print("📦 Processing Shopee dataset...")
        
        data_path = Path(data_dir)
        
        # Load product data
        train_csv = data_path / 'train.csv'
        test_csv = data_path / 'test.csv'
        
        if not train_csv.exists():
            print("❌ Shopee train.csv not found")
            return
        
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        
        unified_annotations = []
        
        for idx, row in train_df.iterrows():
            # Extract product info
            product_id = row.get('posting_id', '')
            image_path = row.get('image', '')
            title = row.get('title', '')
            label_group = row.get('label_group', 0)
            
            # Parse title for attributes
            attributes = self.extract_attributes_from_title(title)
            
            # Create unified annotation
            unified_ann = {
                'image_file': image_path,
                'dataset': 'shopee',
                'split': 'train',
                'product_id': product_id,
                'title': title,
                'label_group': label_group,
                'objects': [{
                    'bbox': [0, 0, 1, 1],  # Full image for product matching
                    'class_name': 'product',
                    'attributes': attributes
                }],
                'scene': {
                    'category': 'ecommerce',
                    'context': 'product_listing'
                },
                'ecommerce_attributes': {
                    'product_type': attributes.get('category', 'general'),
                    'image_type': 'listing_photo'
                }
            }
            
            unified_annotations.append(unified_ann)
            
            if idx % 1000 == 0:
                print(f"Processed {idx} products...")
        
        self.save_unified_annotations(unified_annotations, 'shopee')
        print(f"✅ Processed {len(unified_annotations)} Shopee products")
    
    def extract_attributes_from_title(self, title: str) -> Dict[str, Any]:
        """Extract product attributes from title text"""
        title_lower = title.lower()
        attributes = {}
        
        # Extract colors
        detected_colors = []
        for color, rgb in self.color_palette.items():
            if color in title_lower:
                detected_colors.append(color)
        attributes['colors'] = detected_colors
        
        # Extract categories
        for main_cat, sub_cats in self.product_categories.items():
            for sub_cat, items in sub_cats.items():
                for item in items:
                    if item in title_lower:
                        attributes['main_category'] = main_cat
                        attributes['sub_category'] = sub_cat
                        attributes['item_type'] = item
                        break
        
        # Extract materials (common ones)
        materials = ['cotton', 'silk', 'leather', 'wool', 'polyester', 'denim', 'plastic', 'metal', 'wood']
        detected_materials = [mat for mat in materials if mat in title_lower]
        attributes['materials'] = detected_materials
        
        # Extract size information
        sizes = ['xs', 'small', 'medium', 'large', 'xl', 'xxl']
        detected_sizes = [size for size in sizes if size in title_lower]
        attributes['sizes'] = detected_sizes
        
        return attributes
    
    def save_unified_annotations(self, annotations: List[Dict], dataset_name: str):
        """Save unified annotations"""
        # Split annotations
        train_size = int(0.8 * len(annotations))
        val_size = int(0.1 * len(annotations))
        
        train_data = annotations[:train_size]
        val_data = annotations[train_size:train_size + val_size]
        test_data = annotations[train_size + val_size:]
        
        # Save splits
        for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            output_file = self.output_dir / f'{dataset_name}_{split}_annotations.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(data)} {split} annotations to {output_file}")

class EcommerceTrainingConfig:
    """Generate e-commerce specific training configurations"""
    
    @staticmethod
    def create_fashion_config():
        """Create configuration for fashion datasets"""
        return {
            "experiment_name": "ecommerce_fashion",
            "data_dir": "data/ecommerce",
            "model_config": {
                "backbone": "efficientnet-b2",
                "input_size": 512,
                "fashion_specific": True
            },
            "ecommerce_tasks": {
                "product_detection": {"weight": 1.0, "classes": 50},
                "attribute_recognition": {"weight": 0.8, "attributes": 30},
                "color_detection": {"weight": 0.6, "colors": 15},
                "material_classification": {"weight": 0.7, "materials": 20},
                "style_classification": {"weight": 0.5, "styles": 25}
            },
            "fashion_specific": {
                "landmark_detection": True,
                "outfit_compatibility": True,
                "size_estimation": True,
                "trend_analysis": False
            }
        }
    
    @staticmethod
    def create_general_ecommerce_config():
        """Create configuration for general e-commerce datasets"""
        return {
            "experiment_name": "ecommerce_general",
            "data_dir": "data/ecommerce",
            "model_config": {
                "backbone": "efficientnet-b1",
                "input_size": 448,
                "multi_category": True
            },
            "ecommerce_tasks": {
                "product_detection": {"weight": 1.0, "classes": 100},
                "category_classification": {"weight": 0.9, "categories": 20},
                "brand_recognition": {"weight": 0.6, "brands": 50},
                "quality_assessment": {"weight": 0.7},
                "price_estimation": {"weight": 0.4}
            },
            "general_ecommerce": {
                "multi_view_support": True,
                "background_removal": True,
                "text_detection": True,
                "logo_detection": True
            }
        }

def create_ecommerce_download_script():
    """Create script to download e-commerce datasets"""
    script_content = '''#!/bin/bash

# E-commerce Dataset Downloader
echo "🛒 E-commerce Dataset Downloader"
echo "================================"

mkdir -p data/ecommerce/downloads
cd data/ecommerce/downloads

# Fashion-MNIST (small, good for testing)
echo "📦 Downloading Fashion-MNIST..."
wget -c https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
wget -c https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
wget -c https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
wget -c https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz

# DeepFashion2 (requires manual download from GitHub)
echo "📝 DeepFashion2 requires manual download:"
echo "   1. Visit: https://github.com/switchablenorms/DeepFashion2"
echo "   2. Follow dataset download instructions"
echo "   3. Extract to data/ecommerce/downloads/deepfashion2/"

# Shopee (Kaggle competition - requires Kaggle API)
echo "📝 Shopee dataset (Kaggle):"
echo "   1. Install kaggle: pip install kaggle"
echo "   2. Setup API key: ~/.kaggle/kaggle.json"
echo "   3. Download: kaggle competitions download -c shopee-product-matching"

echo "✅ Available datasets downloaded!"
echo "📁 Location: $(pwd)"
'''
    
    with open('download_ecommerce_datasets.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('download_ecommerce_datasets.sh', 0o755)
    print("✅ Created download_ecommerce_datasets.sh")

def main():
    """Main function for e-commerce adaptation"""
    print("🛒 E-commerce Dataset Adaptation for Visual Intelligence Pipeline")
    print("=" * 70)
    
    # Initialize dataset manager
    dataset_manager = EcommerceDatasetManager()
    
    # Show available datasets
    dataset_manager.list_datasets()
    
    print("\n💡 Dataset Recommendations:")
    print("=" * 30)
    
    use_cases = ['fashion', 'electronics', 'general_products', 'budget_training']
    
    for use_case in use_cases:
        recommendations = dataset_manager.recommend_datasets(use_case, budget_gb=20)
        print(f"\n🎯 {use_case.replace('_', ' ').title()}:")
        for dataset in recommendations:
            info = dataset_manager.get_dataset_info(dataset)
            print(f"   • {info['name']} ({info['size_gb']}GB)")
    
    # Create download script
    create_ecommerce_download_script()
    
    # Generate sample configs
    print("\n⚙️ Creating sample configurations...")
    
    fashion_config = EcommerceTrainingConfig.create_fashion_config()
    with open('config/ecommerce_fashion_config.json', 'w') as f:
        json.dump(fashion_config, f, indent=2)
    
    general_config = EcommerceTrainingConfig.create_general_ecommerce_config()
    with open('config/ecommerce_general_config.json', 'w') as f:
        json.dump(general_config, f, indent=2)
    
    print("✅ Created ecommerce_fashion_config.json")
    print("✅ Created ecommerce_general_config.json")
    
    print("\n🚀 Next Steps:")
    print("1. Run: ./download_ecommerce_datasets.sh")
    print("2. Process datasets with EcommerceDataPreprocessor")
    print("3. Train with: python train.py --config config/ecommerce_fashion_config.json")

if __name__ == "__main__":
    main()