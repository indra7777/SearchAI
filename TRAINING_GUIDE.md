import os
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

class DatasetPreprocessor:
    """Preprocessor to convert various datasets into unified format"""
    
    def __init__(self, source_dirs: dict, output_dir: str):
        self.source_dirs = source_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        
    def process_coco_dataset(self):
        """Process COCO dataset for object detection"""
        print("Processing COCO dataset...")
        
        import pycocotools.coco as coco
        
        # Load COCO annotations
        coco_train = coco.COCO(self.source_dirs['coco'] + '/annotations/instances_train2017.json')
        coco_val = coco.COCO(self.source_dirs['coco'] + '/annotations/instances_val2017.json')
        
        train_annotations = self.convert_coco_annotations(coco_train, 'train2017')
        val_annotations = self.convert_coco_annotations(coco_val, 'val2017')
        
        # Save annotations
        with open(self.output_dir / 'train_annotations.json', 'w') as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(self.output_dir / 'val_annotations.json', 'w') as f:
            json.dump(val_annotations, f, indent=2)
            
        print(f"Processed {len(train_annotations)} training images")
        print(f"Processed {len(val_annotations)} validation images")
    
    def convert_coco_annotations(self, coco_api, split):
        """Convert COCO annotations to our format"""
        annotations = []
        
        img_ids = coco_api.getImgIds()
        
        for img_id in tqdm(img_ids[:10000]):  # Limit for demo
            img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=img_id)
            anns = coco_api.loadAnns(ann_ids)
            
            # Copy image
            src_path = self.source_dirs['coco'] + f'/images/{split}/{img_info["file_name"]}'
            dst_path = self.output_dir / 'images' / ('train' if 'train' in split else 'val') / img_info["file_name"]
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                
                # Convert annotations
                objects = []
                for ann in anns:
                    if ann['area'] > 0:
                        bbox = ann['bbox']  # [x, y, width, height]
                        # Convert to [x1, y1, x2, y2]
                        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                        
                        cat_info = coco_api.loadCats(ann['category_id'])[0]
                        
                        objects.append({
                            'bbox': bbox,
                            'class_id': ann['category_id'],
                            'class_name': cat_info['name'],
                            'area': ann['area']
                        })
                
                # Create unified annotation
                annotation = {
                    'image_file': img_info["file_name"],
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'objects': objects,
                    'relationships': [],  # Will be populated later
                    'scene': {'category': 'unknown', 'indoor_outdoor': 'unknown'},
                    'faces': [],
                    'materials': [],
                    'textures': [],
                    'lighting': {},
                    'shadows': [],
                    'occlusions': []
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def process_places365_dataset(self):
        """Process Places365 dataset for scene classification"""
        print("Processing Places365 dataset...")
        
        places_dir = Path(self.source_dirs['places365'])
        
        # Load category mapping
        with open(places_dir / 'categories_places365.txt', 'r') as f:
            categories = [line.strip().split(' ')[0][3:] for line in f.readlines()]
        
        # Process train and val splits
        for split in ['train', 'val']:
            split_file = places_dir / f'places365_{split}_standard.txt'
            
            if split_file.exists():
                self.process_places365_split(split_file, categories, split)
    
    def process_places365_split(self, split_file, categories, split):
        """Process a single split of Places365"""
        scene_annotations = {}
        
        with open(split_file, 'r') as f:
            for line in tqdm(f.readlines()[:5000]):  # Limit for demo
                parts = line.strip().split(' ')
                img_path = parts[0]
                class_id = int(parts[1])
                
                # Copy image
                src_path = self.source_dirs['places365'] + '/' + img_path
                img_name = os.path.basename(img_path)
                dst_path = self.output_dir / 'images' / split / img_name
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    
                    # Store scene information
                    scene_annotations[img_name] = {
                        'category': categories[class_id],
                        'indoor_outdoor': 'indoor' if 'indoor' in categories[class_id] else 'outdoor'
                    }
        
        # Update existing annotations with scene information
        annotation_file = self.output_dir / f'{split}_annotations.json'
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Update scene information
            for ann in annotations:
                if ann['image_file'] in scene_annotations:
                    ann['scene'] = scene_annotations[ann['image_file']]
            
            # Save updated annotations
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)
    
    def process_facial_dataset(self):
        """Process facial analysis dataset"""
        print("Processing facial dataset...")
        
        # This would process FER2013, CelebA, or other facial datasets
        # Implementation depends on specific dataset format
        pass
    
    def process_material_dataset(self):
        """Process materials dataset"""
        print("Processing materials dataset...")
        
        # Process Materials in Context or similar dataset
        materials_dir = Path(self.source_dirs.get('materials', ''))
        
        if materials_dir.exists():
            # Implementation depends on dataset structure
            pass
    
    def process_texture_dataset(self):
        """Process texture dataset (DTD)"""
        print("Processing texture dataset...")
        
        dtd_dir = Path(self.source_dirs.get('dtd', ''))
        
        if dtd_dir.exists():
            # Process DTD dataset
            for split in ['train', 'val', 'test']:
                split_file = dtd_dir / 'labels' / f'{split}1.txt'
                
                if split_file.exists():
                    self.process_dtd_split(split_file, split)
    
    def process_dtd_split(self, split_file, split):
        """Process DTD split"""
        texture_annotations = {}
        
        with open(split_file, 'r') as f:
            for line in f.readlines():
                img_path = line.strip()
                texture_class = img_path.split('/')[0]
                img_name = os.path.basename(img_path)
                
                # Copy image
                src_path = self.source_dirs['dtd'] + '/images/' + img_path
                dst_path = self.output_dir / 'images' / split / img_name
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    texture_annotations[img_name] = texture_class
        
        # Update annotations
        annotation_file = self.output_dir / f'{split}_annotations.json'
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                if ann['image_file'] in texture_annotations:
                    ann['textures'].append({
                        'class': texture_annotations[ann['image_file']],
                        'bbox': [0, 0, ann['width'], ann['height']]  # Full image
                    })
            
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)
    
    def generate_synthetic_relationships(self):
        """Generate synthetic relationships for training"""
        print("Generating synthetic relationships...")
        
        for split in ['train', 'val']:
            annotation_file = self.output_dir / f'{split}_annotations.json'
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                
                for ann in annotations:
                    objects = ann['objects']
                    relationships = []
                    
                    # Generate spatial relationships
                    for i, obj1 in enumerate(objects):
                        for j, obj2 in enumerate(objects):
                            if i != j:
                                rel = self.infer_spatial_relationship(obj1, obj2)
                                if rel:
                                    relationships.append({
                                        'subject_id': i,
                                        'object_id': j,
                                        'relationship': rel,
                                        'confidence': 0.8
                                    })
                    
                    ann['relationships'] = relationships
                
                # Save updated annotations
                with open(annotation_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
    
    def infer_spatial_relationship(self, obj1, obj2):
        """Infer spatial relationship between two objects"""
        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']
        
        # Calculate centers
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        # Calculate relative position
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # Determine relationship
        if abs(dx) > abs(dy):
            return 'right_of' if dx > 0 else 'left_of'
        else:
            return 'below' if dy > 0 else 'above'
    
    def create_test_split(self, test_ratio=0.1):
        """Create test split from validation data"""
        print("Creating test split...")
        
        val_file = self.output_dir / 'val_annotations.json'
        if val_file.exists():
            with open(val_file, 'r') as f:
                val_annotations = json.load(f)
            
            # Split validation into val and test
            num_test = int(len(val_annotations) * test_ratio)
            test_annotations = val_annotations[:num_test]
            val_annotations = val_annotations[num_test:]
            
            # Move test images
            for ann in test_annotations:
                src_path = self.output_dir / 'images' / 'val' / ann['image_file']
                dst_path = self.output_dir / 'images' / 'test' / ann['image_file']
                
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
            
            # Save splits
            with open(self.output_dir / 'val_annotations.json', 'w') as f:
                json.dump(val_annotations, f, indent=2)
            
            with open(self.output_dir / 'test_annotations.json', 'w') as f:
                json.dump(test_annotations, f, indent=2)
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("Starting dataset preprocessing...")
        
        # Process each dataset
        if 'coco' in self.source_dirs:
            self.process_coco_dataset()
        
        if 'places365' in self.source_dirs:
            self.process_places365_dataset()
        
        if 'materials' in self.source_dirs:
            self.process_material_dataset()
        
        if 'dtd' in self.source_dirs:
            self.process_texture_dataset()
        
        # Generate synthetic data
        self.generate_synthetic_relationships()
        
        # Create test split
        self.create_test_split()
        
        print("Dataset preprocessing completed!")

def main():
    """Main preprocessing function"""
    
    # Configure dataset paths
    source_dirs = {
        'coco': 'data/datasets/coco',
        'places365': 'data/datasets/places365standard_easyformat',
        'dtd': 'data/datasets/dtd',
        'materials': 'data/datasets/materials',  # Optional
    }
    
    # Create preprocessor
    preprocessor = DatasetPreprocessor(source_dirs, 'data/processed')
    
    # Run preprocessing
    preprocessor.run_preprocessing()

if __name__ == "__main__":
    main()