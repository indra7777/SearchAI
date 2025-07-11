#!/usr/bin/env python3
"""
Main training script for Visual Intelligence Pipeline
"""

import argparse
import json
import os
import logging
from pathlib import Path
import torch
import wandb

from training_framework import TrainingFramework, DEFAULT_CONFIG
from data_preprocessing import DatasetPreprocessor

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load training configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge with default config
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    else:
        return DEFAULT_CONFIG

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Visual Intelligence Pipeline')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed dataset directory')
    parser.add_argument('--config', type=str, default='config/training_config.json',
                       help='Path to training configuration file')
    
    # Model arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project-name', type=str, default='visual-intelligence-pipeline',
                       help='WandB project name')
    
    # Preprocessing arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing before training')
    parser.add_argument('--source-datasets', type=str, default='config/dataset_paths.json',
                       help='JSON file with source dataset paths')
    
    return parser.parse_args()

def preprocess_data(source_datasets_path: str, output_dir: str):
    """Run data preprocessing"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    # Load dataset paths
    if os.path.exists(source_datasets_path):
        with open(source_datasets_path, 'r') as f:
            source_dirs = json.load(f)
    else:
        logger.warning(f"Dataset paths file not found: {source_datasets_path}")
        logger.info("Using default paths...")
        source_dirs = {
            'coco': 'data/datasets/coco',
            'places365': 'data/datasets/places365standard_easyformat',
            'dtd': 'data/datasets/dtd',
        }
    
    # Run preprocessing
    preprocessor = DatasetPreprocessor(source_dirs, output_dir)
    preprocessor.run_preprocessing()
    
    logger.info("Data preprocessing completed!")

def validate_dataset(data_dir: str):
    """Validate that dataset is properly formatted"""
    logger = logging.getLogger(__name__)
    
    data_path = Path(data_dir)
    
    # Check directory structure
    required_dirs = ['images/train', 'images/val', 'images/test']
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Check annotation files
    required_files = ['train_annotations.json', 'val_annotations.json', 'test_annotations.json']
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required annotation file not found: {file_path}")
        
        # Validate JSON format
        try:
            with open(file_path, 'r') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                raise ValueError(f"Annotations should be a list: {file_path}")
            
            # Check first annotation structure
            if len(annotations) > 0:
                required_keys = ['image_file', 'objects', 'scene']
                for key in required_keys:
                    if key not in annotations[0]:
                        raise ValueError(f"Missing key '{key}' in annotations: {file_path}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    
    logger.info(f"Dataset validation passed: {data_dir}")

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting Visual Intelligence Pipeline training...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Preprocess data if requested
    if args.preprocess:
        preprocess_data(args.source_datasets, args.data_dir)
    
    # Validate dataset
    try:
        validate_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Dataset validation failed: {e}")
        logger.error("Please run preprocessing first: python train.py --preprocess")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'num_workers': args.num_workers,
        'mixed_precision': args.mixed_precision,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'use_wandb': args.use_wandb,
        'project_name': args.project_name
    })
    
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Initialize WandB if requested
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=config,
            name=f"training_{config.get('experiment_name', 'default')}"
        )
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['device'] = 'cpu'
    
    logger.info(f"Using device: {config['device']}")
    
    try:
        # Initialize training framework
        logger.info("Initializing training framework...")
        trainer = TrainingFramework(config)
        
        # Resume from checkpoint if provided
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()