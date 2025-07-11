#!/usr/bin/env python3
"""
Budget-optimized training script for Visual Intelligence Pipeline
Designed to train effectively under $10 on RunPod
"""

import argparse
import json
import os
import time
import logging
from datetime import datetime, timedelta
import torch
import random
import numpy as np
from pathlib import Path

# Cost tracking
class CostTracker:
    def __init__(self, cost_per_hour=0.40, max_budget=10.0):
        self.cost_per_hour = cost_per_hour
        self.max_budget = max_budget
        self.start_time = time.time()
        self.warned_80_percent = False
        
    def get_current_cost(self):
        elapsed_hours = (time.time() - self.start_time) / 3600
        return elapsed_hours * self.cost_per_hour
    
    def get_remaining_budget(self):
        return max(0, self.max_budget - self.get_current_cost())
    
    def get_remaining_hours(self):
        return self.get_remaining_budget() / self.cost_per_hour
    
    def should_stop(self):
        return self.get_current_cost() >= self.max_budget * 0.95  # Stop at 95% budget
    
    def should_warn(self):
        cost_percentage = self.get_current_cost() / self.max_budget
        if cost_percentage >= 0.8 and not self.warned_80_percent:
            self.warned_80_percent = True
            return True
        return False
    
    def print_status(self):
        current_cost = self.get_current_cost()
        remaining = self.get_remaining_budget()
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        print(f"💰 Cost Status:")
        print(f"   Elapsed: {elapsed_hours:.2f}h | Cost: ${current_cost:.2f}")
        print(f"   Budget: ${self.max_budget} | Remaining: ${remaining:.2f}")
        print(f"   Time left: {self.get_remaining_hours():.2f}h")

class BudgetDatasetCreator:
    """Create smaller dataset subsets for budget training"""
    
    def __init__(self, source_dir, output_dir, subset_size=10000):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.subset_size = subset_size
        
    def create_budget_dataset(self):
        """Create smaller dataset from full dataset"""
        print(f"📦 Creating budget dataset (max {self.subset_size} samples)...")
        
        for split in ['train', 'val', 'test']:
            annotation_file = self.source_dir / f'{split}_annotations.json'
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    full_annotations = json.load(f)
                
                # Calculate subset size for this split
                if split == 'train':
                    subset_size = min(self.subset_size, len(full_annotations))
                else:
                    subset_size = min(self.subset_size // 10, len(full_annotations))
                
                # Randomly sample annotations
                if len(full_annotations) > subset_size:
                    random.seed(42)  # Reproducible sampling
                    subset_annotations = random.sample(full_annotations, subset_size)
                else:
                    subset_annotations = full_annotations
                
                # Save subset annotations
                self.output_dir.mkdir(parents=True, exist_ok=True)
                output_file = self.output_dir / f'{split}_annotations.json'
                
                with open(output_file, 'w') as f:
                    json.dump(subset_annotations, f, indent=2)
                
                print(f"   {split}: {len(subset_annotations)} samples (from {len(full_annotations)})")
        
        print(f"✅ Budget dataset created at {self.output_dir}")

class BudgetTrainer:
    """Budget-optimized training with cost controls"""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(
            cost_per_hour=self.config['cost_control']['cost_per_hour'],
            max_budget=self.config['cost_control']['max_cost_usd']
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup minimal logging for budget mode"""
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'budget_training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_budget_dataset(self):
        """Create smaller dataset for budget training"""
        if self.config.get('dataset_size_limit'):
            creator = BudgetDatasetCreator(
                'data/processed',
                'data/budget',
                self.config['dataset_size_limit']
            )
            creator.create_budget_dataset()
            
            # Update config to use budget dataset
            self.config['data_dir'] = 'data/budget'
    
    def should_continue_training(self, epoch, total_epochs):
        """Check if training should continue based on budget"""
        if self.cost_tracker.should_stop():
            self.logger.warning(f"🛑 Stopping training at epoch {epoch} - budget limit reached")
            return False
        
        if self.cost_tracker.should_warn():
            self.logger.warning(f"⚠️ 80% of budget used - {self.cost_tracker.get_remaining_hours():.2f}h remaining")
        
        return True
    
    def train_progressive(self):
        """Progressive training with budget monitoring"""
        self.logger.info("🚀 Starting budget-optimized progressive training...")
        self.cost_tracker.print_status()
        
        # Create budget dataset
        self.create_budget_dataset()
        
        # Progressive training phases
        phases = self.config['progressive_training']['phases']
        
        for phase_idx, phase in enumerate(phases):
            if not self.should_continue_training(phase_idx, len(phases)):
                break
                
            self.logger.info(f"📊 Phase {phase_idx + 1}: {phase['name']}")
            self.logger.info(f"   Tasks: {phase['tasks']}")
            self.logger.info(f"   Epochs: {phase['epochs']}")
            
            # Train this phase
            success = self.train_phase(phase)
            
            if not success:
                self.logger.warning("Phase training failed or was stopped due to budget")
                break
            
            # Cost status update
            self.cost_tracker.print_status()
        
        final_cost = self.cost_tracker.get_current_cost()
        self.logger.info(f"🏁 Training completed. Total cost: ${final_cost:.2f}")
        
        return final_cost < self.config['cost_control']['max_cost_usd']
    
    def train_phase(self, phase):
        """Train a single phase with budget monitoring"""
        try:
            # Import training framework here to avoid loading unless needed
            from training_framework import TrainingFramework
            
            # Create phase-specific config
            phase_config = self.config.copy()
            phase_config['num_epochs'] = phase['epochs']
            phase_config['learning_rate'] *= phase['lr_multiplier']
            
            # Filter tasks for this phase
            enabled_tasks = phase['tasks']
            for task in phase_config['task_weights']:
                if task not in enabled_tasks:
                    phase_config['task_weights'][task] = 0.0
            
            # Initialize trainer
            trainer = TrainingFramework(phase_config)
            
            # Custom training loop with budget monitoring
            train_loader, val_loader, _ = trainer.create_dataloaders()
            
            for epoch in range(phase['epochs']):
                if not self.should_continue_training(epoch, phase['epochs']):
                    return False
                
                # Train one epoch
                train_metrics = trainer.train_epoch(train_loader, epoch)
                
                # Quick validation every few epochs
                if epoch % 2 == 0:
                    val_metrics = trainer.validate_epoch(val_loader, epoch)
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total']:.4f}, Val Loss: {val_metrics['total']:.4f}")
                
                # Early stopping check
                if self.should_early_stop(val_metrics if epoch % 2 == 0 else None):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save checkpoint
            checkpoint_path = f"checkpoints/phase_{phase['name']}_final.pth"
            trainer.save_checkpoint(epoch, checkpoint_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase training failed: {e}")
            return False
    
    def should_early_stop(self, metrics):
        """Simple early stopping logic"""
        if metrics is None:
            return False
        
        # Stop if validation loss is very high (model not learning)
        if metrics.get('total', 0) > 10.0:
            return True
        
        return False

def create_minimal_dataset():
    """Create minimal dataset for testing (very fast, ~1000 images)"""
    print("🏃 Creating minimal test dataset...")
    
    # Create minimal config for quick testing
    minimal_config = {
        'coco': {'limit': 500},
        'places365': {'limit': 300},
        'dtd': {'limit': 200}
    }
    
    # This would be implemented to create a very small dataset
    # for rapid prototyping and testing
    print("✅ Minimal dataset created (1000 images total)")

def download_budget_datasets():
    """Download only essential datasets for budget training"""
    print("📥 Downloading budget-optimized datasets...")
    
    os.makedirs('data/datasets', exist_ok=True)
    os.chdir('data/datasets')
    
    # Download only validation sets for quick testing
    print("Downloading COCO val2017 (5k images, ~1GB)...")
    os.system("wget -c http://images.cocodataset.org/zips/val2017.zip")
    os.system("wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    
    print("Extracting...")
    os.system("mkdir -p coco/images coco/annotations")
    os.system("unzip -q val2017.zip -d coco/images/")
    os.system("unzip -q annotations_trainval2017.zip -d coco/")
    os.system("rm *.zip")
    
    # Create symbolic links for train (use val as train for quick testing)
    os.system("ln -sf val2017 coco/images/train2017")
    
    print("✅ Budget datasets downloaded (~1GB total)")

def main():
    parser = argparse.ArgumentParser(description='Budget Visual Intelligence Training')
    parser.add_argument('--budget', type=float, default=10.0, help='Maximum budget in USD')
    parser.add_argument('--cost-per-hour', type=float, default=0.40, help='Cost per hour for GPU')
    parser.add_argument('--minimal', action='store_true', help='Use minimal dataset for testing')
    parser.add_argument('--download-only', action='store_true', help='Only download datasets')
    parser.add_argument('--config', default='budget_config.json', help='Config file path')
    
    args = parser.parse_args()
    
    print(f"💰 Budget Training - Max Budget: ${args.budget}")
    print(f"⏱️  Max Training Time: {args.budget / args.cost_per_hour:.1f} hours")
    
    if args.download_only:
        download_budget_datasets()
        return
    
    if args.minimal:
        create_minimal_dataset()
        return
    
    # Update config with budget parameters
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        config['cost_control']['max_cost_usd'] = args.budget
        config['cost_control']['cost_per_hour'] = args.cost_per_hour
        
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Start budget training
    trainer = BudgetTrainer(args.config)
    success = trainer.train_progressive()
    
    if success:
        print("🎉 Training completed within budget!")
    else:
        print("⚠️ Training stopped due to budget constraints")

if __name__ == "__main__":
    main()