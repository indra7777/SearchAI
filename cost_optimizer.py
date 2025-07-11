#!/usr/bin/env python3
"""
Cost Optimization Tools for RunPod Training
Real-time cost monitoring and automatic budget control
"""

import time
import json
import subprocess
import psutil
import os
from datetime import datetime, timedelta
import logging

class RunPodCostOptimizer:
    """Real-time cost monitoring and optimization for RunPod"""
    
    def __init__(self, max_budget=10.0, cost_per_hour=0.40):
        self.max_budget = max_budget
        self.cost_per_hour = cost_per_hour
        self.start_time = time.time()
        self.warnings_sent = set()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cost tracking file
        self.cost_file = 'cost_tracking.json'
        self.init_cost_tracking()
    
    def init_cost_tracking(self):
        """Initialize cost tracking file"""
        cost_data = {
            'start_time': self.start_time,
            'max_budget': self.max_budget,
            'cost_per_hour': self.cost_per_hour,
            'checkpoints': []
        }
        
        with open(self.cost_file, 'w') as f:
            json.dump(cost_data, f, indent=2)
    
    def get_elapsed_time(self):
        """Get elapsed time in hours"""
        return (time.time() - self.start_time) / 3600
    
    def get_current_cost(self):
        """Get current cost based on elapsed time"""
        return self.get_elapsed_time() * self.cost_per_hour
    
    def get_remaining_budget(self):
        """Get remaining budget"""
        return max(0, self.max_budget - self.get_current_cost())
    
    def get_estimated_completion_time(self, progress_percentage):
        """Estimate total time needed based on current progress"""
        if progress_percentage <= 0:
            return float('inf')
        
        elapsed_time = self.get_elapsed_time()
        estimated_total_time = elapsed_time / (progress_percentage / 100)
        return estimated_total_time
    
    def get_estimated_total_cost(self, progress_percentage):
        """Estimate total cost based on current progress"""
        estimated_time = self.get_estimated_completion_time(progress_percentage)
        return estimated_time * self.cost_per_hour
    
    def should_stop_training(self, progress_percentage=None):
        """Determine if training should be stopped"""
        current_cost = self.get_current_cost()
        
        # Hard stop at 95% budget
        if current_cost >= self.max_budget * 0.95:
            return True, "Budget limit reached (95%)"
        
        # Predictive stop if we can estimate completion cost
        if progress_percentage and progress_percentage > 10:
            estimated_cost = self.get_estimated_total_cost(progress_percentage)
            if estimated_cost > self.max_budget:
                return True, f"Estimated total cost (${estimated_cost:.2f}) exceeds budget"
        
        return False, ""
    
    def log_checkpoint(self, epoch, metrics=None):
        """Log a training checkpoint with cost info"""
        checkpoint = {
            'timestamp': time.time(),
            'epoch': epoch,
            'elapsed_hours': self.get_elapsed_time(),
            'current_cost': self.get_current_cost(),
            'remaining_budget': self.get_remaining_budget(),
            'metrics': metrics or {}
        }
        
        # Load existing data
        with open(self.cost_file, 'r') as f:
            cost_data = json.load(f)
        
        # Add checkpoint
        cost_data['checkpoints'].append(checkpoint)
        
        # Save updated data
        with open(self.cost_file, 'w') as f:
            json.dump(cost_data, f, indent=2)
        
        return checkpoint
    
    def send_warning(self, warning_type, message):
        """Send warning (avoid duplicates)"""
        if warning_type not in self.warnings_sent:
            self.logger.warning(f"💰 COST WARNING: {message}")
            self.warnings_sent.add(warning_type)
    
    def monitor_step(self, epoch=None, total_epochs=None, metrics=None):
        """Monitor one training step"""
        current_cost = self.get_current_cost()
        remaining_budget = self.get_remaining_budget()
        elapsed_hours = self.get_elapsed_time()
        
        # Calculate progress if epoch info available
        progress = None
        if epoch is not None and total_epochs is not None:
            progress = (epoch / total_epochs) * 100
        
        # Log checkpoint
        checkpoint = self.log_checkpoint(epoch, metrics)
        
        # Check for warnings
        cost_percentage = current_cost / self.max_budget
        
        if cost_percentage >= 0.5 and cost_percentage < 0.8:
            self.send_warning('50_percent', f"50% budget used (${current_cost:.2f}/${self.max_budget})")
        elif cost_percentage >= 0.8 and cost_percentage < 0.9:
            self.send_warning('80_percent', f"80% budget used - ${remaining_budget:.2f} remaining")
        elif cost_percentage >= 0.9:
            self.send_warning('90_percent', f"90% budget used - STOP SOON!")
        
        # Check if should stop
        should_stop, reason = self.should_stop_training(progress)
        
        # Print status
        self.print_status(progress)
        
        if should_stop:
            self.logger.error(f"🛑 STOPPING TRAINING: {reason}")
            return False
        
        return True
    
    def print_status(self, progress=None):
        """Print current cost status"""
        current_cost = self.get_current_cost()
        remaining_budget = self.get_remaining_budget()
        elapsed_hours = self.get_elapsed_time()
        remaining_hours = remaining_budget / self.cost_per_hour
        
        print(f"\n💰 Cost Status ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Elapsed: {elapsed_hours:.2f}h | Cost: ${current_cost:.2f}")
        print(f"   Budget: ${self.max_budget} | Remaining: ${remaining_budget:.2f}")
        print(f"   Time left: {remaining_hours:.2f}h")
        
        if progress:
            print(f"   Progress: {progress:.1f}%")
            estimated_cost = self.get_estimated_total_cost(progress)
            print(f"   Estimated total: ${estimated_cost:.2f}")
    
    def optimize_for_remaining_time(self):
        """Suggest optimizations based on remaining time"""
        remaining_hours = self.get_remaining_budget() / self.cost_per_hour
        
        suggestions = []
        
        if remaining_hours < 2:
            suggestions.extend([
                "Reduce batch size to 8-16",
                "Skip validation for remaining epochs", 
                "Reduce image size to 224x224",
                "Train only core tasks (object detection + scene)"
            ])
        elif remaining_hours < 5:
            suggestions.extend([
                "Reduce batch size to 16-24",
                "Validate every 2 epochs",
                "Reduce learning rate for faster convergence"
            ])
        elif remaining_hours < 10:
            suggestions.extend([
                "Continue with current settings",
                "Consider early stopping if validation plateaus"
            ])
        
        return suggestions

def monitor_training_process():
    """Monitor training process and enforce budget"""
    optimizer = RunPodCostOptimizer(max_budget=10.0, cost_per_hour=0.40)
    
    print("🔍 Starting cost monitoring...")
    print(f"Budget: ${optimizer.max_budget} | Rate: ${optimizer.cost_per_hour}/hour")
    print("Press Ctrl+C to stop monitoring")
    
    epoch = 0
    try:
        while True:
            # Check if training process is running
            training_running = check_training_process()
            
            if not training_running:
                print("⚠️ No training process detected")
                time.sleep(30)
                continue
            
            # Monitor this step
            continue_training = optimizer.monitor_step(epoch=epoch, total_epochs=100)
            
            if not continue_training:
                # Try to stop training process
                stop_training_process()
                break
            
            # Check for optimization suggestions
            if epoch % 10 == 0:
                suggestions = optimizer.optimize_for_remaining_time()
                if suggestions:
                    print("\n💡 Optimization suggestions:")
                    for suggestion in suggestions:
                        print(f"   • {suggestion}")
            
            epoch += 1
            time.sleep(60)  # Monitor every minute
            
    except KeyboardInterrupt:
        print("\n👋 Cost monitoring stopped")
        optimizer.print_status()

def check_training_process():
    """Check if training process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train'], capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def stop_training_process():
    """Gracefully stop training process"""
    try:
        # Find training process
        result = subprocess.run(['pgrep', '-f', 'train'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"🛑 Stopping training process {pid}")
                subprocess.run(['kill', '-TERM', pid])
    except Exception as e:
        print(f"Error stopping training: {e}")

def estimate_training_cost():
    """Estimate training cost before starting"""
    print("📊 Training Cost Estimator")
    print("=" * 40)
    
    # Get user inputs
    try:
        dataset_size = int(input("Dataset size (number of images): ") or "10000")
        batch_size = int(input("Batch size: ") or "24")
        epochs = int(input("Number of epochs: ") or "25")
        cost_per_hour = float(input("RunPod cost per hour ($): ") or "0.40")
    except ValueError:
        print("Invalid input, using defaults")
        dataset_size, batch_size, epochs, cost_per_hour = 10000, 24, 25, 0.40
    
    # Estimate training time
    batches_per_epoch = dataset_size / batch_size
    seconds_per_batch = 1.5  # Estimate for RTX 4090
    
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * seconds_per_batch
    total_hours = total_seconds / 3600
    
    # Add overhead (data loading, validation, etc.)
    total_hours *= 1.3
    
    total_cost = total_hours * cost_per_hour
    
    print(f"\n📈 Estimates:")
    print(f"   Training time: {total_hours:.1f} hours")
    print(f"   Total cost: ${total_cost:.2f}")
    print(f"   Batches per epoch: {batches_per_epoch:.0f}")
    print(f"   Total batches: {total_batches:.0f}")
    
    if total_cost > 10:
        print(f"\n⚠️  Estimated cost (${total_cost:.2f}) exceeds $10 budget!")
        print("💡 Suggestions to reduce cost:")
        print("   • Reduce epochs to", int(epochs * 10 / total_cost))
        print("   • Reduce dataset size to", int(dataset_size * 10 / total_cost))
        print("   • Use spot instances (30-50% cheaper)")
        print("   • Use smaller GPU (RTX 3090)")
    else:
        print(f"\n✅ Training should fit within $10 budget")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RunPod Cost Optimizer')
    parser.add_argument('--monitor', action='store_true', help='Monitor training process')
    parser.add_argument('--estimate', action='store_true', help='Estimate training cost')
    parser.add_argument('--budget', type=float, default=10.0, help='Budget limit')
    parser.add_argument('--rate', type=float, default=0.40, help='Cost per hour')
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_training_cost()
    elif args.monitor:
        monitor_training_process()
    else:
        print("Use --monitor to monitor training or --estimate to estimate costs")

if __name__ == "__main__":
    main()