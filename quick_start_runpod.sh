#!/bin/bash

# One-command setup and training for RunPod RTX 4090
# Usage: curl -sSL https://raw.githubusercontent.com/yourusername/visual-intelligence-pipeline/main/quick_start_runpod.sh | bash

set -e

echo "🚀 Visual Intelligence Pipeline - RunPod RTX 4090 Quick Start"
echo "============================================================"

# Change to workspace directory
cd /workspace

# Download the setup script
if [ ! -f "runpod_setup.sh" ]; then
    echo "📥 Downloading setup script..."
    # In practice, you would download from your repository
    # wget https://raw.githubusercontent.com/yourusername/visual-intelligence-pipeline/main/runpod_setup.sh
    echo "Please upload the Python files and setup script to this directory first."
    exit 1
fi

# Make setup script executable
chmod +x runpod_setup.sh

# Run setup
echo "🔧 Running setup..."
./runpod_setup.sh

# Navigate to project directory
cd visual-intelligence-pipeline

# Check if user wants to download datasets immediately
echo ""
read -p "📥 Download datasets now? This will take 2-4 hours and use ~100GB storage (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Starting dataset download..."
    ./download_datasets.sh
    
    echo "🔄 Preprocessing data..."
    python train.py --preprocess --source-datasets config/dataset_paths.json --data-dir data/processed
    
    echo ""
    read -p "🏋️ Start training now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Starting training..."
        ./start_training.sh
    else
        echo "✅ Setup complete! Run './start_training.sh' when ready to train."
    fi
else
    echo "✅ Setup complete! Run './download_datasets.sh' when ready to download datasets."
fi

echo ""
echo "🎉 Visual Intelligence Pipeline is ready!"
echo ""
echo "📋 Quick commands:"
echo "   ./download_datasets.sh    # Download datasets (run once)"
echo "   ./start_training.sh       # Start training"
echo "   python monitor_training.py # Monitor training progress"
echo "   python test_setup.py      # Test setup"
echo ""
echo "📁 Project location: /workspace/visual-intelligence-pipeline"
echo "📖 See RUNPOD_README.md for detailed instructions"