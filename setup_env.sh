#!/bin/bash
# Environment setup script for FoNE project
# Source this script to set up environment variables: source setup_env.sh

# Project configuration
export PROJECT_ID="bfdj"
export PROJECT_DIR="/projects/${PROJECT_ID}/${USER}"
export WORK_NVME_DIR="/work/nvme/${PROJECT_ID}/${USER}"
export WORK_HDD_DIR="/work/hdd/${PROJECT_ID}/${USER}"

# Create directories if they don't exist
echo "🔧 Setting up FoNE environment..."
echo "PROJECT_ID: $PROJECT_ID"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "WORK_NVME_DIR: $WORK_NVME_DIR"
echo "WORK_HDD_DIR: $WORK_HDD_DIR"

# Check if directories exist and create them if needed
for dir in "$PROJECT_DIR" "$WORK_NVME_DIR" "$WORK_HDD_DIR"; do
    if [ -n "$dir" ] && [ -d "$(dirname "$dir")" ]; then
        mkdir -p "$dir"
        if [ -d "$dir" ]; then
            echo "✅ Directory ready: $dir"
        else
            echo "❌ Failed to create: $dir"
        fi
    else
        echo "⚠️  Parent directory doesn't exist for: $dir"
    fi
done

# Create outputs subdirectories
for base_dir in "$PROJECT_DIR" "$WORK_HDD_DIR"; do
    if [ -d "$base_dir" ]; then
        mkdir -p "$base_dir/outputs"
        echo "✅ Created outputs directory in: $base_dir"
    fi
done

echo "🚀 Environment setup complete!"
echo ""
echo "Usage:"
echo "  Checkpoints will be saved to: $WORK_HDD_DIR/outputs/"
echo "  Evaluation results will be saved to: $PROJECT_DIR/outputs/"
echo "  Dataset cache will use: $WORK_HDD_DIR/.cache/"
