#!/bin/bash

# Enhanced Polar Registration Training Script
# Usage: ./run_training.sh

echo "Starting Enhanced Polar Registration Training..."
echo "=============================================="

# Set default parameters
CONFIG_FILE="configs/default_config.json"
DEVICE="cuda:4"
LOG_LEVEL="INFO"

# Override with command line arguments if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config CONFIG_FILE] [--device DEVICE] [--log-level LOG_LEVEL]"
            echo "  --config: Path to configuration file (default: configs/default_config.json)"
            echo "  --device: Device to train on (default: cuda:0)"
            echo "  --log-level: Logging level (default: INFO)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Create checkpoints directory
mkdir -p checkpoints

# Run training
echo "Configuration: $CONFIG_FILE"
echo "Device: $DEVICE"
echo "Log Level: $LOG_LEVEL"
echo ""

python main.py \
    --config "$CONFIG_FILE" \
    --device "$DEVICE" \
    --log_level "$LOG_LEVEL"

echo ""
echo "Training completed!"
echo "Check the checkpoints/ directory for saved models and logs."
