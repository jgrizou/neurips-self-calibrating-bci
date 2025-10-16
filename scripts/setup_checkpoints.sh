#!/bin/bash

# Setup GAN checkpoints in required locations
# Copies the checkpoint from the canonical location to all locations where it's needed

CHECKPOINT_SOURCE="checkpoints/pggan_celebahq1024.pth"
CHECKPOINT_TARGETS=(
    "src/user_experiments/checkpoints/pggan_celebahq1024.pth"
    "src/analysis/plots/optim/checkpoints/pggan_celebahq1024.pth"
    "src/analysis/plots/faces/checkpoints/pggan_celebahq1024.pth"
)

echo "Setting up GAN model checkpoints..."

# Check if source exists
if [ ! -f "$CHECKPOINT_SOURCE" ]; then
    echo "Error: Source checkpoint not found at $CHECKPOINT_SOURCE"
    echo "Please ensure the checkpoint file exists in the checkpoints/ directory"
    exit 1
fi

# Copy to each target location
for target in "${CHECKPOINT_TARGETS[@]}"; do
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$target")"

    if [ -f "$target" ]; then
        echo "Checkpoint already exists at $target (skipping)"
    else
        echo "Copying checkpoint to $target"
        cp "$CHECKPOINT_SOURCE" "$target"
    fi
done

echo "Done! All checkpoints are in place."
