#!/bin/bash

# Upload large files to Hugging Face Hub
# Run this script once to upload all data files to Hugging Face

REPO_ID="jgrizou/neurips-self-calibrating-bci-data"

echo "========================================="
echo "Uploading files to Hugging Face Hub"
echo "Repository: $REPO_ID"
echo "========================================="
echo ""

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "ERROR: hf CLI is not installed."
    echo ""
    echo "Please install it:"
    echo "  pip install huggingface-hub"
    echo ""
    exit 1
fi

echo "Note: If you haven't logged in yet, run: hf auth login"
echo ""

# Upload checkpoint
echo "[1/3] Uploading GAN checkpoint..."
hf upload $REPO_ID \
    checkpoints/pggan_celebahq1024.pth \
    checkpoints/pggan_celebahq1024.pth \
    --repo-type=dataset

if [ $? -eq 0 ]; then
    echo "✓ Checkpoint uploaded successfully"
else
    echo "✗ Failed to upload checkpoint"
    exit 1
fi
echo ""

# Upload all parquet files
echo "[2/3] Uploading analysis results (parquet files)..."
hf upload $REPO_ID \
    src/analysis/final_dfs \
    final_dfs \
    --repo-type=dataset

if [ $? -eq 0 ]; then
    echo "✓ Parquet files uploaded successfully"
else
    echo "✗ Failed to upload parquet files"
    exit 1
fi
echo ""

# Upload plot files
echo "[3/3] Uploading plot files..."

PLOT_FILES=(
    "diagonal.eps"
    "diagonal.png"
    "diagonal.svg"
    "random.eps"
    "random.png"
    "random.svg"
)

for file in "${PLOT_FILES[@]}"; do
    echo "  Uploading $file..."
    hf upload $REPO_ID \
        "src/analysis/plots/faces/$file" \
        "plots/$file" \
        --repo-type=dataset

    if [ $? -ne 0 ]; then
        echo "✗ Failed to upload $file"
        exit 1
    fi
done

echo "✓ All plot files uploaded successfully"
echo ""

echo "========================================="
echo "✓ All files uploaded successfully!"
echo "========================================="
echo ""
echo "Your dataset is available at:"
echo "https://huggingface.co/datasets/$REPO_ID"
echo ""
echo "Users can now download files by running:"
echo "  python scripts/download_data.py"
