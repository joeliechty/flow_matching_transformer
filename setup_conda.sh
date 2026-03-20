#!/bin/bash

exec > >(tee -i setup_log.txt) 2>&1

# Create and activate conda environment (used instead of venv for isolation)
echo "Creating and activating conda environment 'FmT'..."

# Initialize conda for bash if not already done
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found. Please run 'conda init bash' and restart your shell."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q "FmT"; then
    echo "Conda environment 'FmT' already exists. Activating it..."
else
    echo "Conda environment 'FmT' does not exist. Creating it..."
    conda create -n FmT python=3.12 -y
fi

conda activate FmT

pip install -r requirements.txt

