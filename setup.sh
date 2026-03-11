#!/bin/bash

ENV_NAME="Diffusion-Models-ENSO"

echo "🔧 Creating conda environment: $ENV_NAME"
conda env create -f environment.yml || conda env update -f environment.yml

echo "✅ Activating environment and installing Poetry dependencies..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "📦 Installing Poetry dependencies..."
poetry install

echo "🎉 Setup complete. To activate the environment, run:"
echo "   conda activate $ENV_NAME"
