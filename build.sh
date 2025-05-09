#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p knowledge-base
mkdir -p vector_db

# Download NLTK data
python -c "import nltk; nltk.download('wordnet')"

# Print success message
echo "Build completed successfully!"