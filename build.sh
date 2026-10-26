#!/usr/bin/env bash
# Exit immediately on error
set -o errexit

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p vector_db

# Download required NLTK corpora
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

echo "RAGex build completed successfully!"
