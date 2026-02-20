#!/bin/bash
# Kronos Installation Script

echo "=========================================="
echo "Installing Kronos Foundation Model"
echo "=========================================="

# Step 1: Clone Kronos repository
echo ""
echo "Step 1: Cloning Kronos repository..."
if [ -d "Kronos" ]; then
    echo "Kronos directory already exists. Pulling latest changes..."
    cd Kronos
    git pull
    cd ..
else
    git clone https://github.com/shiyu-coder/Kronos.git
fi

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing Kronos dependencies..."
cd Kronos

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Installing common dependencies..."
    pip install torch transformers datasets numpy pandas
fi

# Step 3: Add Kronos to Python path
echo ""
echo "Step 3: Adding Kronos to Python path..."
cd ..
KRONOS_PATH="$(pwd)/Kronos"
echo ""
echo "Kronos installed at: $KRONOS_PATH"
echo ""
echo "To use Kronos, you have two options:"
echo ""
echo "Option 1: Set PYTHONPATH environment variable (temporary):"
echo "  export PYTHONPATH=\"$KRONOS_PATH:\$PYTHONPATH\""
echo ""
echo "Option 2: Add to your shell profile (permanent):"
echo "  echo 'export PYTHONPATH=\"$KRONOS_PATH:\$PYTHONPATH\"' >> ~/.bash_profile"
echo "  source ~/.bash_profile"
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set PYTHONPATH (see options above)"
echo "2. Run: python scripts/run_chapter8_kronos.py --mode full"
echo ""
