#!/bin/bash
# Wrapper script to run Kronos FULL evaluation with correct PYTHONPATH

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Kronos directory exists
if [ ! -d "Kronos" ]; then
    echo "Error: Kronos directory not found."
    echo ""
    echo "Please install Kronos first:"
    echo "  ./INSTALL_KRONOS.sh"
    exit 1
fi

# Set PYTHONPATH to include Kronos
export PYTHONPATH="$SCRIPT_DIR/Kronos:$PYTHONPATH"

echo "=========================================="
echo "Running Kronos FULL Evaluation"
echo "=========================================="
echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo ""
echo "This will take approximately 2-4 hours."
echo "Press Ctrl+C to cancel (within 5 seconds)..."
echo ""
sleep 5

# Run the evaluation
python scripts/run_chapter8_kronos.py --mode full "$@"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  evaluation_outputs/chapter8_kronos_full/"
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat evaluation_outputs/chapter8_kronos_full/chapter8_kronos_full/REPORT_SUMMARY.md"
    echo "  2. Compare vs baselines (see CHAPTER_8_NEXT_ACTIONS.md)"
    echo "  3. Proceed to Phase 4 (Comparison & Freeze)"
else
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Please review the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - Kronos not installed: Run ./INSTALL_KRONOS.sh"
    echo "  - Missing dependencies: pip install torch transformers datasets"
    echo "  - Insufficient memory: Try --device cpu or reduce batch size"
fi

exit $EXIT_CODE

