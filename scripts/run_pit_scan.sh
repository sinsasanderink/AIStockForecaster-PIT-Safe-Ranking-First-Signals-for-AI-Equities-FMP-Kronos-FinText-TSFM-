#!/bin/bash
# Pre-commit PIT Scanner
# Run this before committing to ensure no PIT violations

echo "=================================================="
echo "Running PIT Scanner (Pre-commit Check)"
echo "=================================================="
echo ""

cd "$(dirname "$0")/.." || exit 1

python3 src/features/pit_scanner.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ PIT Scanner PASSED - no critical violations"
    echo "   Safe to commit"
else
    echo ""
    echo "❌ PIT Scanner FAILED - critical violations found"
    echo "   Fix violations before committing"
fi

exit $exit_code

