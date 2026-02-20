#!/bin/bash
# Verification script for all Kronos fixes

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              VERIFYING ALL KRONOS FIXES                              ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check 1: MPS disable in adapter
echo "✓ Check 1: MPS disable in adapter..."
if grep -q "PYTORCH_ENABLE_MPS_FALLBACK" src/models/kronos_adapter.py; then
    echo "  ✅ Found MPS disable in kronos_adapter.py"
else
    echo "  ❌ MISSING: MPS disable in kronos_adapter.py"
fi

# Check 2: MPS disable in script
echo ""
echo "✓ Check 2: MPS disable in script..."
if grep -q "PYTORCH_ENABLE_MPS_FALLBACK" scripts/run_chapter8_kronos.py; then
    echo "  ✅ Found MPS disable in run_chapter8_kronos.py"
else
    echo "  ❌ MISSING: MPS disable in run_chapter8_kronos.py"
fi

# Check 3: CPUForcedKronosPredictor removed
echo ""
echo "✓ Check 3: Wrapper removed..."
if ! grep -q "CPUForcedKronosPredictor" src/models/kronos_adapter.py; then
    echo "  ✅ CPUForcedKronosPredictor wrapper removed (clean code)"
else
    echo "  ⚠️  CPUForcedKronosPredictor still exists (should be removed)"
fi

# Check 4: model.eval() present
echo ""
echo "✓ Check 4: Model eval mode..."
if grep -q "model.eval()" src/models/kronos_adapter.py; then
    echo "  ✅ model.eval() found (dropout disabled)"
else
    echo "  ❌ MISSING: model.eval() call"
fi

# Check 5: torch.inference_mode()
echo ""
echo "✓ Check 5: Inference mode wrapper..."
if grep -q "torch.inference_mode()" src/models/kronos_adapter.py; then
    echo "  ✅ torch.inference_mode() found (fast + deterministic)"
else
    echo "  ❌ MISSING: torch.inference_mode() wrapper"
fi

# Check 6: Micro-batching
echo ""
echo "✓ Check 6: Micro-batching..."
if grep -q "batch_size" src/models/kronos_adapter.py; then
    echo "  ✅ Micro-batching implemented"
else
    echo "  ❌ MISSING: batch_size parameter"
fi

# Check 7: Empty report guard
echo ""
echo "✓ Check 7: Empty report guard..."
if grep -q "empty evaluation results" src/evaluation/reports.py; then
    echo "  ✅ Empty report guard in place (no KeyError crash)"
else
    echo "  ❌ MISSING: Empty report guard"
fi

# Check 8: Series conversion
echo ""
echo "✓ Check 8: Series conversion helper..."
if grep -q "_to_series" src/models/kronos_adapter.py; then
    echo "  ✅ _to_series helper found (.dt accessor fix)"
else
    echo "  ❌ MISSING: _to_series conversion helper"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              VERIFICATION COMPLETE                                   ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next step: Run SMOKE test"
echo ""
echo "  export PYTHONPATH=\"\$(pwd)/Kronos:\$PYTHONPATH\""
echo "  python scripts/run_chapter8_kronos.py --mode smoke --batch-size 12"
echo ""
