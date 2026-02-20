#!/usr/bin/env python
"""
Test script to verify Kronos installation.

Usage:
    python scripts/test_kronos_installation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("KRONOS INSTALLATION TEST")
print("=" * 70)

# Test 1: Check if Kronos can be imported
print("\nTest 1: Checking if Kronos can be imported...")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("✓ SUCCESS: Kronos imports successfully")
    KRONOS_AVAILABLE = True
except ImportError as e:
    print(f"✗ FAILED: Cannot import Kronos")
    print(f"  Error: {e}")
    print("\n  Solution:")
    print("  1. Run: ./INSTALL_KRONOS.sh")
    print("  2. Set PYTHONPATH: export PYTHONPATH=\"$(pwd)/Kronos:$PYTHONPATH\"")
    print("  3. Re-run this test")
    KRONOS_AVAILABLE = False

# Test 2: Check if we can load the tokenizer
if KRONOS_AVAILABLE:
    print("\nTest 2: Loading Kronos tokenizer...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        print("✓ SUCCESS: Tokenizer loaded from HuggingFace")
    except Exception as e:
        print(f"✗ FAILED: Cannot load tokenizer")
        print(f"  Error: {e}")
        print("\n  This will download ~100MB on first run.")
        print("  Make sure you have internet connection.")
        KRONOS_AVAILABLE = False

# Test 3: Check if we can load the model
if KRONOS_AVAILABLE:
    print("\nTest 3: Loading Kronos model...")
    try:
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✓ SUCCESS: Model loaded from HuggingFace")
        print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"✗ FAILED: Cannot load model")
        print(f"  Error: {e}")
        print("\n  This will download ~500MB on first run.")
        print("  Make sure you have internet connection and sufficient disk space.")
        KRONOS_AVAILABLE = False

# Test 4: Check if we can create the predictor
if KRONOS_AVAILABLE:
    print("\nTest 4: Creating Kronos predictor...")
    try:
        predictor = KronosPredictor(model=model, tokenizer=tokenizer, max_context=512)
        print("✓ SUCCESS: Predictor created")
        print(f"  Predictor type: {type(predictor)}")
    except Exception as e:
        print(f"✗ FAILED: Cannot create predictor")
        print(f"  Error: {e}")
        KRONOS_AVAILABLE = False

# Test 5: Check if our adapter can be imported
print("\nTest 5: Checking if KronosAdapter can be imported...")
try:
    from src.models.kronos_adapter import KronosAdapter
    print("✓ SUCCESS: KronosAdapter imported")
except ImportError as e:
    print(f"✗ FAILED: Cannot import KronosAdapter")
    print(f"  Error: {e}")
    KRONOS_AVAILABLE = False

# Test 6: Check if adapter can be initialized
if KRONOS_AVAILABLE:
    print("\nTest 6: Initializing KronosAdapter...")
    try:
        adapter = KronosAdapter.from_pretrained(
            db_path="data/features.duckdb",
            use_stub=False,
            lookback=252,
            deterministic=True,
        )
        print("✓ SUCCESS: KronosAdapter initialized")
        print(f"  Device: {adapter.device}")
        print(f"  Lookback: {adapter.lookback}")
        print(f"  Trading calendar: {len(adapter.trading_calendar)} dates")
    except Exception as e:
        print(f"✗ FAILED: Cannot initialize KronosAdapter")
        print(f"  Error: {e}")
        KRONOS_AVAILABLE = False

# Final verdict
print("\n" + "=" * 70)
if KRONOS_AVAILABLE:
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nKronos is ready to use!")
    print("\nNext step:")
    print("  python scripts/run_chapter8_kronos.py --mode full")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    print("=" * 70)
    print("\nKronos is not ready. Please fix the issues above.")
    print("\nQuick fix:")
    print("  1. Run: ./INSTALL_KRONOS.sh")
    print("  2. Set: export PYTHONPATH=\"$(pwd)/Kronos:$PYTHONPATH\"")
    print("  3. Re-run: python scripts/test_kronos_installation.py")
    sys.exit(1)

