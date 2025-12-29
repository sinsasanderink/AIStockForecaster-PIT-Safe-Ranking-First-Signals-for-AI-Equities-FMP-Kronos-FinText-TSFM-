"""
Tests for PIT Scanner
=====================

Validates that the PIT violation scanner works correctly.

Run with: python tests/test_pit_scanner.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

import logging
logging.basicConfig(level=logging.WARNING)

from src.features.pit_scanner import run_pit_scan, PITScanner


def test_pit_scanner():
    """Test that PIT scanner runs and generates report."""
    print("\n" + "="*70)
    print("PIT SCANNER TEST")
    print("="*70)
    
    violations, report = run_pit_scan()
    
    print(report)
    
    # Check report structure
    assert "PIT VIOLATION SCANNER REPORT" in report
    assert "PIT RULES:" in report
    
    # Count severity levels
    critical = [v for v in violations if v.severity == "CRITICAL"]
    high = [v for v in violations if v.severity == "HIGH"]  
    medium = [v for v in violations if v.severity == "MEDIUM"]
    
    print(f"\n📊 Summary:")
    print(f"   CRITICAL: {len(critical)}")
    print(f"   HIGH: {len(high)}")
    print(f"   MEDIUM: {len(medium)}")
    print(f"   TOTAL: {len(violations)}")
    
    # Assert no CRITICAL violations
    if critical:
        print(f"\n🚨 CRITICAL VIOLATIONS FOUND:")
        for v in critical:
            print(f"   • {v.module}.{v.function}: {v.description}")
        assert False, "CRITICAL PIT violations detected - must fix before Chapter 6"
    
    # Warn about HIGH violations but don't fail
    if high:
        print(f"\n⚠️  HIGH VIOLATIONS (should be addressed):")
        for v in high:
            print(f"   • {v.module}.{v.function}: {v.description}")
        # Don't fail - these are warnings
    
    # MEDIUM violations are expected (false positives from static analysis)
    if medium:
        print(f"\nℹ️  MEDIUM VIOLATIONS (likely false positives):")
        for v in medium:
            print(f"   • {v.module}.{v.function}: {v.description}")
    
    print("\n✅ PIT Scanner test passed")
    print("="*70)
    
    return len(critical) == 0


if __name__ == "__main__":
    success = test_pit_scanner()
    sys.exit(0 if success else 1)

