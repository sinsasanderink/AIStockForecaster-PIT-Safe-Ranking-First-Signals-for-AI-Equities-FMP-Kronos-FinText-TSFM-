"""
Environment Variable Utilities
==============================

Shared helpers for:
- Auto-loading .env from repo root
- Resolving FMP API keys with proper precedence
- Never logging secrets

Usage (at top of any script):
    from src.utils.env import load_repo_dotenv, resolve_fmp_key
    
    load_repo_dotenv()  # Auto-loads .env from repo root if present
    api_key = resolve_fmp_key()  # Raises RuntimeError if not found
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    """
    Find the repository root directory.
    
    Searches upward from this file for a directory containing:
    - .git directory, OR
    - pyproject.toml, OR
    - requirements.txt
    
    Returns:
        Path to repo root
    
    Raises:
        RuntimeError if repo root cannot be found
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent
    
    # Also check from cwd as fallback
    check_paths = [current]
    cwd = Path.cwd()
    if cwd != current:
        check_paths.append(cwd)
    
    for start_path in check_paths:
        path = start_path
        for _ in range(10):  # Limit search depth
            if (path / ".git").exists():
                return path
            if (path / "pyproject.toml").exists():
                return path
            if (path / "requirements.txt").exists():
                return path
            if path.parent == path:
                break
            path = path.parent
    
    # Fallback: assume we're somewhere in the repo
    # Return cwd as best guess
    return cwd


def load_repo_dotenv(dotenv_path: Optional[Path] = None) -> bool:
    """
    Load .env file from repo root if present.
    
    This function is safe to call multiple times - subsequent calls
    will not override already-set environment variables.
    
    Args:
        dotenv_path: Optional explicit path to .env file.
                     If not provided, searches for .env in repo root.
    
    Returns:
        True if .env was found and loaded, False otherwise
    
    Notes:
        - Uses python-dotenv if available
        - Falls back to simple parsing if python-dotenv not installed
        - NEVER logs the contents of .env or any secrets
    """
    if dotenv_path is None:
        repo_root = get_repo_root()
        dotenv_path = repo_root / ".env"
    
    if not dotenv_path.exists():
        logger.debug(f".env not found at {dotenv_path}")
        return False
    
    # Try python-dotenv first (preferred)
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path, override=False)
        if loaded:
            logger.debug(f"Loaded .env from {dotenv_path}")
        return loaded
    except ImportError:
        pass
    
    # Fallback: simple .env parsing
    try:
        loaded_count = 0
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=value (handle quotes)
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    # Only set if not already in environment (no override)
                    if key and key not in os.environ:
                        os.environ[key] = value
                        loaded_count += 1
        
        if loaded_count > 0:
            logger.debug(f"Loaded {loaded_count} vars from {dotenv_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Error reading .env: {e}")
        return False


def resolve_fmp_key(cli_key: Optional[str] = None) -> str:
    """
    Resolve FMP API key with proper precedence.
    
    Priority (highest to lowest):
    1. cli_key argument (from --api-key CLI flag)
    2. FMP_KEYS environment variable (comma-separated, uses FIRST key)
    3. FMP_API_KEY environment variable
    
    Args:
        cli_key: Optional key provided via CLI argument
    
    Returns:
        API key string (never logged)
    
    Raises:
        RuntimeError with helpful message if no key found
    
    Security:
        - NEVER logs the actual key value
        - Only logs which source was used
    """
    # 1. CLI argument takes precedence
    if cli_key and cli_key.strip():
        logger.debug("Using API key from CLI argument")
        return cli_key.strip()
    
    # 2. FMP_KEYS (comma-separated, use first non-empty)
    fmp_keys = os.environ.get("FMP_KEYS", "")
    if fmp_keys:
        keys = [k.strip() for k in fmp_keys.split(",") if k.strip()]
        if keys:
            logger.debug(f"Using FMP_KEYS (found {len(keys)} keys, using first)")
            return keys[0]
    
    # 3. FMP_API_KEY fallback
    fmp_api_key = os.environ.get("FMP_API_KEY", "")
    if fmp_api_key.strip():
        logger.debug("Using FMP_API_KEY")
        return fmp_api_key.strip()
    
    # No key found - raise helpful error
    raise RuntimeError(
        "FMP API key not found!\n"
        "\n"
        "Options (in priority order):\n"
        "  1. Pass via CLI:     --api-key YOUR_KEY\n"
        "  2. Set in .env:      FMP_KEYS=your_key\n"
        "  3. Export env var:   export FMP_KEYS='your_key'\n"
        "\n"
        "For comma-separated keys (rotation), use:\n"
        "  FMP_KEYS=key1,key2,key3\n"
        "  (The first non-empty key is used deterministically)\n"
        "\n"
        "Note: .env file should be in the repository root.\n"
        "      Scripts auto-load it if present."
    )

