"""
AI Stock Universe Definition
============================

Defines the ~100 AI-related stocks across 10 subcategories.

Categories:
1. AI Compute & Core Semiconductors (GPUs/CPUs/ASICs + key chipmakers)
2. Semiconductor Manufacturing, Equipment & EDA (the "shovels" of AI chips)
3. Networking, Optics & Data-Center Hardware (moving AI data fast)
4. Power, Cooling, Real Estate & Picks-and-Shovels for AI Data Centers
5. Cloud/Hyperscalers & AI Platforms (where models run + get distributed)
6. Data Platforms & Enterprise Software (AI-enabled workflows + analytics)
7. Cybersecurity & Identity (AI used for detection + response)
8. Robotics, Automation & Industrial AI (physical-world AI)
9. Autonomy, Drones & Defense AI (edge autonomy + ISR)
10. AI-as-a-Product, Ads & Other Notable Adopters (AI is core to monetization)
"""

from typing import Dict, List, Set, Optional


# =============================================================================
# AI Universe by Category
# =============================================================================

AI_UNIVERSE: Dict[str, List[str]] = {
    # 1) AI Compute & Core Semiconductors (GPUs/CPUs/ASICs + key chipmakers)
    "ai_compute_core_semis": [
        "NVDA",   # NVIDIA - GPUs, AI training/inference leader
        "AMD",    # AMD - GPUs, CPUs, data center
        "AVGO",   # Broadcom - AI networking, custom chips
        "QCOM",   # Qualcomm - mobile AI, edge inference
        "INTC",   # Intel - CPUs, AI accelerators
        "ARM",    # ARM Holdings - chip architecture
        "MU",     # Micron - memory for AI workloads
        "MRVL",   # Marvell - custom AI silicon
        "TXN",    # Texas Instruments - analog/embedded
        "ADI",    # Analog Devices - signal processing
        "NXPI",   # NXP Semiconductors - automotive AI
        "MCHP",   # Microchip Technology
        "ON",     # ON Semiconductor - power, sensing
        "STM",    # STMicroelectronics
        "GFS",    # GlobalFoundries - chip manufacturing
        "LSCC",   # Lattice Semiconductor - FPGAs
        "TER",    # Teradyne - chip testing
        "MPWR",   # Monolithic Power Systems
    ],
    
    # 2) Semiconductor Manufacturing, Equipment & EDA
    "semicap_eda_manufacturing": [
        "TSM",    # TSMC - leading-edge foundry
        "ASML",   # ASML - EUV lithography monopoly
        "AMAT",   # Applied Materials - deposition, etch
        "LRCX",   # Lam Research - etch, deposition
        "KLAC",   # KLA Corporation - inspection
        "TEL",    # Tokyo Electron (ADR)
        "ENTG",   # Entegris - materials
        "MKSI",   # MKS Instruments
        "ACMR",   # ACM Research - cleaning equipment
        "ONTO",   # Onto Innovation - metrology
        "CDNS",   # Cadence - EDA software
        "SNPS",   # Synopsys - EDA software
    ],
    
    # 3) Networking, Optics & Data-Center Hardware
    "networking_optics_dc_hw": [
        "ANET",   # Arista Networks - data center switches
        "CSCO",   # Cisco - networking
        "JNPR",   # Juniper Networks
        "CIEN",   # Ciena - optical networking
        "LITE",   # Lumentum - optical components
        "COHR",   # Coherent - lasers, optics
        "DELL",   # Dell Technologies - servers
        "HPE",    # Hewlett Packard Enterprise
        "SMCI",   # Super Micro Computer - AI servers
        "APH",    # Amphenol - connectors
    ],
    
    # 4) Power, Cooling, Real Estate & Data Center Infrastructure
    "datacenter_power_cooling_reits": [
        "VRT",    # Vertiv - power, cooling
        "ETN",    # Eaton - power management
        "PWR",    # Quanta Services - infrastructure
        "TT",     # Trane Technologies - HVAC
        "JCI",    # Johnson Controls - building automation
        "CARR",   # Carrier Global - HVAC
        "EQIX",   # Equinix - data center REIT
        "DLR",    # Digital Realty - data center REIT
    ],
    
    # 5) Cloud/Hyperscalers & AI Platforms
    "cloud_platforms_model_owners": [
        "MSFT",   # Microsoft - Azure, OpenAI partnership
        "AMZN",   # Amazon - AWS, Bedrock
        "GOOGL",  # Alphabet - GCP, DeepMind, Gemini
        "META",   # Meta - LLaMA, AI infrastructure
        "ORCL",   # Oracle - cloud, enterprise AI
        "IBM",    # IBM - watsonx, enterprise AI
        "AAPL",   # Apple - on-device AI
        "TSLA",   # Tesla - FSD, Dojo, robotics
    ],
    
    # 6) Data Platforms & Enterprise Software
    "data_platforms_enterprise_sw": [
        "PLTR",   # Palantir - AI/ML platforms
        "SNOW",   # Snowflake - data cloud
        "MDB",    # MongoDB - database
        "DDOG",   # Datadog - observability
        "NOW",    # ServiceNow - workflow automation
        "CRM",    # Salesforce - CRM, Einstein AI
        "ADBE",   # Adobe - creative AI, Firefly
        "INTU",   # Intuit - financial AI
        "SHOP",   # Shopify - e-commerce AI
        "TEAM",   # Atlassian - collaboration
        "U",      # Unity - real-time 3D, AI
        "PATH",   # UiPath - RPA, automation
        "TWLO",   # Twilio - communications API
        "OKTA",   # Okta - identity
        "NET",    # Cloudflare - edge, AI inference
        "DT",     # Dynatrace - observability
        "INFA",   # Informatica - data management
        "AKAM",   # Akamai - CDN, edge
    ],
    
    # 7) Cybersecurity & Identity
    "cybersecurity": [
        "CRWD",   # CrowdStrike - endpoint, AI-native
        "PANW",   # Palo Alto Networks
        "FTNT",   # Fortinet - firewalls
        "ZS",     # Zscaler - zero trust
        "S",      # SentinelOne - AI security
        "CYBR",   # CyberArk - identity security
        "TENB",   # Tenable - vulnerability mgmt
        "CHKP",   # Check Point
    ],
    
    # 8) Robotics, Automation & Industrial AI
    "robotics_industrial_ai": [
        "ISRG",   # Intuitive Surgical - surgical robots
        "ABB",    # ABB - industrial automation
        "ROK",    # Rockwell Automation
        "HON",    # Honeywell - industrial AI
        "SYM",    # Symbotic - warehouse automation
        "CGNX",   # Cognex - machine vision
    ],
    
    # 9) Autonomy, Drones & Defense AI
    "autonomy_defense_ai": [
        "MBLY",   # Mobileye - autonomous driving
        "AVAV",   # AeroVironment - drones
        "KTOS",   # Kratos Defense - drones, AI
        "LMT",    # Lockheed Martin - defense AI
        "NOC",    # Northrop Grumman - defense
        "RTX",    # RTX (Raytheon) - defense
    ],
    
    # 10) AI-as-a-Product, Ads & Other Notable Adopters
    "ai_apps_ads_misc": [
        "APP",    # AppLovin - AI-driven mobile ads
        "TTD",    # The Trade Desk - programmatic ads
        "UBER",   # Uber - AI for matching, pricing
        "DUOL",   # Duolingo - AI-powered learning
        "AI",     # C3.ai - enterprise AI
        "SOUN",   # SoundHound AI - voice AI
    ],
}


# =============================================================================
# Category Metadata
# =============================================================================

AI_CATEGORIES: List[str] = list(AI_UNIVERSE.keys())

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "ai_compute_core_semis": "AI Compute & Core Semiconductors (GPUs/CPUs/ASICs)",
    "semicap_eda_manufacturing": "Semiconductor Manufacturing, Equipment & EDA",
    "networking_optics_dc_hw": "Networking, Optics & Data-Center Hardware",
    "datacenter_power_cooling_reits": "Power, Cooling & Data Center Infrastructure",
    "cloud_platforms_model_owners": "Cloud/Hyperscalers & AI Platforms",
    "data_platforms_enterprise_sw": "Data Platforms & Enterprise Software",
    "cybersecurity": "Cybersecurity & Identity",
    "robotics_industrial_ai": "Robotics, Automation & Industrial AI",
    "autonomy_defense_ai": "Autonomy, Drones & Defense AI",
    "ai_apps_ads_misc": "AI-as-a-Product, Ads & Other Adopters",
}


# =============================================================================
# Accessor Functions
# =============================================================================

def get_all_tickers() -> List[str]:
    """
    Return sorted unique list of all AI tickers across all categories.
    
    Returns:
        Sorted list of ~100 ticker symbols
    """
    all_tickers: Set[str] = set()
    for tickers in AI_UNIVERSE.values():
        all_tickers.update(tickers)
    return sorted(all_tickers)


def get_tickers_by_category(category: str) -> List[str]:
    """
    Get tickers for a specific category.
    
    Args:
        category: Category key (e.g., "ai_compute_core_semis")
    
    Returns:
        List of tickers in that category
    
    Raises:
        KeyError: If category doesn't exist
    """
    if category not in AI_UNIVERSE:
        raise KeyError(
            f"Unknown category: {category}. "
            f"Valid categories: {AI_CATEGORIES}"
        )
    return AI_UNIVERSE[category].copy()


def get_tickers_by_categories(categories: List[str]) -> List[str]:
    """
    Get unique tickers for multiple categories.
    
    Args:
        categories: List of category keys
    
    Returns:
        Sorted unique list of tickers from those categories
    """
    tickers: Set[str] = set()
    for cat in categories:
        tickers.update(get_tickers_by_category(cat))
    return sorted(tickers)


def get_category_for_ticker(ticker: str) -> Optional[str]:
    """
    Find which category a ticker belongs to.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Category key or None if not found
    """
    for category, tickers in AI_UNIVERSE.items():
        if ticker in tickers:
            return category
    return None


def get_ticker_metadata(ticker: str) -> Optional[Dict]:
    """
    Get metadata for a ticker including its category.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dict with category info or None if not in universe
    """
    category = get_category_for_ticker(ticker)
    if category is None:
        return None
    
    return {
        "ticker": ticker,
        "category": category,
        "category_description": CATEGORY_DESCRIPTIONS[category],
    }


# =============================================================================
# Validation
# =============================================================================

def validate_universe(expected_size: int = 100) -> Dict:
    """
    Validate the universe for duplicates and expected size.
    
    Args:
        expected_size: Expected total unique tickers
    
    Returns:
        Dict with validation results
    
    Raises:
        ValueError: If duplicates found or size mismatch
    """
    all_list = [t for tickers in AI_UNIVERSE.values() for t in tickers]
    unique = set(all_list)
    
    # Check for duplicates
    if len(unique) != len(all_list):
        seen: Set[str] = set()
        dups: Set[str] = set()
        for t in all_list:
            if t in seen:
                dups.add(t)
            seen.add(t)
        raise ValueError(f"Duplicate tickers in universe: {sorted(dups)}")
    
    # Check size
    actual_size = len(unique)
    if actual_size != expected_size:
        raise ValueError(
            f"Expected {expected_size} unique tickers, got {actual_size}"
        )
    
    return {
        "valid": True,
        "total_tickers": actual_size,
        "categories": len(AI_UNIVERSE),
        "tickers_per_category": {
            cat: len(tickers) for cat, tickers in AI_UNIVERSE.items()
        },
    }


def print_universe_summary():
    """Print a summary of the AI universe."""
    all_tickers = get_all_tickers()
    print(f"AI Stock Universe: {len(all_tickers)} unique tickers")
    print(f"Categories: {len(AI_CATEGORIES)}")
    print()
    
    for cat in AI_CATEGORIES:
        tickers = AI_UNIVERSE[cat]
        desc = CATEGORY_DESCRIPTIONS[cat]
        print(f"  {cat} ({len(tickers)} stocks)")
        print(f"    {desc}")
        print(f"    {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
        print()


if __name__ == "__main__":
    # Validate and print summary when run directly
    try:
        result = validate_universe(expected_size=100)
        print("✅ Universe validation passed!")
        print(f"   Total tickers: {result['total_tickers']}")
        print(f"   Categories: {result['categories']}")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
    
    print()
    print_universe_summary()

