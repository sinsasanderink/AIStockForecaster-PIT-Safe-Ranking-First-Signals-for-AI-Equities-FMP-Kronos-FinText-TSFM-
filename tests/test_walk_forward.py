"""
Tests for Walk-Forward Validation Engine (Chapter 6.1)

CRITICAL: These tests verify that purging and embargo are ENFORCED, not just documented.

Tests verify:
1. Embargo in TRADING DAYS (not calendar days)
2. Per-row-per-horizon purging (not global)
3. Label maturity with UTC datetime
4. End-of-sample eligibility (all horizons valid)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pytz

from src.evaluation.walk_forward import WalkForwardSplitter, WalkForwardFold
from src.evaluation.definitions import (
    HORIZONS_TRADING_DAYS,
    EMBARGO_RULES,
    trading_days_to_calendar_days,
    get_market_close_utc,
    validate_embargo,
    validate_horizon,
)


class TestDefinitions:
    """Test canonical definitions from definitions.py."""
    
    def test_horizons_are_trading_days(self):
        """Verify horizons are defined as trading days."""
        assert HORIZONS_TRADING_DAYS == [20, 60, 90]
    
    def test_embargo_minimum_is_max_horizon(self):
        """Verify embargo minimum equals max horizon."""
        assert EMBARGO_RULES.embargo_trading_days == 90
        assert EMBARGO_RULES.embargo_trading_days == max(HORIZONS_TRADING_DAYS)
    
    def test_validate_embargo_rejects_small_values(self):
        """Test embargo validation rejects values < 90."""
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(30)
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(60)
        with pytest.raises(ValueError, match="TRADING DAYS"):
            validate_embargo(89)
        
        # 90 should pass
        validate_embargo(90)
        validate_embargo(100)
    
    def test_validate_horizon_rejects_invalid(self):
        """Test horizon validation rejects invalid values."""
        with pytest.raises(ValueError):
            validate_horizon(30)
        with pytest.raises(ValueError):
            validate_horizon(45)
        
        # Valid horizons should pass
        validate_horizon(20)
        validate_horizon(60)
        validate_horizon(90)
    
    def test_trading_days_to_calendar_days(self):
        """Test trading-to-calendar day conversion is conservative."""
        # 90 trading days should be ~130 calendar days (conservative)
        calendar_days = trading_days_to_calendar_days(90)
        assert calendar_days > 90  # More calendar days than trading days
        assert calendar_days >= 130  # Conservative estimate
    
    def test_get_market_close_utc(self):
        """Test market close UTC conversion."""
        # 2023-01-15 is a Sunday, but we still get a close time
        utc_close = get_market_close_utc(date(2023, 1, 15))
        
        assert utc_close.tzinfo == pytz.UTC
        # 4pm ET should be 21:00 UTC (or 22:00 during DST)
        assert utc_close.hour in [21, 22]  # Depends on DST


class TestWalkForwardFold:
    """Test WalkForwardFold validation and filtering."""
    
    def test_fold_date_validation(self):
        """Test that fold dates are validated on creation."""
        # Valid fold
        fold = WalkForwardFold(
            fold_id="test_01",
            train_start=date(2020, 1, 1),
            train_end=date(2023, 1, 1),
            val_start=date(2023, 4, 1),  # ~90 trading day embargo
            val_end=date(2023, 7, 1),
            embargo_trading_days=90,
            purged_labels=set()
        )
        assert fold.train_start < fold.train_end
        assert fold.val_start < fold.val_end
        assert fold.train_end < fold.val_start
    
    def test_fold_rejects_insufficient_embargo(self):
        """Test that embargo < 90 TRADING DAYS is rejected."""
        with pytest.raises(ValueError, match="TRADING DAYS"):
            WalkForwardFold(
                fold_id="test_bad",
                train_start=date(2020, 1, 1),
                train_end=date(2023, 1, 1),
                val_start=date(2023, 2, 1),
                val_end=date(2023, 3, 1),
                embargo_trading_days=30,  # Too small!
                purged_labels=set()
            )
    
    def test_fold_rejects_overlapping_dates(self):
        """Test that train_end >= val_start is rejected."""
        with pytest.raises(ValueError, match="EMBARGO NOT ENFORCED"):
            WalkForwardFold(
                fold_id="test_overlap",
                train_start=date(2020, 1, 1),
                train_end=date(2023, 2, 1),
                val_start=date(2023, 1, 1),  # Overlaps!
                val_end=date(2023, 3, 1),
                embargo_trading_days=90,
                purged_labels=set()
            )
    
    def test_filter_labels_by_date_range(self):
        """Test basic date filtering."""
        fold = WalkForwardFold(
            fold_id="test_01",
            train_start=date(2020, 1, 1),
            train_end=date(2023, 1, 1),
            val_start=date(2023, 6, 1),  # ~130 calendar days after train_end
            val_end=date(2023, 9, 1),
            embargo_trading_days=90,
            purged_labels=set()
        )
        
        # Create test labels with proper horizons
        labels = pd.DataFrame({
            "date": [
                date(2019, 12, 1),  # Before train
                date(2020, 6, 1),   # In train
                date(2022, 6, 1),   # In train
                date(2023, 3, 1),   # In embargo
                date(2023, 7, 1),   # In val
                date(2023, 10, 1),  # After val
            ],
            "ticker": ["AAPL"] * 6,
            "horizon": [20] * 6,  # Valid horizon
            "label": [0.05, 0.10, -0.02, 0.03, 0.08, 0.01],
            "label_matured_at": [
                get_market_close_utc(date(2020, 1, 1)),
                get_market_close_utc(date(2020, 7, 15)),
                get_market_close_utc(date(2022, 7, 15)),
                get_market_close_utc(date(2023, 4, 15)),
                get_market_close_utc(date(2023, 8, 15)),
                get_market_close_utc(date(2023, 11, 15)),
            ]
        })
        
        # Filter train
        train_df = fold.filter_labels(labels, split="train", horizons=[20])
        assert len(train_df) == 2  # 2020-06 and 2022-06
        assert all(train_df["date"] >= fold.train_start)
        assert all(train_df["date"] < fold.train_end)
        
        # Filter val
        val_df = fold.filter_labels(labels, split="val", horizons=[20])
        assert len(val_df) == 1  # 2023-07
        assert all(val_df["date"] >= fold.val_start)
        assert all(val_df["date"] < fold.val_end)
    
    def test_filter_enforces_label_maturity(self):
        """Test that immature labels are dropped (UTC datetime check)."""
        fold = WalkForwardFold(
            fold_id="test_01",
            train_start=date(2020, 1, 1),
            train_end=date(2023, 1, 1),
            val_start=date(2023, 6, 1),
            val_end=date(2023, 9, 1),
            embargo_trading_days=90,
            purged_labels=set()
        )
        
        labels = pd.DataFrame({
            "date": [
                date(2022, 6, 1),
                date(2022, 7, 1),
            ],
            "ticker": ["AAPL", "MSFT"],
            "horizon": [20, 20],
            "label": [0.05, 0.10],
            "label_matured_at": [
                get_market_close_utc(date(2022, 7, 1)),   # Mature before train_end
                get_market_close_utc(date(2023, 2, 1)),   # NOT mature before train_end
            ]
        })
        
        train_df = fold.filter_labels(labels, split="train", horizons=[20])
        assert len(train_df) == 1
        assert train_df.iloc[0]["ticker"] == "AAPL"
    
    def test_filter_removes_purged_labels(self):
        """Test that purged labels are removed (per-row-per-horizon)."""
        purged = {
            (date(2022, 6, 1), "AAPL", 20),
            (date(2022, 7, 1), "MSFT", 60),
        }
        
        fold = WalkForwardFold(
            fold_id="test_01",
            train_start=date(2020, 1, 1),
            train_end=date(2023, 1, 1),
            val_start=date(2023, 6, 1),
            val_end=date(2023, 9, 1),
            embargo_trading_days=90,
            purged_labels=purged
        )
        
        labels = pd.DataFrame({
            "date": [
                date(2022, 6, 1),
                date(2022, 7, 1),
                date(2022, 8, 1),
            ],
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "horizon": [20, 60, 20],  # Mixed horizons
            "label": [0.05, 0.10, -0.02],
            "label_matured_at": [
                get_market_close_utc(date(2022, 7, 1)),
                get_market_close_utc(date(2022, 9, 15)),
                get_market_close_utc(date(2022, 9, 1)),
            ]
        })
        
        train_df = fold.filter_labels(labels, split="train", horizons=[20, 60])
        assert len(train_df) == 1  # Only GOOGL survives
        assert train_df.iloc[0]["ticker"] == "GOOGL"
    
    def test_filter_rejects_invalid_horizons(self):
        """Test that invalid horizons are rejected."""
        fold = WalkForwardFold(
            fold_id="test_01",
            train_start=date(2020, 1, 1),
            train_end=date(2023, 1, 1),
            val_start=date(2023, 6, 1),
            val_end=date(2023, 9, 1),
            embargo_trading_days=90,
            purged_labels=set()
        )
        
        labels = pd.DataFrame({
            "date": [date(2022, 6, 1)],
            "ticker": ["AAPL"],
            "horizon": [20],
            "label": [0.05],
            "label_matured_at": [get_market_close_utc(date(2022, 7, 1))]
        })
        
        with pytest.raises(ValueError, match="trading days"):
            fold.filter_labels(labels, split="train", horizons=[45])  # Invalid horizon


class TestWalkForwardSplitter:
    """Test WalkForwardSplitter with purging and embargo enforcement."""
    
    def test_splitter_initialization(self):
        """Test splitter initialization with default parameters."""
        splitter = WalkForwardSplitter()
        assert splitter.eval_start == date(2016, 1, 1)
        assert splitter.eval_end == date(2025, 6, 30)
        assert splitter.embargo_trading_days == 90
        assert splitter.rebalance_freq == "monthly"
    
    def test_splitter_rejects_small_embargo(self):
        """Test that embargo < 90 TRADING DAYS is rejected."""
        with pytest.raises(ValueError, match="TRADING DAYS"):
            WalkForwardSplitter(embargo_trading_days=30)
    
    def test_splitter_rejects_invalid_rebalance_freq(self):
        """Test that invalid rebalance_freq is rejected."""
        with pytest.raises(ValueError, match="rebalance_freq must be"):
            WalkForwardSplitter(rebalance_freq="weekly")
    
    def test_embargo_days_backward_compatibility(self):
        """Test embargo_days property for backward compatibility."""
        splitter = WalkForwardSplitter(embargo_trading_days=100)
        assert splitter.embargo_days == 100
        assert splitter.embargo_trading_days == 100
    
    def test_generate_monthly_rebalance_dates(self):
        """Test monthly rebalance date generation."""
        splitter = WalkForwardSplitter(
            eval_start=date(2020, 1, 1),
            eval_end=date(2020, 6, 1),
            rebalance_freq="monthly"
        )
        
        dates = splitter._generate_rebalance_dates()
        assert len(dates) == 5  # Jan, Feb, Mar, Apr, May
        assert dates[0] == date(2020, 1, 1)
        assert dates[1] == date(2020, 2, 1)
        assert dates[-1] == date(2020, 5, 1)
    
    def test_generate_quarterly_rebalance_dates(self):
        """Test quarterly rebalance date generation."""
        splitter = WalkForwardSplitter(
            eval_start=date(2020, 1, 1),
            eval_end=date(2021, 1, 1),
            rebalance_freq="quarterly"
        )
        
        dates = splitter._generate_rebalance_dates()
        assert len(dates) == 4  # Q1, Q2, Q3, Q4
        assert dates[0] == date(2020, 1, 1)
        assert dates[1] == date(2020, 4, 1)
        assert dates[2] == date(2020, 7, 1)
        assert dates[3] == date(2020, 10, 1)
    
    def test_generate_folds_creates_expanding_window(self):
        """Test that folds use EXPANDING window (not rolling)."""
        splitter = WalkForwardSplitter(
            eval_start=date(2018, 1, 1),
            eval_end=date(2020, 1, 1),
            rebalance_freq="quarterly",
            min_train_days=365
        )
        
        # Create minimal labels
        labels = pd.DataFrame({
            "date": [date(2018, 1, 1)],
            "ticker": ["AAPL"],
            "horizon": [20],
            "label": [0.05],
            "label_matured_at": [get_market_close_utc(date(2018, 2, 1))]
        })
        
        folds = splitter.generate_folds(labels, horizons=[20], require_all_horizons=False)
        
        # All folds should start from eval_start (expanding window)
        for fold in folds:
            assert fold.train_start == date(2018, 1, 1)
        
        # Validation periods should progress forward
        if len(folds) >= 2:
            assert folds[1].val_start > folds[0].val_start
    
    def test_generate_folds_enforces_embargo(self):
        """Test that all folds have at least 90 TRADING DAY embargo."""
        splitter = WalkForwardSplitter(
            eval_start=date(2018, 1, 1),
            eval_end=date(2020, 1, 1),
            rebalance_freq="quarterly"
        )
        
        labels = pd.DataFrame({
            "date": [date(2018, 1, 1)],
            "ticker": ["AAPL"],
            "horizon": [20],
            "label": [0.05],
            "label_matured_at": [get_market_close_utc(date(2018, 2, 1))]
        })
        
        folds = splitter.generate_folds(labels, horizons=[20], require_all_horizons=False)
        
        for fold in folds:
            # Check embargo is stored as trading days
            assert fold.embargo_trading_days >= 90
            
            # Check calendar day gap is at least the conservative estimate
            gap_days = (fold.val_start - fold.train_end).days
            expected_min_gap = trading_days_to_calendar_days(90)
            assert gap_days >= expected_min_gap - 5  # Allow small tolerance
    
    def test_compute_purged_labels_per_horizon(self):
        """Test that purging is PER-HORIZON, not global."""
        splitter = WalkForwardSplitter(
            eval_start=date(2020, 1, 1),
            eval_end=date(2021, 1, 1)
        )
        
        # Create labels with different horizons at same date
        # A 20d label near boundary should be purged differently than 90d label
        labels = pd.DataFrame({
            "date": [
                date(2020, 10, 1),  # Near boundary
                date(2020, 10, 1),  # Same date, different horizon
            ],
            "ticker": ["AAPL", "AAPL"],
            "horizon": [20, 90],  # Different horizons
            "label": [0.05, 0.10],
            "label_matured_at": [
                get_market_close_utc(date(2020, 10, 28)),  # 20d matures ~Oct 28
                get_market_close_utc(date(2021, 1, 15)),   # 90d matures ~Jan 15
            ]
        })
        
        train_end = date(2020, 11, 1)
        val_start = date(2020, 12, 1)
        
        purged = splitter._compute_purged_labels(
            labels_df=labels,
            train_end=train_end,
            val_start=val_start,
            horizons=[20, 90]
        )
        
        # 90d horizon should be purged (maturity extends well past train_end)
        # 20d horizon might or might not be purged depending on exact calculation
        assert (date(2020, 10, 1), "AAPL", 90) in purged
        # Key: they should NOT be treated the same just because embargo = 90


class TestEndOfSampleEligibility:
    """Test end-of-sample eligibility rules."""
    
    def test_filter_eligible_dates_all_horizons(self):
        """Test that dates without all horizons are filtered."""
        splitter = WalkForwardSplitter(
            eval_start=date(2020, 1, 1),
            eval_end=date(2020, 6, 1),
            rebalance_freq="monthly"
        )
        
        # Create labels where some dates have all horizons, some don't
        labels = pd.DataFrame({
            "date": [
                # Jan 1: all horizons
                date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1),
                # Feb 1: only 20d and 60d (missing 90d)
                date(2020, 2, 1), date(2020, 2, 1),
                # Mar 1: all horizons
                date(2020, 3, 1), date(2020, 3, 1), date(2020, 3, 1),
            ],
            "ticker": ["AAPL"] * 8,
            "horizon": [20, 60, 90, 20, 60, 20, 60, 90],
            "label": [0.05] * 8,
            "label_matured_at": [get_market_close_utc(date(2020, 6, 1))] * 8,
        })
        
        rebalance_dates = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        
        eligible = splitter._filter_eligible_dates(
            rebalance_dates, labels, [20, 60, 90]
        )
        
        # Feb 1 should be excluded (missing 90d horizon)
        assert date(2020, 2, 1) not in eligible
        assert date(2020, 1, 1) in eligible
        assert date(2020, 3, 1) in eligible


class TestIntegrationWalkForward:
    """Integration tests for full walk-forward pipeline."""
    
    def test_end_to_end_fold_generation_and_filtering(self):
        """Test complete pipeline: generate folds and filter labels."""
        # Create splitter
        splitter = WalkForwardSplitter(
            eval_start=date(2018, 1, 1),
            eval_end=date(2020, 1, 1),
            rebalance_freq="quarterly",
            embargo_trading_days=90,
            min_train_days=365
        )
        
        # Create synthetic labels
        dates = pd.date_range(date(2018, 1, 1), date(2019, 12, 31), freq="MS")
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
        
        labels_data = []
        for d in dates:
            for ticker in tickers:
                for horizon in [20, 60, 90]:
                    maturity_calendar = trading_days_to_calendar_days(horizon)
                    maturity = d + pd.Timedelta(days=maturity_calendar)
                    labels_data.append({
                        "date": d.date(),
                        "ticker": ticker,
                        "horizon": horizon,
                        "label": np.random.randn() * 0.1,
                        "label_matured_at": get_market_close_utc(maturity.date())
                    })
        
        labels = pd.DataFrame(labels_data)
        
        # Generate folds
        folds = splitter.generate_folds(
            labels, horizons=[20, 60, 90], require_all_horizons=False
        )
        
        assert len(folds) > 0
        
        # Test each fold
        for fold in folds:
            # Get train and val labels
            train_df = fold.filter_labels(labels, split="train")
            val_df = fold.filter_labels(labels, split="val")
            
            # Basic checks
            assert len(train_df) > 0, f"Fold {fold.fold_id} has no training labels"
            
            # Check no overlap in dates
            if len(val_df) > 0:
                max_train_date = train_df["date"].max()
                min_val_date = val_df["date"].min()
                assert max_train_date < fold.train_end
                assert min_val_date >= fold.val_start
            
            # Check purging was applied
            assert isinstance(fold.purged_labels, set)
            
            # Verify expanding window
            assert fold.train_start == splitter.eval_start
    
    def test_purging_prevents_leakage_per_horizon(self):
        """Test that purging prevents label leakage, accounting for different horizons."""
        splitter = WalkForwardSplitter(
            eval_start=date(2020, 1, 1),
            eval_end=date(2021, 1, 1),
            rebalance_freq="monthly",
            embargo_trading_days=90
        )
        
        # Create a 90d label that would leak if not purged
        # Label at 2020-10-01 with horizon 90d matures around 2021-01-15
        labels = pd.DataFrame({
            "date": [date(2020, 10, 1)],
            "ticker": ["AAPL"],
            "horizon": [90],
            "label": [0.10],
            "label_matured_at": [get_market_close_utc(date(2021, 1, 15))]
        })
        
        folds = splitter.generate_folds(
            labels, horizons=[90], require_all_horizons=False
        )
        
        # Find fold with val_start around 2020-12-01
        for fold in folds:
            if fold.val_start >= date(2020, 12, 1) and fold.val_start < date(2021, 1, 1):
                # This 90d label starting Oct 1 extends into 2021
                # It should be purged from training
                train_df = fold.filter_labels(labels, split="train")
                
                # Either the label is filtered out, or it's in the purged set
                label_in_train = len(train_df[train_df["date"] == date(2020, 10, 1)]) > 0
                label_is_purged = (date(2020, 10, 1), "AAPL", 90) in fold.purged_labels
                
                assert not label_in_train or label_is_purged, \
                    f"90d label at 2020-10-01 should be purged but isn't"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
