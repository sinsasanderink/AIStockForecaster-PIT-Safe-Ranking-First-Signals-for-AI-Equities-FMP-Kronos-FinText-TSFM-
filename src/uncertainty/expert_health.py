"""
Expert Health Signal H(t) — Chapter 13.4b
============================================

Date-level "expert health" throttle that complements per-stock ê(x).

    ê(x) = cross-sectional position-level risk control (which names are
            dangerous today)
    H(t) = expert-level regime control (whether the expert is usable today)

Three complementary PIT-safe signals:
    H_realized(t)  — trailing EWMA of matured daily RankIC (lagged by horizon)
    H_drift(t)     — feature + score distribution drift vs reference window
    H_disagree(t)  — cross-expert ranking disagreement

Combined into a single scalar H(t) ∈ [0, 1] via normalized weighted sum +
sigmoid squash. Higher H = healthier expert. G(t) = exposure multiplier.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HealthConfig:
    """Configuration for ExpertHealthEstimator."""
    horizon: int = 20
    ewma_halflife: int = 30
    ewma_min_periods: int = 20
    reference_window: int = 252
    drift_features: List[str] = field(default_factory=lambda: [
        "vol_20d", "mom_1m", "adv_20d", "vix_percentile_252d",
        "market_vol_21d", "vol_60d",
    ])
    correlation_lookback: int = 20
    alpha_drift: float = 0.3
    beta_disagree: float = 0.3
    gate_threshold_low: float = 0.3
    gate_threshold_high: float = 0.7


class ExpertHealthEstimator:
    """
    Computes per-date expert health H(t) for a single model/expert.

    Usage:
        est = ExpertHealthEstimator(config)
        health_df = est.compute(enriched_residuals, other_model_residuals)
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()

    def compute(
        self,
        enriched: pd.DataFrame,
        other_models: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute all health signals and combined H(t).

        Args:
            enriched: Enriched residuals for the primary expert.
            other_models: Optional dict of {model_name: enriched_residuals_df}
                          for disagreement computation.

        Returns:
            health_df: Per-date health DataFrame with columns:
                [date, H_realized, H_drift, H_disagree, H_combined, G_exposure,
                 daily_rankic, matured_rankic, ...]
            diagnostics: Summary statistics and validation metrics.
        """
        cfg = self.config
        hz = enriched[enriched["horizon"] == cfg.horizon].copy()
        hz["as_of_date"] = pd.to_datetime(hz["as_of_date"])

        if len(hz) == 0:
            raise ValueError(f"No data for horizon {cfg.horizon}")

        logger.info(f"Computing expert health for {cfg.horizon}d ({len(hz):,} rows)")

        # Step 1: Daily RankIC
        daily_ic = self._compute_daily_rankic(hz)

        # Step 2: H_realized (matured + EWMA)
        daily_ic = self._compute_h_realized(daily_ic)

        # Step 3: H_drift
        daily_ic = self._compute_h_drift(hz, daily_ic)

        # Step 4: H_disagree
        daily_ic = self._compute_h_disagree(hz, other_models, daily_ic)

        # Step 5: Combine into H(t) and G(t)
        daily_ic = self._combine_health(daily_ic)

        # Step 6: Diagnostics
        diagnostics = self._compute_diagnostics(daily_ic)

        return daily_ic, diagnostics

    # ── Signal 1: Realized rolling efficacy ───────────────────────────────

    def _compute_daily_rankic(self, hz: pd.DataFrame) -> pd.DataFrame:
        """Compute daily RankIC = Spearman(score, excess_return) per date."""
        records = []
        for date, group in hz.groupby("as_of_date"):
            if len(group) < 5:
                continue
            r = stats.spearmanr(group["score"], group["excess_return"])
            records.append({
                "date": date,
                "daily_rankic": r.statistic if not np.isnan(r.statistic) else 0.0,
                "n_stocks": len(group),
                "mean_rank_loss": group["rank_loss"].mean(),
            })
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        logger.info(f"  Daily RankIC: {len(df)} dates, mean={df['daily_rankic'].mean():.4f}")
        return df

    def _compute_h_realized(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Matured RankIC + trailing EWMA.

        At date t, the matured RankIC is from t - horizon trading days.
        This is PIT-safe: we only use labels that would have been observed by t.
        """
        cfg = self.config
        daily = daily.copy()

        # Shift daily_rankic by horizon trading days (rows, since sorted by date)
        daily["matured_rankic"] = daily["daily_rankic"].shift(cfg.horizon)

        # EWMA of matured RankIC
        daily["H_realized"] = daily["matured_rankic"].ewm(
            halflife=cfg.ewma_halflife,
            min_periods=cfg.ewma_min_periods,
        ).mean()

        n_valid = daily["H_realized"].notna().sum()
        logger.info(f"  H_realized: {n_valid} valid dates (EWMA halflife={cfg.ewma_halflife})")
        return daily

    # ── Signal 2: Drift / anomaly ─────────────────────────────────────────

    def _compute_h_drift(
        self,
        hz: pd.DataFrame,
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Feature + score distribution drift vs trailing reference window.

        Components:
        - Feature drift: mean absolute PSI across key features
        - Score drift: KS statistic of score distribution vs reference
        - Correlation spike: average pairwise correlation of trailing returns
        """
        cfg = self.config
        all_dates = sorted(daily["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        feat_cols = [c for c in cfg.drift_features if c in hz.columns]
        hz_sorted = hz.sort_values("as_of_date")

        # Pre-compute per-date feature stats and score distributions
        date_feat_stats = {}
        date_scores = {}
        for date, group in hz_sorted.groupby("as_of_date"):
            if date not in date_to_idx:
                continue
            # Deduplicate by ticker (mean across sub-models)
            deduped = group.groupby("ticker")[feat_cols + ["score"]].mean() if feat_cols else group.groupby("ticker")[["score"]].mean()
            feat_means = deduped[feat_cols].mean().values if feat_cols else np.array([])
            date_feat_stats[date] = feat_means
            date_scores[date] = deduped["score"].dropna().values

        # Pre-compute per-date return vectors for correlation spike
        # Deduplicate by ticker (take mean across sub-models if multiple)
        date_returns = {}
        if "excess_return" in hz.columns:
            for date, group in hz_sorted.groupby("as_of_date"):
                if date in date_to_idx:
                    date_returns[date] = group.groupby("ticker")["excess_return"].mean()

        drift_scores = []
        score_drift_vals = []
        corr_spike_vals = []

        for i, date in enumerate(all_dates):
            # Reference window: trailing reference_window dates
            ref_start = max(0, i - cfg.reference_window)
            ref_end = i
            ref_dates = all_dates[ref_start:ref_end]

            if len(ref_dates) < 60:
                drift_scores.append(np.nan)
                score_drift_vals.append(np.nan)
                corr_spike_vals.append(np.nan)
                continue

            # Feature drift: compare today's feature means vs reference mean
            if feat_cols and date in date_feat_stats:
                ref_means = np.array([date_feat_stats[d] for d in ref_dates if d in date_feat_stats])
                if len(ref_means) > 10:
                    ref_mean = ref_means.mean(axis=0)
                    ref_std = ref_means.std(axis=0) + 1e-8
                    today_mean = date_feat_stats[date]
                    z_drift = np.abs((today_mean - ref_mean) / ref_std).mean()
                else:
                    z_drift = 0.0
            else:
                z_drift = 0.0
            drift_scores.append(z_drift)

            # Score drift: KS statistic vs reference
            if date in date_scores:
                ref_score_samples = []
                for rd in ref_dates[-60:]:
                    if rd in date_scores:
                        ref_score_samples.extend(date_scores[rd][:20])
                if len(ref_score_samples) > 20 and len(date_scores[date]) > 5:
                    ks_stat = stats.ks_2samp(date_scores[date], ref_score_samples).statistic
                else:
                    ks_stat = 0.0
            else:
                ks_stat = 0.0
            score_drift_vals.append(ks_stat)

            # Correlation spike: avg pairwise corr of trailing returns
            lookback_dates = all_dates[max(0, i - cfg.correlation_lookback):i + 1]
            ret_frames = [date_returns[d] for d in lookback_dates if d in date_returns]
            if len(ret_frames) >= 10:
                ret_matrix = pd.concat(ret_frames, axis=1).dropna(axis=0, how="all")
                if ret_matrix.shape[0] >= 5 and ret_matrix.shape[1] >= 5:
                    corr_mat = ret_matrix.T.corr()
                    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
                    avg_corr = corr_mat.values[mask].mean()
                else:
                    avg_corr = 0.0
            else:
                avg_corr = 0.0
            corr_spike_vals.append(avg_corr)

        daily = daily.copy()
        daily["feat_drift"] = drift_scores
        daily["score_drift"] = score_drift_vals
        daily["corr_spike"] = corr_spike_vals

        # Combine drift components: higher drift = lower health (inverted later)
        daily["H_drift_raw"] = (
            daily["feat_drift"].fillna(0) * 0.4
            + daily["score_drift"].fillna(0) * 0.3
            + daily["corr_spike"].fillna(0) * 0.3
        )

        n_valid = daily["H_drift_raw"].notna().sum()
        logger.info(f"  H_drift: {n_valid} valid dates, mean={daily['H_drift_raw'].mean():.4f}")
        return daily

    # ── Signal 3: Cross-expert disagreement ───────────────────────────────

    def _compute_h_disagree(
        self,
        hz: pd.DataFrame,
        other_models: Optional[Dict[str, pd.DataFrame]],
        daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cross-expert ranking disagreement.

        If other models available: Spearman rank correlation between experts'
        scores per date. Low correlation = high disagreement = lower health.

        If no other models: use score dispersion as a proxy (high dispersion
        in model scores may indicate uncertainty).
        """
        cfg = self.config
        daily = daily.copy()

        if other_models and len(other_models) > 0:
            primary_scores = hz.groupby(["as_of_date", "ticker"])["score"].mean()

            disagree_records = {}
            for model_name, model_df in other_models.items():
                model_hz = model_df[model_df["horizon"] == cfg.horizon].copy()
                model_hz["as_of_date"] = pd.to_datetime(model_hz["as_of_date"])

                if "score" not in model_hz.columns:
                    continue

                other_scores = model_hz.groupby(["as_of_date", "ticker"])["score"].mean()
                for date in model_hz["as_of_date"].unique():
                    if date not in primary_scores.index.get_level_values(0):
                        continue
                    try:
                        primary = primary_scores.loc[date]
                        other = other_scores.loc[date]
                        common = primary.index.intersection(other.index)
                        if len(common) >= 5:
                            rho = stats.spearmanr(
                                primary.loc[common], other.loc[common]
                            ).statistic
                            if date not in disagree_records:
                                disagree_records[date] = []
                            disagree_records[date].append(rho)
                    except (KeyError, ValueError):
                        continue

            if disagree_records:
                disagree_series = pd.Series({
                    d: 1.0 - np.mean(rhos)
                    for d, rhos in disagree_records.items()
                })
                daily["H_disagree_raw"] = daily["date"].map(disagree_series)
                logger.info(f"  H_disagree: {disagree_series.notna().sum()} dates from {len(other_models)} other model(s)")
            else:
                daily["H_disagree_raw"] = 0.0
                logger.info("  H_disagree: no valid cross-expert data, defaulting to 0")
        else:
            # Fallback: score dispersion as disagreement proxy
            score_std = hz.groupby("as_of_date")["score"].std()
            ref_std = score_std.expanding(min_periods=60).mean()
            relative_dispersion = (score_std - ref_std) / (ref_std + 1e-8)
            daily["H_disagree_raw"] = daily["date"].map(relative_dispersion).fillna(0)
            logger.info(f"  H_disagree: using score dispersion proxy (no other models)")

        return daily

    # ── Combination ───────────────────────────────────────────────────────

    def _combine_health(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Combine signals into H(t) ∈ [0, 1] and compute G(t) exposure multiplier.

        H(t) = sigmoid(z(H_realized) - α·z(H_drift) - β·z(H_disagree))
        Higher H = healthier expert.

        Uses expanding z-scores for stability. H_realized is the dominant
        signal; drift/disagreement provide supplementary adjustment.
        """
        cfg = self.config
        daily = daily.copy()

        def _trailing_zscore(s: pd.Series) -> pd.Series:
            mu = s.expanding(min_periods=60).mean()
            sigma = s.expanding(min_periods=60).std()
            return ((s - mu) / (sigma + 1e-8)).clip(-3, 3)

        z_real = _trailing_zscore(daily["H_realized"].fillna(0))
        z_drift = _trailing_zscore(daily["H_drift_raw"].fillna(0))
        z_disagree = _trailing_zscore(daily["H_disagree_raw"].fillna(0))

        # Higher realized = good; higher drift/disagree = bad
        raw_h = z_real - cfg.alpha_drift * z_drift - cfg.beta_disagree * z_disagree

        # Sigmoid squash to [0, 1]
        daily["H_combined"] = 1.0 / (1.0 + np.exp(-raw_h))

        # Also produce a "realized-only" health for comparison
        daily["H_realized_only"] = 1.0 / (1.0 + np.exp(-z_real))

        # Components for transparency
        daily["z_realized"] = z_real
        daily["z_drift"] = z_drift
        daily["z_disagree"] = z_disagree

        # Exposure multiplier G(t)
        daily["G_exposure"] = np.clip(
            (daily["H_combined"] - cfg.gate_threshold_low)
            / (cfg.gate_threshold_high - cfg.gate_threshold_low),
            0.0, 1.0,
        )

        n_valid = daily["H_combined"].notna().sum()
        logger.info(
            f"  H_combined: {n_valid} valid dates, "
            f"mean={daily['H_combined'].mean():.4f}, "
            f"G mean={daily['G_exposure'].mean():.4f}"
        )
        return daily

    # ── Diagnostics ───────────────────────────────────────────────────────

    def _compute_diagnostics(self, daily: pd.DataFrame) -> Dict[str, Any]:
        """Validate H(t) against realized outcomes."""
        cfg = self.config
        valid = daily.dropna(subset=["H_combined", "daily_rankic"])

        if len(valid) < 20:
            return {"skip": True, "n": len(valid)}

        diag: Dict[str, Any] = {"n_dates": len(valid), "horizon": cfg.horizon}

        # Correlation of H(t) with realized daily RankIC
        rho_h_ic = float(stats.spearmanr(valid["H_combined"], valid["daily_rankic"]).statistic)
        diag["rho_H_rankic"] = round(rho_h_ic, 4)

        # AUROC: can H(t) predict bad days (RankIC < 0)?
        valid = valid.copy()
        valid["bad_day"] = (valid["daily_rankic"] < 0).astype(int)
        n_bad = valid["bad_day"].sum()
        n_good = len(valid) - n_bad

        if n_bad >= 5 and n_good >= 5:
            from sklearn.metrics import roc_auc_score
            # Low H should predict bad days → use 1-H as predictor
            auroc = float(roc_auc_score(valid["bad_day"], 1 - valid["H_combined"]))
            diag["auroc_bad_day"] = round(auroc, 4)
        else:
            diag["auroc_bad_day"] = None

        # Quintile analysis: H quintiles vs mean RankIC
        valid = valid.copy()
        try:
            valid["h_quintile"] = pd.qcut(valid["H_combined"], 5, labels=False, duplicates="drop")
            qtable = valid.groupby("h_quintile").agg(
                n=("daily_rankic", "size"),
                mean_rankic=("daily_rankic", "mean"),
                mean_rank_loss=("mean_rank_loss", "mean"),
                pct_bad=("bad_day", "mean"),
            ).to_dict("index")
            diag["quintile_analysis"] = {str(k): {kk: round(vv, 4) for kk, vv in v.items()} for k, v in qtable.items()}
        except ValueError:
            diag["quintile_analysis"] = {}

        # DEV vs FINAL
        from src.uncertainty.epistemic_signal import HOLDOUT_START
        for period_name, mask in [
            ("DEV", valid["date"] < HOLDOUT_START),
            ("FINAL", valid["date"] >= HOLDOUT_START),
        ]:
            pdata = valid[mask]
            if len(pdata) < 10:
                diag[period_name] = {"n": len(pdata)}
                continue
            rho = float(stats.spearmanr(pdata["H_combined"], pdata["daily_rankic"]).statistic)
            pdata_c = pdata.copy()
            pdata_c["bad"] = (pdata_c["daily_rankic"] < 0).astype(int)
            n_b, n_g = pdata_c["bad"].sum(), len(pdata_c) - pdata_c["bad"].sum()
            if n_b >= 3 and n_g >= 3:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(pdata_c["bad"], 1 - pdata_c["H_combined"]))
            else:
                auc = None
            diag[period_name] = {
                "n": len(pdata),
                "rho_H_rankic": round(rho, 4),
                "auroc_bad_day": round(auc, 4) if auc is not None else None,
                "mean_H": round(float(pdata["H_combined"].mean()), 4),
                "mean_G": round(float(pdata["G_exposure"].mean()), 4),
            }

        # Mar-Jul 2024 crisis analysis
        crisis_mask = (valid["date"] >= "2024-03-01") & (valid["date"] <= "2024-07-31")
        final_mask = valid["date"] >= HOLDOUT_START
        non_crisis_final = valid[final_mask & ~crisis_mask]
        crisis = valid[crisis_mask]

        if len(crisis) > 0 and len(non_crisis_final) > 0:
            diag["crisis_2024"] = {
                "crisis_mean_H": round(float(crisis["H_combined"].mean()), 4),
                "crisis_mean_G": round(float(crisis["G_exposure"].mean()), 4),
                "crisis_mean_rankic": round(float(crisis["daily_rankic"].mean()), 4),
                "non_crisis_mean_H": round(float(non_crisis_final["H_combined"].mean()), 4),
                "non_crisis_mean_G": round(float(non_crisis_final["G_exposure"].mean()), 4),
                "non_crisis_mean_rankic": round(float(non_crisis_final["daily_rankic"].mean()), 4),
                "H_drops_in_crisis": float(crisis["H_combined"].mean()) < float(non_crisis_final["H_combined"].mean()),
                "G_reduces_in_crisis": float(crisis["G_exposure"].mean()) < float(non_crisis_final["G_exposure"].mean()),
                "n_crisis": len(crisis),
                "n_non_crisis": len(non_crisis_final),
            }
        else:
            diag["crisis_2024"] = {"skip": True}

        # Worst-10% H days overlap with negative RankIC days
        h_10th = valid["H_combined"].quantile(0.10)
        worst_h_days = valid[valid["H_combined"] <= h_10th]
        if len(worst_h_days) > 0:
            overlap_pct = float(worst_h_days["bad_day"].mean())
            diag["worst_10pct_H_bad_day_overlap"] = round(overlap_pct, 4)
        else:
            diag["worst_10pct_H_bad_day_overlap"] = None

        # Component correlations
        for comp in ["H_realized", "H_drift_raw", "H_disagree_raw", "H_realized_only"]:
            if comp in valid.columns and valid[comp].notna().sum() > 20:
                r = stats.spearmanr(valid[comp].fillna(0), valid["daily_rankic"])
                diag[f"rho_{comp}_rankic"] = round(float(r.statistic), 4)

        # Realized-only AUROC for comparison
        if "H_realized_only" in valid.columns and n_bad >= 5 and n_good >= 5:
            from sklearn.metrics import roc_auc_score
            auroc_ro = float(roc_auc_score(valid["bad_day"], 1 - valid["H_realized_only"]))
            diag["auroc_bad_day_realized_only"] = round(auroc_ro, 4)

        # Monthly summary for Mar-Jul 2024 time series
        valid_ts = valid.copy()
        valid_ts["ym"] = valid_ts["date"].dt.to_period("M")
        monthly = valid_ts.groupby("ym").agg(
            mean_H=("H_combined", "mean"),
            mean_G=("G_exposure", "mean"),
            mean_IC=("daily_rankic", "mean"),
            pct_bad=("bad_day", "mean"),
        )
        crisis_months = monthly[
            (monthly.index >= "2024-03") & (monthly.index <= "2024-07")
        ]
        if len(crisis_months) > 0:
            diag["crisis_monthly"] = {
                str(k): {kk: round(float(vv), 4) for kk, vv in v.items()}
                for k, v in crisis_months.to_dict("index").items()
            }

        return diag
