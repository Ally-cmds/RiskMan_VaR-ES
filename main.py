from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import sys
import os
import subprocess

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Optional GARCH; we catch ImportError gracefully
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# --------------------------- Config ---------------------------
TICKERS = ["NVDA", "PLD"]
START = "2023-10-15"       # ensures ≥ 2 years window ends on END
END   = "2025-10-15"       # Yahoo end is exclusive; this includes 2025-10-15
INIT_INVESTMENT = 10_000.0
CONF_LEVEL = 0.99
LAMBDA = 0.94              # EWMA lambda (RiskMetrics)
PLOT_BINS = 60
RNG_SEED = 42

np.random.seed(RNG_SEED)

# ------------------------ Paths / IO --------------------------
BASE_DIR = Path(__file__).parent.resolve()
OUT_DIR  = BASE_DIR / "outputs"
TAB_DIR  = OUT_DIR / "tables"
FIG_DIR  = OUT_DIR / "figs"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def out_table(name: str) -> Path:
    p = TAB_DIR / name
    print(f"[write] {p}")
    return p

def out_fig(name: str) -> Path:
    p = FIG_DIR / name
    print(f"[write] {p}")
    return p

def open_file(path: Path):
    """Best-effort open the file using OS default app."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)              # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"[warn] Could not open {path}: {e}")

# ------------------------ Data class --------------------------
@dataclass
class Portfolio:
    tickers: list
    weights: np.ndarray  # e.g., equal weights

    def returns(self, ret_df: pd.DataFrame) -> pd.Series:
        return ret_df[self.tickers].dot(self.weights)

# ------------------ Data download & returns -------------------
print("[info] Downloading prices from Yahoo Finance...")
df = yf.download(
    TICKERS,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=True,      # adjusted prices => use 'Close'
    progress=False,
    threads=False
)

if df.empty:
    raise RuntimeError("Downloaded price DataFrame is empty. Check tickers, internet, and Python interpreter packages.")

# Handle multi-ticker column structure and pick Close
if isinstance(df.columns, pd.MultiIndex):
    # Columns like ('Open'/'High'/.../'Close', 'NVDA'/'PLD')
    px = df['Close'].dropna(how="all")
else:
    # Single ticker fallback
    px = df[['Close']].rename(columns={'Close': TICKERS[0]}).dropna()

# Ensure columns exactly as TICKERS (uppercased)
px.columns = [str(c).upper() for c in px.columns]
px = px[[t.upper() for t in TICKERS]].dropna()

if px.empty:
    raise RuntimeError("Close price panel is empty after filtering.")

# Daily simple returns
rets = px.pct_change().dropna()
rets = rets[[t.upper() for t in TICKERS]]

if rets.empty:
    raise RuntimeError("No returns computed (not enough rows after pct_change).")

print(f"[info] Date range: {px.index.min().date()} → {px.index.max().date()} (N={len(px)})")

# --------------------- Summary statistics ---------------------
quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
summary_rows = []
for t in TICKERS:
    col = t.upper()
    s = rets[col]
    row = {
        "asset": col,
        "mean": s.mean(),
        "std": s.std(ddof=1),
        "skew": skew(s, bias=False),
        "kurtosis_excess": kurtosis(s, fisher=True, bias=False)
    }
    for q in quantiles:
        row[f"q_{int(q*100)}"] = s.quantile(q)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("asset")
cov_df = rets.cov()
cor_df = rets.corr()

summary_df.to_csv(out_table("summary_stats.csv"))
cov_df.to_csv(out_table("covariance.csv"))
cor_df.to_csv(out_table("correlation.csv"))

# ----------------- Equal-weighted portfolio -------------------
weights = np.array([1/len(TICKERS)] * len(TICKERS))
port = Portfolio([t.upper() for t in TICKERS], weights)
port_rets = port.returns(rets).dropna()
port_vals = (1 + port_rets).cumprod() * INIT_INVESTMENT

# Save a simple portfolio value chart
plt.figure(figsize=(8,4.5))
port_vals.plot()
plt.title("Equal-Weighted Portfolio Value ($10,000 initial)")
plt.ylabel("USD")
plt.xlabel("")
plt.tight_layout()
plt.savefig(out_fig("portfolio_value.png"), dpi=160)
plt.close()

# --------------------- EWMA volatility ------------------------
# σ_{t+1}^2 = λ σ_t^2 + (1-λ) r_t^2 on portfolio returns
def ewma_var(series: pd.Series, lam: float) -> pd.Series:
    v = np.zeros(len(series))
    v[0] = series.var(ddof=1)  # initialize with sample variance
    for i in range(1, len(series)):
        v[i] = lam * v[i-1] + (1 - lam) * series.iloc[i-1]**2
    return pd.Series(v, index=series.index)

port_var_ewma = ewma_var(port_rets, LAMBDA)
port_vol_ewma = np.sqrt(port_var_ewma)

# ---------------- Historical Simulation helpers ---------------
def basic_hs_losses(series: pd.Series, scale_to_dollars: float = INIT_INVESTMENT) -> pd.Series:
    # Convert returns to P/L (loss positive)
    return -series * scale_to_dollars

def weighted_hs_quantile(losses: pd.Series, alpha: float, lam: float) -> Tuple[float, float]:
    # RiskMetrics weights: ω_i ∝ λ^{t-i}
    n = len(losses)
    idx = np.arange(n)
    raw_w = lam ** (n - 1 - idx)
    w = raw_w / raw_w.sum()
    dfw = pd.DataFrame({"loss": losses.values, "w": w}, index=losses.index).sort_values("loss")
    csum = dfw["w"].cumsum()
    pos_mask = csum >= alpha
    var_idx = pos_mask.idxmax()
    var_val = float(dfw.loc[var_idx, "loss"])
    # ES using actual tail mass, not (1-alpha)
    tail = dfw.loc[pos_mask]
    tail_w = tail["w"].sum()
    es_val = float((tail["loss"] * tail["w"]).sum() / tail_w)
    return var_val, es_val

def vol_scaled_hs_losses(series: pd.Series, vol_target: float, vol_hist: pd.Series, scale_to_dollars: float = INIT_INVESTMENT) -> pd.Series:
    # Scale each historical return by (σ_{t+1} / σ_i)
    vol_hist_safe = vol_hist.replace(0.0, np.nan).fillna(method="bfill")
    ratio = vol_target / vol_hist_safe
    adj_returns = series * ratio
    return -adj_returns * scale_to_dollars

# --------------- 1-Day VaR/ES (three approaches) --------------
alpha = CONF_LEVEL

# (A) Basic HS on portfolio returns
losses_basic = basic_hs_losses(port_rets)
var_basic = float(np.quantile(losses_basic, alpha))
es_basic = float(losses_basic[losses_basic >= var_basic].mean())

# (B) EWMA-weighted HS
var_w, es_w = weighted_hs_quantile(losses_basic, alpha, LAMBDA)

# (C) Volatility-scaled HS (estimate σ_{t+1} as last EWMA vol)
vol_t1 = float(port_vol_ewma.iloc[-1])
losses_volscaled = vol_scaled_hs_losses(port_rets, vol_t1, port_vol_ewma)
var_vs = float(np.quantile(losses_volscaled, alpha))
es_vs = float(losses_volscaled[losses_volscaled >= var_vs].mean())

# --------------- 5-Day VaR/ES (two approaches) ----------------
# (1) √time scaling from 1-day (simple shortcut)
var_5d_sqrt = var_basic * np.sqrt(5.0)
es_5d_sqrt  = es_basic  * np.sqrt(5.0)

# (2) Historical 5-day overlapping returns
ret_5d = (1 + port_rets).rolling(5).apply(np.prod, raw=True) - 1
ret_5d = ret_5d.dropna()
losses_5d_hist = -ret_5d * INIT_INVESTMENT
var_5d_hist = float(np.quantile(losses_5d_hist, alpha))
es_5d_hist = float(losses_5d_hist[losses_5d_hist >= var_5d_hist].mean())

# ---------------------- Backtesting (opt) ---------------------
lookback = 250
if len(losses_basic) > lookback:
    recent_losses = losses_basic.iloc[-lookback:]
    exceptions_basic = int((recent_losses > var_basic).sum())
else:
    exceptions_basic = np.nan

# ---------------------- GARCH(1,1) block ----------------------
def compute_var_es_from_losses(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if len(tail) else float('nan')
    return var, es

def fit_garch_t(port_rets: pd.Series):
    # Fit on percent returns for numerical stability
    r_pct = port_rets * 100.0
    am = arch_model(r_pct, p=1, q=1, mean="Constant", vol="GARCH", dist="t")
    res = am.fit(disp="off")
    return res

def garch_forecast_vols(res, horizon: int = 5) -> np.ndarray:
    fc = res.forecast(horizon=horizon, reindex=False)
    var_h = fc.variance.values[-1]      # percent^2
    sig_h = np.sqrt(var_h) / 100.0      # decimal returns
    return sig_h

def garch_fhs_losses(res, horizon: int, n_sims: int, init_investment: float, rng: np.random.Generator) -> np.ndarray:
    sig_h = garch_forecast_vols(res, horizon=horizon)   # length=h
    std_resid = pd.Series(res.std_resid).dropna().values
    shocks = rng.choice(std_resid, size=(n_sims, horizon), replace=True)
    rets_h = shocks * sig_h[None, :]
    hday_ret = np.prod(1.0 + rets_h, axis=1) - 1.0
    losses = -hday_ret * init_investment
    return losses

def garch_parametric_t_1day(res, init_investment: float, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(2024)
    sig1 = garch_forecast_vols(res, horizon=1)[0]
    df = float(res.params.get("nu", 8.0))
    sim_std = rng.standard_t(df, size=200_000)
    sim_ret = sim_std * sig1
    sim_loss = -sim_ret * init_investment
    return compute_var_es_from_losses(sim_loss, alpha)

def garch_parametric_t_5day(res, init_investment: float, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(2025)
    sig5 = garch_forecast_vols(res, horizon=5)
    df = float(res.params.get("nu", 8.0))
    sims = 150_000
    std = rng.standard_t(df, size=(sims, 5))
    rets = std * sig5[None, :]
    hday = np.prod(1.0 + rets, axis=1) - 1.0
    losses = -hday * init_investment
    return compute_var_es_from_losses(losses, alpha)

garch_rows = []
if HAS_ARCH:
    try:
        rng = np.random.default_rng(RNG_SEED)
        garch_res = fit_garch_t(port_rets)
        omega = float(garch_res.params.get("omega", np.nan))
        alpha1 = float(garch_res.params.get("alpha[1]", np.nan))
        beta1  = float(garch_res.params.get("beta[1]", np.nan))
        nu_df  = float(garch_res.params.get("nu", np.nan))

        var_garch_1d, es_garch_1d = garch_parametric_t_1day(garch_res, INIT_INVESTMENT, CONF_LEVEL)
        var_garch_5d, es_garch_5d = garch_parametric_t_5day(garch_res, INIT_INVESTMENT, CONF_LEVEL)

        losses_garch_fhs_1d = garch_fhs_losses(garch_res, horizon=1, n_sims=100_000,
                                               init_investment=INIT_INVESTMENT, rng=rng)
        var_fhs_1d, es_fhs_1d = compute_var_es_from_losses(losses_garch_fhs_1d, CONF_LEVEL)

        losses_garch_fhs_5d = garch_fhs_losses(garch_res, horizon=5, n_sims=100_000,
                                               init_investment=INIT_INVESTMENT, rng=rng)
        var_fhs_5d, es_fhs_5d = compute_var_es_from_losses(losses_garch_fhs_5d, CONF_LEVEL)

        garch_rows.extend([
            {"approach": "1d – GARCH(1,1) Parametric-t",
             "lambda": np.nan, "var_99": var_garch_1d, "es_99": es_garch_1d,
             "notes": f"GARCH(1,1) t-resid; ω={omega:.3g}, α={alpha1:.3g}, β={beta1:.3g}, ν={nu_df:.2f}"},
            {"approach": "5d – GARCH(1,1) Parametric-t",
             "lambda": np.nan, "var_99": var_garch_5d, "es_99": es_garch_5d,
             "notes": "5× t shocks scaled by h-step σ"},
            {"approach": "1d – GARCH(1,1) FHS",
             "lambda": np.nan, "var_99": var_fhs_1d, "es_99": es_fhs_1d,
             "notes": "Bootstrap standardized residuals × σ_{t+1}"},
            {"approach": "5d – GARCH(1,1) FHS",
             "lambda": np.nan, "var_99": var_fhs_5d, "es_99": es_fhs_5d,
             "notes": "Bootstrap standardized residuals × σ_{t+1..t+5}, compounded"}
        ])

        # Plot FHS histograms
        def plot_hist(losses: pd.Series, var_val: float, es_val: float, title: str, fn: str, show=False, open_after_save=False):
            plt.figure(figsize=(8,5))
            plt.hist(losses, bins=PLOT_BINS, alpha=0.85)
            plt.axvline(var_val, linestyle='--', linewidth=2, label=f"VaR@99% = {var_val:,.0f}")
            plt.axvline(es_val, linestyle=':', linewidth=2, label=f"ES@99% = {es_val:,.0f}")
            plt.title(title)
            plt.xlabel("Loss ($)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            p = out_fig(fn)
            plt.savefig(p, dpi=160)
            if show:
                plt.show()
            plt.close()
            if open_after_save:
                open_file(p)

        plot_hist(pd.Series(losses_garch_fhs_1d), var_fhs_1d, es_fhs_1d, "1-day Losses – GARCH(1,1) FHS", "hist_1d_garch_fhs.png")
        plot_hist(pd.Series(losses_garch_fhs_5d), var_fhs_5d, es_fhs_5d, "5-day Losses – GARCH(1,1) FHS", "hist_5d_garch_fhs.png")

        print("\n=== GARCH(1,1) params (t-resid) ===")
        print(f"omega={omega:.6g}, alpha1={alpha1:.6g}, beta1={beta1:.6g}, nu={nu_df:.3f}")

    except Exception as e:
        print(f"[GARCH] Skipped due to error: {e}")
else:
    print("[GARCH] 'arch' package not installed; skipping GARCH block. (pip install arch)")

# ----------------------- Results table ------------------------
rows = [
    {"approach": "1d – Basic HS", "lambda": np.nan, "var_99": var_basic, "es_99": es_basic, "notes": "Unweighted historical simulation on portfolio returns"},
    {"approach": "1d – EWMA-weighted HS", "lambda": LAMBDA, "var_99": var_w, "es_99": es_w, "notes": "Weights ω_i ∝ λ^{t-i}"},
    {"approach": "1d – Vol-scaled HS (EWMA)", "lambda": LAMBDA, "var_99": var_vs, "es_99": es_vs, "notes": "Scale each return by σ_{t+1}/σ_i using EWMA"},
    {"approach": "5d – √time from 1d", "lambda": np.nan, "var_99": var_5d_sqrt, "es_99": es_5d_sqrt, "notes": "Approximate using √5 scaling of 1-day VaR/ES"},
    {"approach": "5d – Historical 5-day HS", "lambda": np.nan, "var_99": var_5d_hist, "es_99": es_5d_hist, "notes": "Overlapping 5-day portfolio returns"},
]

# Append GARCH rows if available
rows.extend(garch_rows)

results_df = pd.DataFrame(rows)
results_df.to_csv(out_table("var_es_table.csv"), index=False)

# ----------------------- Plot histograms ----------------------
def plot_hist(losses: pd.Series, var_val: float, es_val: float, title: str, fn: str, show=False, open_after_save=False):
    plt.figure(figsize=(8,5))
    plt.hist(losses, bins=PLOT_BINS, alpha=0.85)
    plt.axvline(var_val, linestyle='--', linewidth=2, label=f"VaR@99% = {var_val:,.0f}")
    plt.axvline(es_val, linestyle=':', linewidth=2, label=f"ES@99% = {es_val:,.0f}")
    plt.title(title)
    plt.xlabel("Loss ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    p = out_fig(fn)
    plt.savefig(p, dpi=160)
    if show:
        plt.show()
    plt.close()
    if open_after_save:
        open_file(p)

# Basic HS
plot_hist(losses_basic, var_basic, es_basic, "1-day Losses – Basic HS", "hist_1d_basic.png")
# EWMA-weighted HS (same sample, different quantile/ES)
plot_hist(losses_basic, var_w, es_w, "1-day Losses – EWMA-weighted HS", "hist_1d_weighted.png")
# Vol-scaled HS
plot_hist(losses_volscaled, var_vs, es_vs, "1-day Losses – Vol-scaled HS (EWMA)", "hist_1d_volscaled.png")
# 5d sqrt scaling
plot_hist(pd.Series((losses_basic - losses_basic.mean()) * np.sqrt(5)), var_5d_sqrt, es_5d_sqrt, "5-day Losses – √time from 1-day", "hist_5d_sqrt.png")
# 5d historical overlapping
plot_hist(losses_5d_hist, var_5d_hist, es_5d_hist, "5-day Losses – Historical 5-day HS", "hist_5d_hist.png")

print("\n=== Downloaded prices (head) ===")
print(px.head())
print("\n=== Daily returns (head) ===")
print(rets.head())
print("\n=== Summary stats ===")
print(summary_df)
print("\nCovariance matrix:\n", cov_df)
print("\nCorrelation matrix:\n", cor_df)
print("\n=== VaR/ES results (USD) ===")
print(results_df)
if not np.isnan(exceptions_basic):
    print(f"\nBacktest (last {lookback} days): Basic 1-day 99% VaR exceptions = {exceptions_basic}")

print("\nFiles written under:")
print(f"  Tables: {TAB_DIR}")
print(f"  Figures: {FIG_DIR}")
print("Attach these CSVs and PNGs in your report.")

garch_res = fit_garch_t(port_rets)
omega = float(garch_res.params.get("omega", np.nan))
alpha1 = float(garch_res.params.get("alpha[1]", np.nan))
beta1  = float(garch_res.params.get("beta[1]", np.nan))

def vfs_ewma_5day_losses(series: pd.Series, vol_hist: pd.Series, scale_to_dollars: float = INIT_INVESTMENT, 
                         horizon: int = 5, n_sims: int = 100_000, rng: np.random.Generator = None) -> np.ndarray:
    """
    Volatility Forecast Scaling (VFS) using EWMA volatility for multi-day losses.
    
    For each simulation path:
    1. Sample a random historical return
    2. Scale it by the ratio of forward-looking volatility to historical volatility
    3. Compound over the horizon
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    
    # Get the most recent volatility forecast (for t+1)
    vol_forecast = float(vol_hist.iloc[-1])
    
    # Clean historical volatility (replace zeros)
    vol_hist_safe = vol_hist.replace(0.0, np.nan).fillna(method="bfill")
    
    # Create simulation paths
    n_hist = len(series)
    sim_losses = np.zeros(n_sims)
    
    for i in range(n_sims):
        # Sample random historical returns for the horizon
        random_indices = rng.integers(0, n_hist, size=horizon)
        random_returns = series.iloc[random_indices].values
        random_vols = vol_hist_safe.iloc[random_indices].values
        
        # Scale returns by volatility ratio: σ_forecast / σ_historical
        # For multi-period, we scale each daily return in the path
        scaling_factors = vol_forecast / random_vols
        scaled_returns = random_returns * scaling_factors
        
        # Compound returns over the horizon
        compounded_return = np.prod(1 + scaled_returns) - 1
        
        # Calculate loss (positive = loss)
        sim_losses[i] = -compounded_return * scale_to_dollars
    
    return sim_losses

# Add this to your existing code after the other 5-day approaches:

# --------------- 5-Day VaR/ES using VFS-EWMA ---------------
print("[info] Computing 5-day VFS-EWMA losses...")
rng_vfs = np.random.default_rng(RNG_SEED + 1)
losses_5d_vfs_ewma = vfs_ewma_5day_losses(
    port_rets, 
    port_vol_ewma, 
    scale_to_dollars=INIT_INVESTMENT,
    horizon=5,
    n_sims=100_000,
    rng=rng_vfs
)

var_5d_vfs_ewma, es_5d_vfs_ewma = compute_var_es_from_losses(losses_5d_vfs_ewma, CONF_LEVEL)

# Add to results table
vfs_ewma_row = {
    "approach": "5d – VFS-EWMA", 
    "lambda": LAMBDA, 
    "var_99": var_5d_vfs_ewma, 
    "es_99": es_5d_vfs_ewma, 
    "notes": "Volatility Forecast Scaling with EWMA volatility; bootstrap historical returns scaled by σ_forecast/σ_historical"
}

# Insert this row after the other 5-day approaches in your results table
rows.insert(5, vfs_ewma_row)  # Insert at position 5 (after the existing 5-day methods)

# Also plot the histogram
plot_hist(
    pd.Series(losses_5d_vfs_ewma), 
    var_5d_vfs_ewma, 
    es_5d_vfs_ewma, 
    "5-day Losses – VFS with EWMA Volatility", 
    "hist_5d_vfs_ewma.png"
)