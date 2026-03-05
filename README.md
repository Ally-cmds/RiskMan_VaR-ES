# RiskMan_VaR-ES
A comprehensive Value at Risk (VaR) and Expected Shortfall (ES) analysis for a two-stock portfolio (NVDA and PLD) using multiple estimation methodologies 
including Historical Simulation, EWMA, GARCH(1,1), and Volatility Forecast Scaling (VFS).

## Technologies Used
-Python – Core programming language
-NumPy / Pandas – Data manipulation and numerical computing
-yfinance – Financial data download
-SciPy – Statistical analysis (skewness, kurtosis)
-Matplotlib – Data visualization
-arch – GARCH model estimation (optional)

## Features
Horizon	Method	Description
1-day	Basic HS	Unweighted historical simulation
1-day	EWMA-weighted HS	RiskMetrics weighting (λ=0.94)
1-day	Vol-scaled HS	Scale returns by volatility ratio
1-day	GARCH(1,1) Parametric-t	t-distributed residuals
1-day	GARCH(1,1) FHS	Filtered Historical Simulation
5-day	√time scaling	Approximate from 1-day metrics
5-day	Historical 5-day HS	Overlapping 5-day returns
5-day	VFS-EWMA	Volatility Forecast Scaling
5-day	GARCH(1,1) Parametric-t	Multi-period t-shocks
5-day	GARCH(1,1) FHS	Bootstrap standardized residuals

## How to Run
1. Clone this repository
   git clone https://github.com/Ally-cmds/RiskMan_VaR-ES.git
   cd RiskMan_VaR-ES
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python main.py`
