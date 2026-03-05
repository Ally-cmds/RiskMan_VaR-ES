'''
import yfinance as yf
import pandas as pd
import numpy as np

# Define the time period
start_date = '2025-10-01'
end_date = '2025-10-15'

print(f"Fetching data from {start_date} to {end_date}")

# Fetch historical data for NVDA and PLD
nvda_data = yf.download('NVDA', start=start_date, end=end_date)
pld_data = yf.download('PLD', start=start_date, end=end_date)

print(f"NVDA data columns: {nvda_data.columns}")
print(f"PLD data columns: {pld_data.columns}")

# Extract the Close prices (simplest approach)
# For MultiIndex columns, we need to handle them differently
if isinstance(nvda_data.columns, pd.MultiIndex):
    # MultiIndex structure - extract Close prices
    nvda = nvda_data['Close']
    pld = pld_data['Close']
else:
    # Single index structure
    nvda = nvda_data['Close']
    pld = pld_data['Close']

# If we got Series with MultiIndex, extract the values
if isinstance(nvda, pd.DataFrame):
    nvda = nvda.iloc[:, 0]
if isinstance(pld, pd.DataFrame):
    pld = pld.iloc[:, 0]

print(f"NVDA prices: {len(nvda)} points")
print(f"PLD prices: {len(pld)} points")

# Create a DataFrame with both stocks' close prices
data = pd.DataFrame({'NVDA': nvda, 'PLD': pld}).dropna()

print(f"Final data shape: {data.shape}")

if len(data) == 0:
    print("No data available for the specified period.")
else:
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Portfolio weights: 50% NVDA, 50% PLD
    weights = np.array([0.5, 0.5])

    # Portfolio daily returns
    portfolio_returns = returns.dot(weights)

    # Summary statistics for the portfolio
    summary_stats = {
        'Mean Daily Return': portfolio_returns.mean(),
        'Standard Deviation (Volatility)': portfolio_returns.std(),
        'Cumulative Return': (1 + portfolio_returns).prod() - 1,
        'Min Daily Return': portfolio_returns.min(),
        'Max Daily Return': portfolio_returns.max(),
        'Number of Trading Days': len(portfolio_returns)
    }

    # Display summary
    print("\nPortfolio Summary Statistics (50% NVDA, 50% PLD)")
    print("=" * 50)
    for key, value in summary_stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    # Optional: Display individual stock stats for comparison
    print("\nIndividual Stock Summary Statistics")
    print("=" * 40)
    for stock in ['NVDA', 'PLD']:
        stock_returns = returns[stock]
        print(f"\n{stock}:")
        print(f"  Mean Daily Return: {stock_returns.mean():.4f}")
        print(f"  Standard Deviation: {stock_returns.std():.4f}")
        print(f"  Cumulative Return: {(1 + stock_returns).prod() - 1:.4f}")
'''
import yfinance as yf
import pandas as pd
import numpy as np

# Define the time period
start_date = '2023-10-15'
end_date = '2025-10-15'

print(f"Analyzing data from {start_date} to {end_date}")

# Fetch data with proper handling
nvda_data = yf.download('NVDA', start=start_date, end=end_date, auto_adjust=True)
pld_data = yf.download('PLD', start=start_date, end=end_date, auto_adjust=True)

print(f"NVDA data type: {type(nvda_data)}")
print(f"NVDA shape: {nvda_data.shape}")
print(f"NVDA columns: {nvda_data.columns}")
print(f"PLD data type: {type(pld_data)}")
print(f"PLD shape: {pld_data.shape}")
print(f"PLD columns: {pld_data.columns}")

# Properly extract the Close prices
def extract_close_price(data, ticker):
    """Safely extract close prices from yfinance data"""
    print(f"\nExtracting {ticker} close prices...")
    
    # Check if we have MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        print(f"MultiIndex columns detected for {ticker}")
        print(f"Column levels: {data.columns.levels}")
        
        # Try to extract Close prices from MultiIndex
        if 'Close' in data.columns.get_level_values(0):
            close_prices = data['Close'].iloc[:, 0]  # Get first column of Close
            print(f"Successfully extracted Close prices using MultiIndex")
            return close_prices
        else:
            print("Warning: 'Close' not found in MultiIndex columns")
            return None
    else:
        # Regular Index columns
        print(f"Regular Index columns for {ticker}")
        if 'Close' in data.columns:
            return data['Close']
        else:
            print(f"Available columns: {data.columns.tolist()}")
            return data.iloc[:, 3]  # Usually Close is the 4th column (0-indexed 3)

# Extract prices
nvda_close = extract_close_price(nvda_data, 'NVDA')
pld_close = extract_close_price(pld_data, 'PLD')

print(f"\nNVDA close type: {type(nvda_close)}")
print(f"PLD close type: {type(pld_close)}")

if nvda_close is not None:
    print(f"NVDA close length: {len(nvda_close)}")
    print(f"NVDA close head:\n{nvda_close.head()}")
    
if pld_close is not None:
    print(f"PLD close length: {len(pld_close)}")
    print(f"PLD close head:\n{pld_close.head()}")

# Create DataFrame - handle different scenarios
if nvda_close is not None and pld_close is not None:
    # Both are Series with indexes
    data = pd.DataFrame({
        'NVDA': nvda_close, 
        'PLD': pld_close
    }).dropna()
    
    print(f"\nFinal data shape: {data.shape}")
    print(f"Final data:\n{data}")
    
    # Continue with analysis if we have data
    if len(data) > 0:
        returns = data.pct_change().dropna()
        print(f"\nReturns calculated for {len(returns)} days")

        
        # Portfolio analysis
        weights = np.array([0.5, 0.5])
        portfolio_returns = returns.dot(weights)
        
        print(f"\n📈 PORTFOLIO PERFORMANCE:")
        print(f"Mean daily return: {portfolio_returns.mean():.4f}")
        print(f"Total return: {(1 + portfolio_returns).prod() - 1:.4f}")
        print(f"Volatility: {portfolio_returns.std():.4f}")
        
    else:
        print("No data available after processing")
else:
    print("Failed to extract price data")
