import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Fetch historical data
cl_data = yf.download('CL=F', start='2016-01-01', end='2024-01-01')
ng_data = yf.download('NG=F', start='2016-01-01', end='2024-01-01')

# Print column names to debug
print("CL=F Columns:", cl_data.columns)
print("NG=F Columns:", ng_data.columns)

# Flatten the MultiIndex by dropping the 'Ticker' level if needed
if isinstance(cl_data.columns, pd.MultiIndex):
    cl_data = cl_data.copy()  # Avoid modifying a view
    cl_data.columns = cl_data.columns.droplevel('Ticker')
if isinstance(ng_data.columns, pd.MultiIndex):
    ng_data = ng_data.copy()
    ng_data.columns = ng_data.columns.droplevel('Ticker')

# Use 'Adj Close' if available, otherwise use 'Close'
cl = cl_data['Adj Close'] if 'Adj Close' in cl_data.columns else cl_data['Close']
ng = ng_data['Adj Close'] if 'Adj Close' in ng_data.columns else ng_data['Close']

# Make copies to avoid SettingWithCopyWarning
cl = cl.copy()
ng = ng.copy()

# Replace zeros with NaN and drop them
cl.replace(0, np.nan, inplace=True)
ng.replace(0, np.nan, inplace=True)
cl.dropna(inplace=True)
ng.dropna(inplace=True)

# Ensure datasets are aligned by combining them and dropping rows with missing values
df = pd.DataFrame({'CL': cl, 'NG': ng}).dropna()

# Filter the DataFrame to include only rows with positive prices for both CL and NG.
df = df[(df['CL'] > 0) & (df['NG'] > 0)]

# Compute log price spread (this should no longer produce warnings)
df['Spread'] = np.log(df['CL']) - np.log(df['NG'])
df.dropna(inplace=True)

# Calculate rolling mean and standard deviation
window = 30
df['Rolling Mean'] = df['Spread'].rolling(window=window).mean()
df['Rolling Std'] = df['Spread'].rolling(window=window).std()
df['Z-Score'] = (df['Spread'] - df['Rolling Mean']) / df['Rolling Std']
df.dropna(inplace=True)

# Augmented Dickey-Fuller test (ensure input is an array)
def adf_test(series):
    series_array = series.dropna().values
    result = adfuller(series_array)
    return result[1]  # Return p-value

# Optional: Check length of Spread series to make sure we have enough data
print("Length of Spread series:", len(df['Spread']))
print(f'ADF p-value: {adf_test(df["Spread"])}')

# Trade signals
long_signal = df['Z-Score'] < -2
short_signal = df['Z-Score'] > 2

# Simulated PnL using trade signals (tracked positions)
df['Position'] = 0  # 0 = no position, 1 = long, -1 = short

# Open long position when Z-Score < -2
df['Position'] = np.where(long_signal, 1, df['Position'])

# Open short position when Z-Score > 2
df['Position'] = np.where(short_signal, -1, df['Position'])

# Ensure the position doesn't switch multiple times by using shift() and avoid overlap
df['Position'] = df['Position'].shift(1).fillna(0)  # Carry previous position if no new signal

# Calculate simulated PnL
pnl = df['Z-Score'].shift(1) * df['Position'].diff().shift(-1)  # Calculate the profit/loss



# ## Sharpe Ratio Calculation
# def sharpe_ratio(returns, risk_free_rate=0.02):
#     # Adjust for daily risk-free rate if using daily returns
#     excess_returns = returns - risk_free_rate
#     return excess_returns.mean() / excess_returns.std()

# Sharpe Ratio Calculation
def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

print(f'Sharpe Ratio: {sharpe_ratio(pnl)}')

# Define a volatility-based stop-loss based on the rolling standard deviation
stop_loss_factor = 2  # Example: 2x the rolling standard deviation
df['Upper Stop-Loss'] = df['Rolling Mean'] + (stop_loss_factor * df['Rolling Std'])
df['Lower Stop-Loss'] = df['Rolling Mean'] - (stop_loss_factor * df['Rolling Std'])

# Track stop-loss exits based on the spread
df['Stop-Loss Exit'] = 0  # 0 = no stop-loss exit, 1 = triggered stop-loss

# Check if price crosses the stop-loss levels
df['Stop-Loss Exit'] = np.where((df['Spread'] > df['Upper Stop-Loss']) | (df['Spread'] < df['Lower Stop-Loss']), 1, 0)

# Plot price spread with rolling mean and bands
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Spread'], label='Price Spread')
plt.plot(df.index, df['Rolling Mean'], linestyle='dashed', label='Rolling Mean')
plt.fill_between(df.index,
                 df['Rolling Mean'] - df['Rolling Std'],
                 df['Rolling Mean'] + df['Rolling Std'],
                 color='gray', alpha=0.2)
plt.legend()
plt.title('Crude Oil - Natural Gas Price Spread')
plt.show()

# Plot z-score with trade signals
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Z-Score'], label='Z-score')
plt.axhline(2, color='red', linestyle='dashed', label='Short Signal')
plt.axhline(-2, color='green', linestyle='dashed', label='Long Signal')
plt.axhline(0, color='black', linestyle='dashed', label='Mean')
plt.legend()
plt.title('Z-score and Trade Signals')
plt.show()

# Check if df is populated and contains the 'CL' and 'NG' columns
print(f'DataFrame shape: {df.shape}')
print(f'DataFrame columns: {df.columns}')
print(f'First few rows of df:\n{df.head()}')

# Define the spread using the 'Close' prices for 'CL' and 'NG'
spread = np.log(df['CL']) - np.log(df['NG'])

# Check if spread has values and print the first few rows
print(f'Spread preview:\n{spread.head()}')

# Ensure the spread is not empty
if spread.isnull().all():
    print("Spread is empty or contains only NaNs!")
else:
    print(f'Spread contains data: {spread.describe()}')

# Define the ADF test function
def adf_test(series):
    print("Running ADF test...")
    result = adfuller(series)
    print(f"ADF Test Results: {result}")
    return result[1]  # p-value

# Define the half-life calculation function
def half_life(series):
    print("Running half-life calculation...")
    delta = series.diff().dropna()
    lagged = series.shift(1).dropna()
    beta = np.polyfit(lagged, delta, 1)[0]
    return -np.log(2) / beta

# Test stationarity and half-life
if not spread.isnull().all():
    print(f'ADF p-value: {adf_test(spread)}')
    print(f'Half-life: {half_life(spread)}')
else:
    print("Spread is not valid for testing.")
