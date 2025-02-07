import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Fetch Data and Clean Columns
# -------------------------------

# Fetch historical data
cl_data = yf.download('CL=F', start='2021-01-01', end='2025-01-01')
g_data = yf.download('NG=F', start='2021-01-01', end='2025-01-01')

# Display the multi-index column structure for debugging
print("CL Data Columns:", cl_data.columns)
print("NG Data Columns:", g_data.columns)

# Extract the full DataFrame (Close, High, Low) as a copy
cl = cl_data[['Close', 'High', 'Low']].copy()
g = g_data[['Close', 'High', 'Low']].copy()

# Define a helper function to clean multi-index columns by stripping whitespace
def clean_multiindex_columns(df):
    new_cols = [(str(col[0]).strip(), str(col[1]).strip()) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_cols, names=df.columns.names)
    return df

# Clean the columns of both DataFrames
cl = clean_multiindex_columns(cl)
g = clean_multiindex_columns(g)

print("CL columns after cleaning:", cl.columns)
print("NG columns after cleaning:", g.columns)

# -------------------------------
# Step 2: Flatten Columns for Easy Access
# -------------------------------
# Flatten the multi-index columns into single-level names.
cl.columns = [f"{col[0]}_{col[1]}" for col in cl.columns]
g.columns = [f"{col[0]}_{col[1]}" for col in g.columns]

print("CL columns after flattening:", cl.columns)
print("NG columns after flattening:", g.columns)

# -------------------------------
# Step 3: Preprocessing
# -------------------------------

# Check sample data and missing values
print("CL data sample:\n", cl.head())
print("NG data sample:\n", g.head())
print("Missing values in CL:\n", cl.isna().sum())
print("Missing values in NG:\n", g.isna().sum())

# Replace zeros with NaN (using .loc to avoid SettingWithCopyWarning)
cl.loc[:, ['Close_CL=F', 'High_CL=F', 'Low_CL=F']] = cl.loc[:, ['Close_CL=F', 'High_CL=F', 'Low_CL=F']].replace(0, np.nan)
g.loc[:, ['Close_NG=F', 'High_NG=F', 'Low_NG=F']] = g.loc[:, ['Close_NG=F', 'High_NG=F', 'Low_NG=F']].replace(0, np.nan)

# Align the two datasets by their index (rows) only
cl, g = cl.align(g, join='inner', axis=0)
print(f"After alignment, CL length: {len(cl)}, NG length: {len(g)}")
print("CL index:", cl.index[:5])
print("NG index:", g.index[:5])

# -------------------------------
# Step 4: Extract 'Close' Prices
# -------------------------------

# Now, extract the Close prices using the flattened column names.
df = pd.DataFrame({
    'CL': cl['Close_CL=F'].values,
    'NG': g['Close_NG=F'].values
}, index=cl.index).dropna()

if df.empty:
    raise ValueError("Error: No valid data left after cleaning. Check data sources.")

# Compute log price spread
df['Spread'] = np.log(df['CL']) - np.log(df['NG'])

# -------------------------------
# Step 5: Strategy Calculations
# -------------------------------

# Calculate rolling mean and standard deviation for the spread
window = 10
df['Rolling Mean'] = df['Spread'].rolling(window=window).mean()
df['Rolling Std'] = df['Spread'].rolling(window=window).std()
df['Z-Score'] = (df['Spread'] - df['Rolling Mean']) / df['Rolling Std']
df.dropna(inplace=True)

# Generate trade signals based on Z-Score thresholds
long_signal = df['Z-Score'] < -2
short_signal = df['Z-Score'] > 2

df['Position'] = 0  # 0 = no position, 1 = long, -1 = short
df['Position'] = np.where(long_signal, 1, df['Position'])
df['Position'] = np.where(short_signal, -1, df['Position'])
df['Position'] = df['Position'].shift(1).fillna(0)

# Calculate simulated PnL (based on the change in the spread)
df['PnL'] = df['Position'] * df['Spread'].diff()

# -------------------------------
# Step 6: ATR-Based Stop Loss & Position Sizing for CL
# -------------------------------

# For Crude Oil (CL), calculate ATR:
# ATR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
cl['ATR'] = np.maximum(
    cl['High_CL=F'] - cl['Low_CL=F'],
    abs(cl['High_CL=F'] - cl['Close_CL=F'].shift(1)),
    abs(cl['Low_CL=F'] - cl['Close_CL=F'].shift(1))
)

# Set Stop Loss and Take Profit levels as multiples of ATR
ATR_multiple = 1.5  # Adjust this multiplier as needed
cl['Stop_Loss'] = cl['Close_CL=F'] - ATR_multiple * cl['ATR']
cl['Take_Profit'] = cl['Close_CL=F'] + ATR_multiple * cl['ATR']

# Position sizing based on volatility (ATR)
risk_per_trade = 0.02  # Risk 2% of portfolio per trade
capital = 500000       # Example portfolio size

cl['Position_Size'] = (capital * risk_per_trade) / (ATR_multiple * cl['ATR'])

# Display a sample of ATR, Stop Loss, Take Profit, and Position Size
print(cl[['Close_CL=F', 'ATR', 'Stop_Loss', 'Take_Profit', 'Position_Size']].head())

# Visualize Crude Oil (CL) with Stop Loss and Take Profit levels
cl[['Close_CL=F', 'Stop_Loss', 'Take_Profit']].plot(figsize=(10, 6))
plt.title("Crude Oil (CL=F) with Stop Loss and Take Profit")
plt.show()

# -------------------------------
# Step 7: Performance Metrics
# -------------------------------

def sharpe_ratio(returns, risk_free_rate=0.01):
    mean_excess_return = returns.mean() - risk_free_rate / 250  # Daily risk-free rate
    std_excess_return = returns.std()
    return (mean_excess_return * 250) / (std_excess_return * np.sqrt(250))

def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdowns = cumulative_returns - cumulative_returns.cummax()
    return drawdowns.min()

def win_loss_ratio(returns):
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return len(wins) / len(losses) if len(losses) > 0 else np.nan

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    return gross_profit / gross_loss if gross_loss > 0 else np.nan
# Additional Performance Metrics
def annualized_return(returns, periods_per_year=250):
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(returns)) - 1
    return annualized_return

def sortino_ratio(returns, target_return=0.01, periods_per_year=250):
    downside_returns = returns[returns < target_return]
    downside_deviation = downside_returns.std()
    excess_return = returns.mean() - target_return / periods_per_year
    return excess_return / downside_deviation * np.sqrt(periods_per_year)


returns = df['PnL'].dropna()
print()
print(f"Sharpe Ratio: {sharpe_ratio(returns)}")
print(f"Maximum Drawdown: {max_drawdown(returns)}")
print(f"Win/Loss Ratio: {win_loss_ratio(returns)}")
print(f"Profit Factor: {profit_factor(returns)}")
print(f"Annualized Return: {annualized_return(returns)}")
print(f"Sortino Ratio: {sortino_ratio(returns)}")
