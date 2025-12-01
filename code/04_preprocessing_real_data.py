import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os

# Get Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUTS = os.path.join(PROJECT_ROOT, 'outputs')

# Load cleaned data from script 03
try:
    df = pd.read_csv(os.path.join(DATA_RAW, '03_cleaned_data.csv'), index_col=0, parse_dates=True)

except FileNotFoundError:
    print("Error: Cleaned data not found.")
    exit(1)

# ============================================================================
# SECTION 1: CREATING TECHNICAL INDICATORS FROM THE CLEANED DATASET
# ============================================================================

# Calculating Averages of 7 or 30 days to smooth noise and highlight trends
df['BTC_MA_7'] = df['BTC_Price'].rolling(7).mean()
df['BTC_MA_30'] = df['BTC_Price'].rolling(30).mean()
df['ETH_MA_7'] = df['ETH_Price'].rolling(7).mean()

# Use GPU Stock Index for "GPU Price Index" trends
df['GPU_MA_7'] = df['GPU_Stock_Index'].rolling(7).mean()

# Calculating change in price from previous day to highlight momentum/movement in price
df['BTC_Momentum'] = df['BTC_Price'].pct_change()
df['ETH_Momentum'] = df['ETH_Price'].pct_change()
df['GPU_Momentum'] = df['GPU_Stock_Index'].pct_change()
df['NVDA_Momentum'] = df['NVDA_Close'].pct_change()

# Calculating Standard Deviation of 7 days to highlight price stability
df['BTC_Volatility'] = df['BTC_Price'].rolling(7).std()
df['GPU_Volatility'] = df['GPU_Stock_Index'].rolling(7).std()

# Creates a column contains prices of the previous day, used for comparision
df['BTC_Price_Lag1'] = df['BTC_Price'].shift(1)
df['ETH_Price_Lag1'] = df['ETH_Price'].shift(1)
df['GPU_Price_Lag1'] = df['GPU_Stock_Index'].shift(1)

# Average trading volume of 7 days to highlight trading activity
df['BTC_Volume_MA'] = df['BTC_Volume'].rolling(7).mean()

# Comparing Crypto prices to GPU prices to see if one is overvalued compared to the other
# Adding +1 to avoid zero errors
df['BTC_to_GPU_Ratio'] = df['BTC_Price'] / (df['GPU_Stock_Index'] + 1)
df['ETH_to_GPU_Ratio'] = df['ETH_Price'] / (df['GPU_Stock_Index'] + 1)

# ============================================================================
# SECTION 2: CREATING COMPARATIVE INDICATORS FROM THE CLEANED DATASET
# ============================================================================

# Compares the impact of regulations specifically when Bitcoin prices are high
df['Gov_Crypto_Interaction'] = (df['Regulatory_Event_Flag'] * (df['BTC_Price'] / df['BTC_Price'].mean()))

# Compares the impact of military spending specifically when GPU prices are high
df['Military_GPU_Demand'] = (df['Military_Spending_Flag'] * (df['GPU_Stock_Index'] / df['GPU_Stock_Index'].mean()))

# Compares the long-term pressure of regulations on GPU prices
df['Reg_Stringency_GPU_Impact'] = (df['Regulatory_Stringency_Index'] * (df['GPU_Stock_Index'] / df['GPU_Stock_Index'].mean()))

# ============================================================================
# SECTION 3: HANDLING EMPTY VALUES
# ============================================================================

df_clean = df.dropna().copy()
nan_removed = len(df) - len(df_clean)

# ============================================================================
# SECTION 4: CATEGORICAL FEATURES
# ============================================================================

# Categorize GPU Stock Index into Low, Medium, or High
q1 = df_clean['GPU_Stock_Index'].quantile(0.33)
q2 = df_clean['GPU_Stock_Index'].quantile(0.67)

def categorize_gpu(val):
    if val < q1: return 'Low'
    elif val < q2: return 'Medium'
    else: return 'High'

df_clean['GPU_Price_Category'] = df_clean['GPU_Stock_Index'].apply(categorize_gpu)

# ============================================================================
# SECTION 5: SCALING & SAVING
# ============================================================================

# Selecting all numerical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Using StandardScaler to apply Z-Score Normalization
# This ensures that large numbers don't skew the machine learning model
scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

# Saving the processed data
df_clean.to_csv(os.path.join(DATA_PROCESSED, 'data_visualization_analysis.csv'))
df_scaled.to_csv(os.path.join(DATA_PROCESSED, 'data_machinelearning_clustering.csv'))

print(f"\n✓ Saved: {os.path.join(DATA_PROCESSED, 'data_visualization_analysis.csv')}")
print(f"✓ Saved: {os.path.join(DATA_PROCESSED, 'data_machinelearning_clustering.csv')}")

# Saving Documentation
feature_doc = {
    "Original Features": ["BTC_Price", "BTC_Volume", "ETH_Price", "ETH_Volume", "NVDA_Close", "NVDA_Volume", "AMD_Close", "AMD_Volume", "INTC_Close", "INTC_Volume", "GPU_Stock_Index"],
    "Gov/Military Features": ["Regulatory_Event_Flag", "Military_Spending_Flag", "Regulatory_Stringency_Index", "Defense_AI_Budget_Billions", "Global_Regulatory_Index", "Cybersecurity_Threat_Level"],
    "Technical Indicators": ["BTC_MA_7", "BTC_MA_30", "ETH_MA_7", "GPU_MA_7", "BTC_Momentum", "ETH_Momentum", "GPU_Momentum", "NVDA_Momentum", "BTC_Volatility", "GPU_Volatility", "BTC_Price_Lag1", "ETH_Price_Lag1", "GPU_Price_Lag1", "BTC_Volume_MA", "BTC_to_GPU_Ratio", "ETH_to_GPU_Ratio"],
    "Interaction Features": ["Gov_Crypto_Interaction", "Military_GPU_Demand", "Reg_Stringency_GPU_Impact"],
    "Categorical": ["GPU_Price_Category"]
}

with open(os.path.join(OUTPUTS, 'feature_engineering_documentation.json'), 'w') as f:
    json.dump(feature_doc, f, indent=2)