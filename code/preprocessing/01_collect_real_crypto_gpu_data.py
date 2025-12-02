import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import os

# Getting Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Defining the date range
start_date = "2019-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# ============================================================================
# SECTION 1: DOWNLOAD REAL DATA FROM "yfinance"
# ============================================================================

tickers = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    'NVDA': 'NVDA',
    'AMD': 'AMD',
    'INTC': 'INTC'
}

data_frames = {}
for ticker, name in tickers.items():
    try:
        # Using Ticker.history as it is more reliable
        dat = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        # Normalizing index to remove timezones
        if dat.index.tz is not None:
            dat.index = dat.index.tz_localize(None)
        
        data_frames[name] = dat
    
    except Exception as e:
        print(f"    ! Error fetching {ticker}: {e}")

# ============================================================================
# SECTION 2: ALIGNING & INTEGRATING THE DATASETS
# ============================================================================

# Basing the DataFrame on Bitcoin as trades happen 24/7 for cryptocurrencies
btc = data_frames['BTC'][['Close', 'Volume']].rename(columns={'Close': 'BTC_Price', 'Volume': 'BTC_Volume'})
eth = data_frames['ETH'][['Close', 'Volume']].rename(columns={'Close': 'ETH_Price', 'Volume': 'ETH_Volume'})
nvda = data_frames['NVDA'][['Close', 'Volume']].rename(columns={'Close': 'NVDA_Close', 'Volume': 'NVDA_Volume'})
amd = data_frames['AMD'][['Close', 'Volume']].rename(columns={'Close': 'AMD_Close', 'Volume': 'AMD_Volume'})
intc = data_frames['INTC'][['Close', 'Volume']].rename(columns={'Close': 'INTC_Close', 'Volume': 'INTC_Volume'})

# Combining all the datasets
combined = pd.concat([btc, eth, nvda, amd, intc], axis=1)

# Forward and backward fill stock data the fill empty data for weekends and holidays with "best guess" values
combined['NVDA_Close'] = combined['NVDA_Close'].ffill().bfill()
combined['AMD_Close'] = combined['AMD_Close'].ffill().bfill()
combined['INTC_Close'] = combined['INTC_Close'].ffill().bfill()

# ============================================================================
# SECTION 3: CREATING GPU INDEXES & LAUNCH DATES
# ============================================================================

# Real GPU Stock Index to track how much the market has grown or dropped since 2019 till the current date
# Normalize to start
nvda_norm = combined['NVDA_Close'] / combined['NVDA_Close'].iloc[0] * 100
amd_norm = combined['AMD_Close'] / combined['AMD_Close'].iloc[0] * 100
intc_norm = combined['INTC_Close'] / combined['INTC_Close'].iloc[0] * 100

# Creating the GPU Stock Index by combining all the stock prices with their weights
combined['GPU_Stock_Index'] = (nvda_norm * 0.4) + (amd_norm * 0.4) + (intc_norm * 0.2)

# GPU Launch Dates
launch_dates = []
try:
    url = "https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json"
    
    response = requests.get(url)
    gpu_data = response.json()
    
    for model, details in gpu_data.items():
        if "Launch" in details and details["Launch"] and details["Launch"] != "nan":
            try:
                dt = pd.to_datetime(details["Launch"])
                if dt >= pd.to_datetime(start_date) and dt <= pd.to_datetime(end_date):
                    launch_dates.append(dt)
            
            except: pass

except:
    print("    ! Warning: Could not fetch gpu.json")

combined['GPU_Launch_Dates'] = 0
for date in launch_dates:
    # Finding the nearest trading date in the index
    try:
        idx = combined.index.get_indexer([date], method='nearest')[0]
        combined.iloc[idx, combined.columns.get_loc('GPU_Launch_Dates')] = 1
    
    except: pass

# ============================================================================
# SECTION 4: SAVING DATA TO "data/raw/" FOLDER
# ============================================================================
output_path = os.path.join(DATA_RAW, '01_base_data.csv')
combined.to_csv(output_path)

print(f"\n✓ DATASET SAVED TO: {output_path}")