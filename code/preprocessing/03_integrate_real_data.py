import pandas as pd
import numpy as np
import os

# Get Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Load the combined dataset from script 02
try:
    data = pd.read_csv(os.path.join(DATA_RAW, '02_combined_data.csv'), index_col=0, parse_dates=True)

except FileNotFoundError:
    print(f"Error: '{os.path.join(DATA_RAW, '02_combined_data.csv')}' not found.")
    exit(1)

# ============================================================================
# SECTION 1: CLEANING THE DATA
# ============================================================================

# Forward fill (filling empty data with last known price/value)
data_clean = data.ffill()

# Backward fill (filling empty data with next known price/value)
data_clean = data_clean.bfill()

# Remove any remaining empty rows if there are any
data_clean = data_clean.dropna()

# ============================================================================
# SECTION 2: SAVE CLEANED DATA
# ============================================================================

output_path = os.path.join(DATA_RAW, '03_cleaned_data.csv')
data_clean.to_csv(output_path)

print(f"\n✓ CLEANED DATASET SAVED TO: {output_path}")