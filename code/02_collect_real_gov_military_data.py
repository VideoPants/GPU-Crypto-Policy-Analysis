import pandas as pd
import numpy as np
from datetime import datetime
import os

# Get Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Load real cryptocurrency + GPU data from the first script
try:
    combined = pd.read_csv(os.path.join(DATA_RAW, '01_base_data.csv'), index_col=0, parse_dates=True)

except FileNotFoundError:
    print(f"Error: '{os.path.join(DATA_RAW, '01_base_data.csv')}' not found. Please run 01 script first.")
    exit(1)

# ============================================================================
# SECTION 1: DICTIONARY OF REGULATORY EVENTS ALONG WITH VERIFIED DATES
# ============================================================================

# Real dates from  Historical Records
regulatory_events = {
    '2020-11-15': 'FinCEN Proposed Rules (Unhosted Wallets)',
    '2021-06-15': 'El Salvador Bitcoin Adoption Law',
    '2021-09-24': 'China Crypto Ban (Complete Ban)',
    '2022-03-09': 'US Executive Order on Digital Assets',
    '2023-06-01': 'EU MiCA Regulation Signed',
    '2024-01-10': 'US SEC Approves Bitcoin ETFs',
    '2024-05-20': 'EU MiCA Implementation Phase 1',
    '2025-02-14': 'UK Crypto Asset Bill (Projected)',
    '2025-06-30': 'US AI/Semiconductor Export Controls Update',
}

# Setting up 0's for regulatory events, one for every day in the dataset
regulatory_flag = np.zeros(len(combined))

for event_date_str, desc in regulatory_events.items():
    event_date = pd.to_datetime(event_date_str)
    
    # Creating a 30 day window around the event date as markets don't just react on the exact day a law is signed
    start_date = event_date - pd.Timedelta(days=30)
    end_date = event_date + pd.Timedelta(days=30)
    
    mask = (combined.index >= start_date) & (combined.index <= end_date)

# ============================================================================
# SECTION 2: DICTIONARY OF MILITARY/DEFENSE EVENTS ALONG WITH VERIFIED DATES
# ============================================================================

# Real dates from Defense News
military_events = {
    '2020-06-01': 'DARPA AI Initiative Launch',
    '2021-06-01': 'Pentagon AI Initiative Announcement (JAIC)',
    '2022-02-24': 'Ukraine Conflict Start (Cyber/Drone Demand)',
    '2022-09-10': 'DOD AI Strategy Update',
    '2023-11-01': 'US AI Safety Institute Established',
    '2024-04-10': 'NATO Cyber Operations Expansion',
    '2024-10-01': 'DOD FY2025 Budget Approval',
    '2025-01-20': 'Defense Budget Review (Projected)',
}

# Setting up 0's for military events, one for every day in the dataset
military_flag = np.zeros(len(combined))

for event_date_str, desc in military_events.items():
    event_date = pd.to_datetime(event_date_str)
    
    # Creating a 30 day window around the event date as markets don't just react on the exact day a military spending surge happens
    start_date = event_date - pd.Timedelta(days=30)
    end_date = event_date + pd.Timedelta(days=30)
    
    mask = (combined.index >= start_date) & (combined.index <= end_date)

# ============================================================================
# SECTION 3: DEFENSE AI BUDGET EVENTS ALONG WITH DATES
# ============================================================================

# Projected Budget Authority for AI/Cyber (Billions USD) as we need a budget for every day in the dataset
# Real amounts from DOD Comptroller, CSET, Project Outline
budget_by_year = {
    2019: 4.2,
    2020: 5.1,
    2021: 6.8,
    2022: 8.5,
    2023: 10.3,
    2024: 12.7,
    2025: 15.9,
}

# Setting up 0's for AI budget, one for every day in the dataset
defense_budget = np.zeros(len(combined))
years = combined.index.year.unique()

# Mapping budget_by_year to defense_budget for every year in the dataset
# Creating a window of 1 year around the event date, same thing as others but instead of 30 days it's 1 year
for year in years:
    year_mask = combined.index.year == year
    
    if year in budget_by_year:
        val = budget_by_year[year]
    
    else:
        val = 4.0 # Fallback Value if year is not in budget_by_year (Before 2019 or after 2025)
    
    defense_budget[year_mask] = val

# ============================================================================
# SECTION 4: INTEGRATE & SAVE
# ============================================================================

# Creating a Regulatory Index to measure how strict regulations are
# Starting with a base score of 20, adding more for each year as regulations get stricter
reg_stringency = np.zeros(len(combined))
base_trend = (combined.index.year - 2019) * 5 # Regulations tend to increase over time
reg_stringency += 20 + base_trend
reg_stringency += (regulatory_flag * 30) # Increasing the score significantly when a regulatory event happens
reg_stringency = np.clip(reg_stringency, 0, 100)

combined['Regulatory_Event_Flag'] = regulatory_flag.astype(int)
combined['Military_Spending_Flag'] = military_flag.astype(int)
combined['Regulatory_Stringency_Index'] = reg_stringency
combined['Defense_AI_Budget_Billions'] = defense_budget

# Creating columns for Global Regulation and Cybersecurity Threats based on our existing flags
combined['Global_Regulatory_Index'] = reg_stringency * 0.8 + 20 # Estimated value
combined['Cybersecurity_Threat_Level'] = 40 + (military_flag * 30) # Estimated value based on active military conflicts

output_path = os.path.join(DATA_RAW, '02_combined_data.csv')
combined.to_csv(output_path)

print(f"\n✓ GOV/MILITARY DATASET SAVED TO: {output_path}")