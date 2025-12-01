# GPU-Crypto-Policy-Nexus 📉🏛️

**Analyzing the Hidden Drivers of GPU Prices: Cryptocurrency, Government Regulation, and Military Defense Spending.**

## 📋 Project Overview

This project explores the complex relationship between **GPU hardware prices** and three distinct market forces:

1.  **Cryptocurrency Markets:** Bitcoin & Ethereum mining demand.
2.  **Government Regulation:** The impact of crypto bans, SEC approvals, and global policy.
3.  **Military & Defense:** The "hidden driver" of AI/Cybersecurity defense procurement.

Using a **Knowledge Discovery in Databases (KDD)** approach, we integrated real-world data from 2019-2025 to prove that while Crypto drives volatility, Government and Military factors act as critical "regime shifters" in the market.

---

## 🚀 Key Features

- **Real-World Data Pipeline:** Integrates Yahoo Finance (Stocks/Crypto), Federal Register (Regulations), and USASpending.gov (Defense Budgets).
- **28 Engineered Features:** Includes Technical Indicators (RSI, MACD), Interaction Terms (`Gov_Crypto_Interaction`), and Volatility metrics.
- **Regime Detection:** Identifies distinct market states (e.g., "Regulated Bull Market", "Defense-Driven Demand").
- **Automated Preprocessing:** Full Python pipeline from raw API data to ML-ready scaled datasets.

---

## 📂 Repository Structure

```
├── code/
│   ├── 01_collect_real_crypto_gpu_data.py   # Fetches Crypto & Stock Data
│   ├── 02_collect_real_gov_military_data.py # Adds Gov/Military Events
│   ├── 03_integrate_real_data.py            # Merges & Cleans Data
│   └── 04_preprocessing_real_data.py        # Feature Engineering & Scaling
├── data/
│   ├── raw/                                 # Raw CSVs
│   └── processed/                           # Final ML-ready datasets
├── reports/
│   ├── Section_1_Introduction.md            # Problem Statement & Research Questions
│   └── Section_2_Approach.md                # Methodology & Data Dictionary
└── outputs/
    └── feature_engineering_documentation.json # Full feature list
```

---

## 📊 Data Sources

- **Cryptocurrency:** Bitcoin (BTC), Ethereum (ETH) via `yfinance`.
- **Stocks:** NVIDIA (NVDA), AMD (AMD), Intel (INTC).
- **Government:** Major regulatory events (EU MiCA, US GENIUS Act).
- **Military:** Defense AI Budget Authority & Geopolitical Conflict Events.

---

This repository represents the **Data Architect** phase of the project:

- ✅ **Data Collection:** Complete
- ✅ **Preprocessing:** Complete
- ✅ **Feature Engineering:** Complete
- ✅ **Ready for:** Data Mining & Analysis (Teammate 2)
