# GPU-Crypto-Policy-Analysis 📉🏛️

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
- ✅ **Data Mining & Analysis:** Complete



## 🔬 Data Analysis & Model Implementation

Correlation Analysis

- **Objective:** Identify relationships between GPU prices and market factors
- **Method:** Pearson correlation matrix with 18 key features
- **Key Findings:**
    - Extremely strong correlation between NVDA and GPU Index (r = 0.9960)
    - Very strong BTC-ETH correlation (r = 0.8118)
    - Moderate GPU-Crypto correlation (r = 0.9135)
    - Intel shows negative correlation with GPU markets
- **Outputs:**
    - correlation_heatmap.png: Visual correlation matrix
    - correlation_analysis_results.txt: Detailed correlation results

Regression Analysis (Ridge Regression)

- **Objective:** Predict GPU_Stock_Index using normalized features
- **Method:** Ridge Regression with α=1.0, 80/20 train-test split
- **Metrics:**
    - R² Score: Explained variance of GPU prices
    - RMSE & MAE: Prediction error metrics
    - Feature coefficients: Impact of each feature
- **Key Insights:**
    - Negative military coefficient reveals complex market dynamics
    - Regulatory features significantly impact GPU prices
    - Crypto market features are strong predictors
- **Outputs:**
    - ridge_coefficients.png: Top 15 feature coefficients
    - regression_actual_vs_predicted.png: Model performance visualization
    - regression_analysis_results.txt: Complete regression metrics

Classification Analysis (Random Forest)

- **Objective:** Classify GPU_Price_Category (Low/Medium/High)
- **Method:** Random Forest with 100 trees, max_depth=10, stratified split
- **Metrics:**
    - Accuracy: Percentage of correct classifications
    - Precision & Recall: Class-specific performance
    - Feature Importance: Most predictive features
- **Key Insights:**
    - Government/military features rank high in importance
    - Model validates three-way market interaction hypothesis
    - High classification accuracy achieved
- **Outputs:**
    - confusion_matrix.png: Classification performance matrix
    - feature_importance_rf.png: Top 20 feature importances
    - classification_performance_by_class.png: Per-class metrics
    - classification_analysis_results.txt: Full classification report

Clustering Analysis (K-Means)

- **Objective:** Identify distinct market regimes
- **Method:** K-Means clustering with Elbow Method optimization
- **Results:**
    - Optimal clusters: k=2 (from silhouette and inertia analysis)
    - Cluster 0: High-Performance, High-Regulation Market
    - Cluster 1: Low-Performance, High-Regulation Market
- **Key Insights:**
    - Government policies create distinct market regimes
    - Military spending correlates with specific cluster characteristics
    - Time series shows regime evolution
- **Outputs:**
    - elbow_silhouette.png: Optimal k determination
    - clustering_pca.png: 2D cluster visualization
    - clusters_over_time.png: Temporal cluster distribution
    - cluster_radar_chart.png: Multi-dimensional cluster comparison
    - clustering_analysis_results.txt: Complete cluster profiles
    - data_with_clusters.csv: Dataset with cluster labels


This repository represents the complete project implementation:

- ✅ **Data Collection:** Complete
- ✅ **Preprocessing:** Complete
- ✅ **Feature Engineering:** Complete
- ✅ **Correlation Analysis:** Complete
- ✅ **Regression Analysis:** Complete
- ✅ **Classification Analysis:** Complete
- ✅ **Clustering Analysis:** Complete
- ✅ **Ready for:** Teammate 3


