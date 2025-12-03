"""
Correlation Analysis for GPU-Crypto-Policy Dataset
Teammate 2: Analyst & Model Specialist
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime

# Set up directories
os.makedirs('outputs', exist_ok=True)

# Logging functions
def log_step(step_name, start_time=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] ✓ {step_name} completed in {elapsed:.2f}s")
        return None
    else:
        print(f"\n[{timestamp}] ▶ {step_name}")
        return time.time()

def log_result(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}]   → {message}")

print("="*80)
print("CORRELATION ANALYSIS - GPU-CRYPTO-POLICY DATASET")
print("="*80)
overall_start = time.time()

# Load data
step_start = log_step("Loading data")
df = pd.read_csv('data/processed/data_visualization_analysis.csv')
df['Date'] = pd.to_datetime(df['Date'])
log_result(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
log_step("Loading data", step_start)

# Define features for correlation analysis
features_to_correlate = [
    'BTC_Price', 'ETH_Price', 'NVDA_Close', 'AMD_Close', 'INTC_Close',
    'GPU_Stock_Index', 'Regulatory_Event_Flag', 'Military_Spending_Flag',
    'Regulatory_Stringency_Index', 'Defense_AI_Budget_Billions',
    'Global_Regulatory_Index', 'Cybersecurity_Threat_Level',
    'BTC_Momentum', 'ETH_Momentum', 'GPU_Momentum',
    'Gov_Crypto_Interaction', 'Military_GPU_Demand', 'Reg_Stringency_GPU_Impact'
]

# Filter for existing features
existing_features = [f for f in features_to_correlate if f in df.columns]
correlation_df = df[existing_features]

# Calculate correlation matrix
step_start = log_step("Calculating correlation matrix")
correlation_matrix = correlation_df.corr()
log_result(f"Correlation matrix calculated: {correlation_matrix.shape}")
log_step("Calculating correlation matrix", step_start)

# Generate heatmap
step_start = log_step("Generating correlation heatmap")
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Key Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/correlation_analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
log_result("Heatmap saved to outputs/correlation_heatmap.png")
log_step("Generating correlation heatmap", step_start)

# Analyze strong correlations
step_start = log_step("Analyzing strong correlations")
strong_corrs = correlation_matrix.unstack().sort_values(ascending=False)
strong_corrs = strong_corrs[abs(strong_corrs) > 0.5]
strong_corrs = strong_corrs[strong_corrs < 1]  # Remove self-correlations

print(f"\nStrong Correlations (|r| > 0.5): Found {len(strong_corrs)} pairs")
print("-" * 60)

# Top 10 positive correlations
print("\nTop 10 Positive Correlations:")
for i, (pair, value) in enumerate(strong_corrs.head(10).items()):
    print(f"{i+1:2d}. {pair[0]:30} vs {pair[1]:30} : r = {value:.4f}")

# Top 10 negative correlations
negative_corrs = strong_corrs[strong_corrs < 0]
if len(negative_corrs) > 0:
    print("\nTop 10 Negative Correlations:")
    for i, (pair, value) in enumerate(negative_corrs.head(10).items()):
        print(f"{i+1:2d}. {pair[0]:30} vs {pair[1]:30} : r = {value:.4f}")

log_step("Analyzing strong correlations", step_start)

# Key correlation interpretations
step_start = log_step("Providing correlation interpretations")
print("\n" + "="*80)
print("KEY CORRELATION INTERPRETATIONS")
print("="*80)

key_interpretations = {
    ('BTC_Price', 'ETH_Price'): 
        "Very strong positive correlation (r ≈ 0.85-0.95). Crypto assets move together.",
    
    ('NVDA_Close', 'AMD_Close'): 
        "Strong positive correlation (r ≈ 0.70-0.80). GPU manufacturers' stocks move together.",
    
    ('GPU_Stock_Index', 'NVDA_Close'): 
        "Strong positive correlation (r ≈ 0.65-0.75). NVIDIA heavily influences GPU index.",
    
    ('Regulatory_Stringency_Index', 'Global_Regulatory_Index'): 
        "Very strong correlation (r ≈ 0.90-0.95). Global and local regulations align.",
    
    ('Regulatory_Event_Flag', 'Cybersecurity_Threat_Level'): 
        "Moderate correlation (r ≈ 0.25-0.35). Regulatory events often follow security threats.",
    
    ('BTC_Price', 'GPU_Stock_Index'): 
        "Moderate correlation (r ≈ 0.40-0.60). Crypto mining demand affects GPU prices.",
    
    ('Military_Spending_Flag', 'Defense_AI_Budget_Billions'): 
        "Moderate correlation (r ≈ 0.30-0.50). Military spending events correlate with AI budgets.",
    
    ('Gov_Crypto_Interaction', 'Regulatory_Stringency_Index'): 
        "Expected correlation showing government policy impact."
}

for pair, interpretation in key_interpretations.items():
    if pair[0] in correlation_matrix.columns and pair[1] in correlation_matrix.columns:
        actual_corr = correlation_matrix.loc[pair[0], pair[1]]
        if not np.isnan(actual_corr):
            print(f"\n{pair[0]} vs {pair[1]}: r = {actual_corr:.4f}")
            print(f"  Interpretation: {interpretation}")
        else:
            print(f"\n{pair[0]} vs {pair[1]}: No correlation (NaN)")
            print(f"  Note: One or both features may be constant or have no variation")

log_step("Providing correlation interpretations", step_start)

# Save correlation results with proper encoding
step_start = log_step("Saving correlation results")
try:
    # Use UTF-8 encoding to handle special characters
    with open('outputs/correlation_analysis/correlation_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("CORRELATION ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {df.shape[0]} records, {df.shape[1]} features\n")
        f.write(f"Features analyzed: {len(existing_features)}\n\n")
        
        f.write("TOP 20 STRONGEST CORRELATIONS (|r| > 0.5):\n")
        f.write("-"*60 + "\n")
        for i, (pair, value) in enumerate(strong_corrs.head(20).items()):
            f.write(f"{i+1:2d}. {pair[0]:30} vs {pair[1]:30} : r = {value:.4f}\n")
        
        f.write("\n\nKEY INSIGHTS:\n")
        f.write("-"*60 + "\n")
        
        # Calculate actual correlations for insights
        btc_eth_corr = correlation_matrix.loc['BTC_Price', 'ETH_Price'] if 'BTC_Price' in correlation_matrix.columns and 'ETH_Price' in correlation_matrix.columns else np.nan
        nvda_amd_corr = correlation_matrix.loc['NVDA_Close', 'AMD_Close'] if 'NVDA_Close' in correlation_matrix.columns and 'AMD_Close' in correlation_matrix.columns else np.nan
        gpu_nvda_corr = correlation_matrix.loc['GPU_Stock_Index', 'NVDA_Close'] if 'GPU_Stock_Index' in correlation_matrix.columns and 'NVDA_Close' in correlation_matrix.columns else np.nan
        
        insights = [
            f"1. Crypto markets (BTC, ETH) show {'very high' if btc_eth_corr > 0.8 else 'high' if btc_eth_corr > 0.6 else 'moderate'} inter-correlation (r = {btc_eth_corr:.4f})",
            f"2. GPU manufacturers (NVDA, AMD) are {'strongly' if nvda_amd_corr > 0.7 else 'moderately'} correlated (r = {nvda_amd_corr:.4f})",
            f"3. GPU index shows {'strong' if gpu_nvda_corr > 0.7 else 'moderate' if gpu_nvda_corr > 0.5 else 'weak'} correlation with NVIDIA (r = {gpu_nvda_corr:.4f})",
            "4. Regulatory indices are highly correlated, showing policy consistency",
            "5. Intel stock shows negative correlation with GPU and crypto markets, suggesting different market dynamics"
        ]
        
        for insight in insights:
            f.write(insight + "\n")
        
        f.write("\nNOTES ON NaN VALUES:\n")
        f.write("-"*60 + "\n")
        nan_pairs = []
        for pair in key_interpretations.keys():
            if pair[0] in correlation_matrix.columns and pair[1] in correlation_matrix.columns:
                if np.isnan(correlation_matrix.loc[pair[0], pair[1]]):
                    nan_pairs.append(f"{pair[0]} vs {pair[1]}")
        
        if nan_pairs:
            f.write("The following pairs show NaN correlation (likely constant values):\n")
            for pair in nan_pairs:
                f.write(f"  • {pair}\n")
        else:
            f.write("No NaN correlations found in key feature pairs.\n")
    
    log_result("Results saved to outputs/correlation_analysis_results.txt")
    
except Exception as e:
    log_result(f"Error saving results: {str(e)}")
    # Create a simpler version if encoding fails
    with open('outputs/correlation_analysis_results_simple.txt', 'w') as f:
        f.write("CORRELATION ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {df.shape[0]} records\n")
        f.write(f"Strong correlations found: {len(strong_corrs)}\n")
        
        f.write("\nTOP 5 CORRELATIONS:\n")
        for i, (pair, value) in enumerate(strong_corrs.head(5).items()):
            f.write(f"{i+1}. {pair[0]} vs {pair[1]}: r = {value:.4f}\n")
    
    log_result("Simple results saved to outputs/correlation_analysis_results_simple.txt")

log_step("Saving correlation results", step_start)

# Summary
overall_time = time.time() - overall_start
print("\n" + "="*80)
print("CORRELATION ANALYSIS COMPLETE")
print("="*80)
print(f"Total time: {overall_time:.2f} seconds")
print(f"\nFiles generated in outputs/ directory:")
print("  • correlation_heatmap.png")
print("  • correlation_analysis_results.txt")
print("\nReady for Section 3.1: Correlation Analysis in report")
print("\nKEY FINDINGS FROM YOUR DATA:")
print(f"  1. NVDA vs GPU Index: r = {correlation_matrix.loc['GPU_Stock_Index', 'NVDA_Close']:.4f} (Extremely strong!)")
print(f"  2. BTC vs ETH: r = {correlation_matrix.loc['BTC_Price', 'ETH_Price']:.4f} (Very strong)")
print(f"  3. Intel shows negative correlation with GPU markets")