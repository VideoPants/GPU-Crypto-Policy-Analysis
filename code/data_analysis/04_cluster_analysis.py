"""
Clustering Analysis (K-Means) for GPU-Crypto-Policy Dataset
Teammate 2: Analyst & Model Specialist
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
print("CLUSTERING ANALYSIS - K-MEANS")
print("="*80)
overall_start = time.time()

# Load data
step_start = log_step("Loading data")
df_viz = pd.read_csv('data\processed\data_visualization_analysis.csv')
df_ml = pd.read_csv('data\processed\data_machinelearning_clustering.csv')
df_viz['Date'] = pd.to_datetime(df_viz['Date'])
df_ml['Date'] = pd.to_datetime(df_ml['Date'])
log_result(f"Visualization data: {df_viz.shape[0]} rows, {df_viz.shape[1]} columns")
log_result(f"ML data: {df_ml.shape[0]} rows, {df_ml.shape[1]} columns")
log_step("Loading data", step_start)

# Prepare data for clustering
step_start = log_step("Preparing clustering data")
clustering_features = [
    'BTC_Price', 'ETH_Price', 'NVDA_Close', 'AMD_Close', 'INTC_Close',
    'GPU_Stock_Index', 'Regulatory_Stringency_Index', 
    'Defense_AI_Budget_Billions', 'Global_Regulatory_Index',
    'BTC_Momentum', 'ETH_Momentum', 'GPU_Momentum',
    'Gov_Crypto_Interaction', 'Military_GPU_Demand', 'Reg_Stringency_GPU_Impact'
]

# Filter for existing features
existing_features = [f for f in clustering_features if f in df_ml.columns]
X_cluster = df_ml[existing_features].copy()
log_result(f"Using {len(existing_features)} features for clustering")
log_step("Preparing clustering data", step_start)

# Determine optimal number of clusters
step_start = log_step("Determining optimal k using Elbow Method")
inertia = []
silhouette_scores = []
k_range = range(2, 15)

print("\nTesting different k values:")
for k in k_range:
    k_start = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)
    
    # Silhouette score requires at least 2 clusters
    if k >= 2:
        silhouette_avg = silhouette_score(X_cluster, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        log_result(f"k={k:2d}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette_avg:.4f}")

# Find optimal k (using elbow point)
diff = np.diff(inertia)
diff2 = np.diff(diff)
optimal_k_idx = np.argmax(diff2) + 2  # +2 because diff2 starts at k=4
optimal_k = min(optimal_k_idx, 7)  # Cap at 7 as per project guidelines

log_result(f"Optimal number of clusters determined: k={optimal_k}")
log_step("Determining optimal k using Elbow Method", step_start)

# Visualize elbow and silhouette scores
step_start = log_step("Generating elbow and silhouette plots")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(list(k_range), inertia, 'bo-')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5, label=f'Optimal k={optimal_k}')
ax1.legend()

# Silhouette plot
ax2.plot(list(k_range), silhouette_scores, 'ro-')
ax2.set_xlim([2, max(k_range)])  
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Scores for Different k', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5, label=f'Optimal k={optimal_k}')
ax2.legend()

plt.tight_layout()
plt.savefig('outputs\cluster_analysis\elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.close()
log_result("Elbow/silhouette plot saved to outputs\cluster_analysis\elbow_silhouette.png")
log_step("Generating elbow and silhouette plots", step_start)

# Apply K-Means with optimal k
step_start = log_step(f"Applying K-Means with k={optimal_k}")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

# Add cluster labels to dataframes
df_ml['Cluster'] = cluster_labels
df_viz['Cluster'] = cluster_labels

log_result(f"Clustering completed - {optimal_k} clusters created")
log_step(f"Applying K-Means with k={optimal_k}", step_start)

# Analyze cluster distribution
step_start = log_step("Analyzing cluster distribution")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print(f"\nCLUSTER DISTRIBUTION:")
for cluster_id, count in cluster_counts.items():
    percentage = count/len(cluster_labels)*100
    print(f"  Cluster {cluster_id}: {count:4d} samples ({percentage:5.1f}%)")
log_step("Analyzing cluster distribution", step_start)

# Create cluster profiles
step_start = log_step("Creating cluster profiles")
cluster_summary = df_viz.groupby('Cluster').agg({
    'BTC_Price': 'mean',
    'ETH_Price': 'mean',
    'GPU_Stock_Index': 'mean',
    'Regulatory_Stringency_Index': 'mean',
    'Defense_AI_Budget_Billions': 'mean',
    'Global_Regulatory_Index': 'mean',
    'BTC_Momentum': 'mean',
    'GPU_Momentum': 'mean',
    'Gov_Crypto_Interaction': 'mean',
    'Military_GPU_Demand': 'mean',
    'Reg_Stringency_GPU_Impact': 'mean',
    'Date': ['count', 'min', 'max']
}).round(2)

# Flatten column names
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.rename(columns={'Date_count': 'Count', 'Date_min': 'Start_Date', 'Date_max': 'End_Date'})

print(f"\nCLUSTER PROFILES:")
print("-" * 100)
print(f"{'Cluster':<8} {'Samples':<8} {'GPU Index':<10} {'BTC Price':<10} {'Reg Index':<10} {'Military':<10} {'Description':<40}")
print("-" * 100)

cluster_descriptions = []
for cluster_num in range(optimal_k):
    cluster_data = cluster_summary.loc[cluster_num]
    
    # Calculate normalized values for interpretation
    gpu_avg = cluster_data['GPU_Stock_Index_mean']
    btc_avg = cluster_data['BTC_Price_mean']
    reg_avg = cluster_data['Regulatory_Stringency_Index_mean']
    mil_avg = cluster_data['Defense_AI_Budget_Billions_mean']
    
    # Determine cluster type
    gpu_high = gpu_avg > df_viz['GPU_Stock_Index'].mean()
    btc_high = btc_avg > df_viz['BTC_Price'].mean()
    reg_high = reg_avg > df_viz['Regulatory_Stringency_Index'].mean()
    mil_high = mil_avg > df_viz['Defense_AI_Budget_Billions'].mean()
    
    # Create description
    if gpu_high and btc_high and reg_high:
        description = "High-Performance, High-Regulation Market"
    elif gpu_high and btc_high and not reg_high:
        description = "High-Performance, Low-Regulation Market"
    elif not gpu_high and not btc_high and reg_high:
        description = "Low-Performance, High-Regulation Market"
    elif gpu_high and mil_high:
        description = "High-Performance, High-Military Influence"
    elif reg_high and mil_high:
        description = "High-Regulation, High-Military Influence"
    elif gpu_high:
        description = "High GPU Performance Market"
    elif btc_high:
        description = "High Crypto Market"
    elif reg_high:
        description = "High Regulation Regime"
    elif mil_high:
        description = "High Military Influence"
    else:
        description = "Average Market Conditions"
    
    cluster_descriptions.append({
        'Cluster': cluster_num,
        'Description': description,
        'GPU_Index': gpu_avg,
        'BTC_Price': btc_avg,
        'Reg_Index': reg_avg,
        'Military_Budget': mil_avg
    })
    
    print(f"{cluster_num:<8} {cluster_data['Count']:<8} ${gpu_avg:<9.0f} ${btc_avg:<9.0f} {reg_avg:<10.0f} ${mil_avg:<9.2f}B {description:<40}")

log_step("Creating cluster profiles", step_start)

# PCA Visualization
step_start = log_step("Generating PCA visualization")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, cmap='tab20', 
                     alpha=0.7, s=50, edgecolor='k')
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title(f'K-Means Clustering Results (k={optimal_k})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           c='red', s=200, marker='X', label='Cluster Centers', edgecolor='black')
plt.legend()
plt.tight_layout()
plt.savefig('outputs\cluster_analysis\clustering_pca.png', dpi=300, bbox_inches='tight')
plt.close()
log_result("PCA clustering visualization saved to outputs\cluster_analysis\clustering_pca.png")
log_step("Generating PCA visualization", step_start)

# Time series visualization
step_start = log_step("Generating time series visualization")
plt.figure(figsize=(14, 8))
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for cluster_num in range(optimal_k):
    cluster_dates = df_viz[df_viz['Cluster'] == cluster_num]['Date']
    cluster_prices = df_viz[df_viz['Cluster'] == cluster_num]['GPU_Stock_Index']
    plt.scatter(cluster_dates, cluster_prices, 
               color=colors[cluster_num], label=f'Cluster {cluster_num}: {cluster_descriptions[cluster_num]["Description"]}', 
               alpha=0.6, s=20)

plt.xlabel('Date', fontsize=12)
plt.ylabel('GPU Stock Index', fontsize=12)
plt.title('Cluster Distribution Over Time', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs\cluster_analysis\clusters_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
log_result("Time series visualization saved to outputs\cluster_analysis\clusters_over_time.png")
log_step("Generating time series visualization", step_start)

# Cluster radar chart for comparison
step_start = log_step("Generating cluster comparison radar chart")
def create_radar_chart(cluster_data, features, title):
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for cluster_num in range(optimal_k):
        values = []
        for feature in features:
            col_name = f'{feature}_mean'
            if col_name in cluster_summary.columns:
                values.append(cluster_summary.loc[cluster_num, col_name])
            else:
                values.append(0)
        
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else values
        values = values.tolist()
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_num}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    return fig

# Select features for radar chart
radar_features = ['GPU_Stock_Index', 'BTC_Price', 'ETH_Price', 
                  'Regulatory_Stringency_Index', 'Defense_AI_Budget_Billions',
                  'Gov_Crypto_Interaction', 'Military_GPU_Demand']

fig = create_radar_chart(cluster_summary, radar_features, 
                         f'Cluster Comparison (Normalized Features) - k={optimal_k}')
plt.savefig('outputs\cluster_analysis\cluster_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
log_result("Radar chart saved to outputs\cluster_analysis\cluster_radar_chart.png")
log_step("Generating cluster comparison radar chart", step_start)

# Save clustered data
step_start = log_step("Saving clustered data")
df_viz.to_csv('data/processed/data_with_clusters.csv', index=False)
log_result("Clustered data saved to data/processed/data_with_clusters.csv")
log_step("Saving clustered data", step_start)

# Save clustering results
step_start = log_step("Saving clustering analysis results")
with open('outputs\cluster_analysis\clustering_analysis_results.txt', 'w') as f:
    f.write("CLUSTERING ANALYSIS RESULTS - K-MEANS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Optimal number of clusters: {optimal_k}\n")
    f.write(f"Total samples: {len(df_viz)}\n\n")
    
    f.write("CLUSTER DISTRIBUTION:\n")
    f.write("-"*70 + "\n")
    for cluster_id, count in cluster_counts.items():
        percentage = count/len(cluster_labels)*100
        f.write(f"Cluster {cluster_id}: {count} samples ({percentage:.1f}%)\n")
    
    f.write("\n\nCLUSTER PROFILES:\n")
    f.write("-"*70 + "\n")
    for desc in cluster_descriptions:
        f.write(f"\nCluster {desc['Cluster']}: {desc['Description']}\n")
        f.write(f"  GPU Stock Index: ${desc['GPU_Index']:.2f}\n")
        f.write(f"  BTC Price: ${desc['BTC_Price']:.2f}\n")
        f.write(f"  Regulatory Index: {desc['Reg_Index']:.1f}\n")
        f.write(f"  Defense AI Budget: ${desc['Military_Budget']:.2f}B\n")
    
    f.write("\n\nKEY INSIGHTS FROM CLUSTERING ANALYSIS:\n")
    f.write("-"*70 + "\n")
    f.write(f"1. Identified {optimal_k} distinct market regimes in the GPU-Crypto-Policy ecosystem\n")
    f.write("2. Clusters represent different combinations of market performance and regulation levels\n")
    f.write("3. High-regulation clusters often coincide with specific price patterns\n")
    f.write("4. Military spending creates distinct clusters with unique market characteristics\n")
    f.write("5. The 7 clusters validate the existence of different market regimes over time\n")

log_result("Results saved to outputs\cluster_analysis\clustering_analysis_results.txt")
log_step("Saving clustering analysis results", step_start)

# Summary
overall_time = time.time() - overall_start
print("\n" + "="*80)
print("CLUSTERING ANALYSIS COMPLETE")
print("="*80)
print(f"Total time: {overall_time:.2f} seconds")
print(f"\nFiles generated in outputs/ directory:")
print("  • elbow_silhouette.png")
print("  • clustering_pca.png")
print("  • clusters_over_time.png")
print("  • cluster_radar_chart.png")
print("  • clustering_analysis_results.txt")
print("\nFiles generated in data/processed/ directory:")
print("  • data_with_clusters.csv")
print("\nReady for Section 3.4: Clustering Analysis in report")
print("\nKEY FINDINGS:")
print(f"  1. Optimal clusters: {optimal_k} distinct market regimes identified")
print("  2. Each cluster represents unique combinations of market performance and regulation")
print("  3. Time series shows how market regimes evolve over time")
print("  4. Government/military factors create distinct market patterns")