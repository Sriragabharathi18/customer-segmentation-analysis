# ==============================================================================
# PROJECT: Customer Segmentation using K-Means Clustering
# LEVEL: Advanced Beginner / Intermediate Internship Project
# LIBRARIES: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# STEP 1: Synthetic Dataset Generation (Self-Contained)
# ------------------------------------------------------------------------------
print("Generating dataset...")
np.random.seed(42)

# Creating simulated customer profiles
n_customers = 200
customer_ids = np.arange(1, n_customers + 1)
ages = np.random.randint(18, 70, size=n_customers)

# Generating clear segments for demonstration purposes
income_low = np.random.normal(25, 5, size=50)
spending_high = np.random.normal(80, 5, size=50)

income_mid = np.random.normal(55, 10, size=100)
spending_mid = np.random.normal(50, 10, size=100)

income_high = np.random.normal(90, 8, size=50)
spending_low = np.random.normal(20, 5, size=50)

annual_incomes = np.concatenate([income_low, income_mid, income_high])
spending_scores = np.concatenate([spending_high, spending_mid, spending_low])

# Creating the final DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': ages,
    'AnnualIncome_k$': np.clip(annual_incomes, 15, 130).astype(int),
    'SpendingScore_1to100': np.clip(spending_scores, 1, 100).astype(int)
})

print("\n--- FIRST 5 ROWS OF CUSTOMER DATA ---")
print(df.head())

# ------------------------------------------------------------------------------
# VISUALIZATION 1: Distribution Plots (Age, Income, Spending Score)
# ------------------------------------------------------------------------------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['AnnualIncome_k$'], kde=True, color='green')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['SpendingScore_1to100'], kde=True, color='red')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.suptitle('Output 1: Feature Distributions', y=1.05, fontsize=14, fontweight='bold')
plt.show()

# ------------------------------------------------------------------------------
# VISUALIZATION 2: Raw Data Relationship (Scatter Plot)
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='AnnualIncome_k$', y='SpendingScore_1to100', color='purple', s=60)
plt.title('Output 2: Income vs Spending Score (Before Clustering)', fontsize=12, fontweight='bold')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------------------------------------------
# STEP 2: Feature Selection & Scaling
# ------------------------------------------------------------------------------
# We select Annual Income and Spending Score for primary clustering
features = df[['AnnualIncome_k$', 'SpendingScore_1to100']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ------------------------------------------------------------------------------
# VISUALIZATION 3: The Elbow Method (Finding Optimal Clusters)
# ------------------------------------------------------------------------------
wcss = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker='o', linestyle='--', color='darkblue', linewidth=2)
plt.title('Output 3: The Elbow Method to Find Optimal K', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(cluster_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------------------------------------------
# STEP 3: Applying K-Means Clustering (K=3 chosen based on data structure)
# ------------------------------------------------------------------------------
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_transform(scaled_features).astype(str)

# Extract unscaled centroids for plotting
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)

# ------------------------------------------------------------------------------
# VISUALIZATION 4: Final 2D Clusters with Centroids
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='AnnualIncome_k$', y='SpendingScore_1to100', hue='Cluster', 
                palette='Set1', s=70, edgecolor='black', alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='yellow', marker='*', 
            edgecolor='black', label='Centroids', zorder=5)
plt.title('Output 4: Final Customer Segments (K=3)', fontsize=12, fontweight='bold')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------------------------------------------
# VISUALIZATION 5: 3D Cluster Visualization (Age Included)
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = {'0': 'red', '1': 'blue', '2': 'green'}
for cluster_id in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster_id]
    ax.scatter(cluster_data['Age'], cluster_data['AnnualIncome_k$'], cluster_data['SpendingScore_1to100'],
               c=colors[cluster_id], label=f'Cluster {cluster_id}', s=50, edgecolor='k', alpha=0.7)

ax.set_title('Output 5: 3D View of Customer Segments', fontsize=12, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
plt.show()

# ------------------------------------------------------------------------------
# VISUALIZATION 6: Cluster Metrics Analysis (Bar Chart Comparison)
# ------------------------------------------------------------------------------
cluster_summary = df.groupby('Cluster')[['AnnualIncome_k$', 'SpendingScore_1to100']].mean().reset_index()

plt.figure(figsize=(10, 5))
df_melted = pd.melt(cluster_summary, id_vars='Cluster', var_name='Metric', value_name='Average Value')
sns.barplot(data=df_melted, x='Cluster', y='Average Value', hue='Metric', palette='muted')
plt.title('Output 6: Average Metrics per Customer Cluster', fontsize=12, fontweight='bold')
plt.xlabel('Cluster Segment')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------------------------------------------
# STEP 4: Live Prediction for a New Customer
# ------------------------------------------------------------------------------
print("\n--- TEST PREDICTION SYSTEM ---")
new_income = 85  # k$
new_spending = 15  # Score out of 100

new_data_scaled = scaler.transform([[new_income, new_spending]])
predicted_cluster = kmeans_final.predict(new_data_scaled)[0]

print(f"A customer making ${new_income}k with a spending score of {new_spending} belongs to:")
print(f"👉 CLUSTER {predicted_cluster}")
