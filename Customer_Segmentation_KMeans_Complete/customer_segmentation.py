"""Customer Segmentation using K-Means.
Run with: python customer_segmentation.py
The included dataset is a synthetic educational demo dataset. See DATASET_NOTE.txt.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Mall_Customers.csv"
OUT = ROOT / "outputs"
RES = ROOT / "results"
OUT.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(DATA)
print("\n--- FIRST FIVE CUSTOMERS ---")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Select annual income and spending score for segmentation
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].dropna()
if len(X) < 5:
    raise ValueError("The dataset must contain at least five complete customer records.")

# 3. Standardize features so both variables contribute fairly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow method: calculate within-cluster sum of squares for K=1..10
wcss = []
k_values = range(1, 11)
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), wcss, marker="o")
plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Within-cluster sum of squares (WCSS)")
plt.xticks(list(k_values))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "01_elbow_method.png", dpi=200)
plt.close()

# 5. Fit K-Means with five customer segments
# Five is a common teaching choice for the classic Mall Customers exercise.
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
result = df.loc[X.index].copy()
result["Cluster"] = labels

# 6. Evaluate the grouping
silhouette = silhouette_score(X_scaled, labels)
print("\nSilhouette score:", round(silhouette, 3))
print("\nCustomers in each cluster:\n", result["Cluster"].value_counts().sort_index())

# 7. Create a cluster profile table
profile = result.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Annual_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean"),
    Average_Age=("Age", "mean")
).round(2)
profile.to_csv(RES / "cluster_profile.csv")
print("\nCluster profile:\n", profile)

# 8. Plot customer segments
plt.figure(figsize=(9, 6))
sns.scatterplot(data=result, x=features[0], y=features[1], hue="Cluster", palette="tab10", s=75)
plt.title("Customer Segments using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(OUT / "02_customer_segments.png", dpi=200)
plt.close()

# 9. Plot average values for each cluster
profile[["Average_Annual_Income", "Average_Spending_Score"]].plot(kind="bar", figsize=(9, 5))
plt.title("Average Income and Spending Score by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "03_cluster_comparison.png", dpi=200)
plt.close()

# 10. Save final labelled customer data
result.to_csv(RES / "customer_segments.csv", index=False)
with open(RES / "model_summary.txt", "w", encoding="utf-8") as f:
    f.write("CUSTOMER SEGMENTATION - MODEL SUMMARY\n")
    f.write(f"Dataset records used: {len(result)}\n")
    f.write(f"Number of clusters: {K}\n")
    f.write(f"Silhouette score: {silhouette:.3f}\n\n")
    f.write("Cluster profile:\n")
    f.write(profile.to_string())
print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Graphs saved in the outputs folder.")
print("CSV and summary saved in the results folder.")
