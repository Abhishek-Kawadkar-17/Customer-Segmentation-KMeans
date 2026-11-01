# Customer Segmentation using K-Means

## Overview
This beginner-friendly machine learning project groups customers according to annual income and spending score using K-Means clustering.

## Included dataset
A synthetic educational dataset containing 200 sample customer records is bundled in `data/Mall_Customers.csv`, so the project does not require a separate dataset download. See `DATASET_NOTE.txt`. If your instructor requires the original Kaggle dataset, replace the bundled CSV with the original file using the same column names.

## Features used
- Annual Income (k$)
- Spending Score (1-100)

## Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

## Run on Windows
1. Extract the ZIP.
2. Install Python 3.9 or later if needed.
3. Double-click `RUN_PROJECT.bat`. It installs dependencies and runs the program. An internet connection is needed for the first-time library installation.

## Run from VS Code terminal
```bash
python -m pip install -r requirements.txt
python customer_segmentation.py
```

## Generated files
- `outputs/01_elbow_method.png`
- `outputs/02_customer_segments.png`
- `outputs/03_cluster_comparison.png`
- `results/customer_segments.csv`
- `results/cluster_profile.csv`
- `results/model_summary.txt`

## Methodology
1. Load and inspect the customer data.
2. Select annual income and spending score.
3. Standardize the features.
4. Plot the Elbow Method (K=1 to 10).
5. Fit K-Means with K=5.
6. Evaluate with silhouette score.
7. Visualize and save customer segments and cluster profiles.

## Note
K=5 is used as a common teaching choice for the classic Mall Customers exercise. The Elbow graph is provided so you can discuss whether this is a reasonable choice for the data.
