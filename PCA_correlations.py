import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Process data

# coupling=1.0
# cor_records = []
# 
# for run in range(10):
#     print(f'Calculating simulation {run+1}/10...')
# 
#     r1 = pd.read_csv(f"data/savedata/r1_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
#     r2 = pd.read_csv(f"data/savedata/r2_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
# 
#     scaler = StandardScaler()
#     scaled_r1 = scaler.fit_transform(r1)
#     scaled_r2 = scaler.fit_transform(r2)
# 
#     pca_r1 = PCA(n_components=500).fit_transform(scaled_r1)
#     pca_r2 = PCA(n_components=500).fit_transform(scaled_r2)
# 
#     # Compute correlations between all PC pairs
#     for i in range(500):
#         vec1 = pca_r1[:, i]
#         for j in range(500):
#             vec2 = pca_r2[:, j]
#             cor_val = np.corrcoef(vec1, vec2)[0, 1]  # Pearson correlation
#             cor_records.append({'r1': i + 1, 'r2': j + 1, 'cor': cor_val, 'sim': run})
# 
# cor_df = pd.DataFrame(cor_records)
# 
# summary_df = cor_df.groupby(['r1', 'r2'], as_index=False).agg(
#     avg_cor=('cor', 'mean'),
#     se_cor=('cor', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
# )
# 
# summary_df.to_csv(f"data/savedata/avg_PCA_correlations_cs={coupling}.csv", index=False)


########################################################################################


# # Parameters
# coupling = 1.0
# n_components = 500
# n_runs = 10
# cor_records = []
# 
# # Step 1: Load and concatenate all r1 and r2 data for global PCA fitting
# print("Loading all data for global PCA fit...")
# all_data = []
# 
# for run in range(n_runs):
#     r1 = pd.read_csv(f"data/savedata/r1_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
#     r2 = pd.read_csv(f"data/savedata/r2_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
#     all_data.append(r1)
#     all_data.append(r2)
# 
# combined_data = pd.concat(all_data, axis=0)
# 
# # Step 2: Fit global scaler and PCA
# print("Fitting global scaler and PCA...")
# scaler = StandardScaler().fit(combined_data)
# scaled_data = scaler.transform(combined_data)
# 
# pca_model = PCA(n_components=n_components).fit(scaled_data)
# 
# # Step 3: Apply same PCA to each run and compute PC correlations
# for run in range(n_runs):
#     print(f"Processing run {run+1}/{n_runs}...")
# 
#     r1 = pd.read_csv(f"data/savedata/r1_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
#     r2 = pd.read_csv(f"data/savedata/r2_cs={coupling}_r1=500_r2=500_sample=0.5_sim={run}.csv")
# 
#     # Use global scaler
#     scaled_r1 = scaler.transform(r1)
#     scaled_r2 = scaler.transform(r2)
# 
#     # Project into shared PCA space
#     pca_r1 = pca_model.transform(scaled_r1)
#     pca_r2 = pca_model.transform(scaled_r2)
# 
#     # Compute correlations between all PC pairs
#     for i in range(n_components):
#         vec1 = pca_r1[:, i]
#         for j in range(n_components):
#             vec2 = pca_r2[:, j]
#             cor_val = np.corrcoef(vec1, vec2)[0, 1]
#             cor_records.append({'r1': i + 1, 'r2': j + 1, 'cor': cor_val, 'sim': run})
# 
# # Step 4: Aggregate correlations over runs
# cor_df = pd.DataFrame(cor_records)
# 
# summary_df = cor_df.groupby(['r1', 'r2'], as_index=False).agg(
#     avg_cor=('cor', 'mean'),
#     se_cor=('cor', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
# )
# 
# # Save results
# summary_df.to_csv(f"data/savedata/aligned_avg_PCA_correlations_cs={coupling}.csv", index=False)


########################################################################################


# Plot

# Load CSVs
cor_df_none = pd.read_csv("data/savedata/avg_PCA_correlations_cs=0.0.csv")
cor_df_loose = pd.read_csv("data/savedata/avg_PCA_correlations_cs=0.5.csv")
cor_df_tight = pd.read_csv("data/savedata/avg_PCA_correlations_cs=1.0.csv")

# Add coupling labels
cor_df_none["coupling"] = "Coupling=0.0"
cor_df_loose["coupling"] = "Coupling=0.5"
cor_df_tight["coupling"] = "Coupling=1.0"

# Combine into one DataFrame
cor_df = pd.concat([cor_df_none, cor_df_loose, cor_df_tight], ignore_index=True)
cor_df["coupling"] = pd.Categorical(
    cor_df["coupling"],
    categories=["Coupling=0.0", "Coupling=0.5", "Coupling=1.0"],
    ordered=True
)


cor_df = cor_df[(cor_df["r1"] <= 50) & (cor_df["r2"] <= 50)]

g = sns.FacetGrid(
    cor_df,
    col="coupling",
    col_order=["Coupling=0.0", "Coupling=0.5", "Coupling=1.0"],
    col_wrap=3,
    height=4,
    aspect=1,
    sharex=False,
    sharey=False
)

def plot_heatmap(data, colorbar=False, **kwargs):
    pivot = data.pivot(index="r2", columns="r1", values="avg_cor")
    pivot = pivot.sort_index(ascending=False)  # y axis bottom-up

    ax = plt.gca()
    sns.heatmap(
        pivot,
        cmap="viridis",
        square=True,
        xticklabels=10,   # Adjust tick frequency for 50 PCs
        yticklabels=10,
        vmin=-1,
        vmax=1,
        cbar=colorbar,
        ax=ax,
        **kwargs
    )

for i, ax in enumerate(g.axes.flat):
    subset = cor_df[cor_df["coupling"] == g.col_names[i]]
    colorbar = (i == len(g.axes.flat) - 1)  # Only last plot gets colorbar
    plt.sca(ax)
    plot_heatmap(subset, colorbar=colorbar)

# Remove default titles and add custom facet labels
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title("")
    ax.text(0.5, 1.05, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=12)

# Set ticks for first 50 PCs (every 10th tick for clarity)
tick_positions = np.arange(1, 51, 10)
tick_labels = [str(i) for i in tick_positions]

for ax in g.axes.flat:
    ax.set_xlabel(r"$R_1$")
    ax.set_ylabel(r"$R_2$")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels[::-1])  # reversed y-axis

plt.tight_layout()
plt.savefig("plt_figs/PCA_correlations_50PCs.png", dpi=300, bbox_inches='tight')
