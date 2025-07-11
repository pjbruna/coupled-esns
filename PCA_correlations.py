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

# Set up FacetGrid
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

# Plot heatmap in each facet
def plot_heatmap(data, colorbar=False, **kwargs):
    pivot = data.pivot(index="r2", columns="r1", values="avg_cor")
    pivot = pivot.sort_index(ascending=False)  # y axis bottom-up

    ax = plt.gca()
    sns.heatmap(
        pivot,
        cmap="viridis",
        square=True,
        xticklabels=100,
        yticklabels=100,
        vmin=-1,
        vmax=1,
        cbar=colorbar,
        ax=ax,
        **kwargs
    )

# Use FacetGrid to plot each heatmap
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

# Set identical ticks for x and y axes for all subplots
tick_positions = np.arange(1, 501, 100)  # ticks at 0,100,200,...500
tick_labels = [str(i) for i in tick_positions]

for ax in g.axes.flat:
    ax.set_xlabel(r"$R_1$")
    ax.set_ylabel(r"$R_2$")
    
    # Set ticks - note heatmap axis indices start at 0, so adjust accordingly
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels[::-1])  # reversed for bottom-up y-axis

plt.tight_layout()
plt.savefig("plt_figs/PCA_correlations.png", dpi=300, bbox_inches='tight')
