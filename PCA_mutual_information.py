import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Process data

# Parameters
coupling = 1.0
bins = 10 # bin size
num_pcs = 50 # number of PCs

mi_records = []

for run in range(10):
    r1 = pd.read_csv(f"data/unequal/r1_cs={coupling}_r1=500_r2=500_sample=0.5_inequality=0.2_sim={run}.csv")
    r2 = pd.read_csv(f"data/unequal/r2_cs={coupling}_r1=500_r2=500_sample=0.5_inequality=0.2_sim={run}.csv")
    
    scaler = StandardScaler()
    scaled_r1 = scaler.fit_transform(r1)
    scaled_r2 = scaler.fit_transform(r2)
    
    pca_r1 = PCA(n_components=num_pcs).fit_transform(scaled_r1)
    pca_r2 = PCA(n_components=num_pcs).fit_transform(scaled_r2)

    for i in range(num_pcs):
        x_pc = pca_r1[:, i]
        y_pc = pca_r2[:, i]

        r1_disc = pd.cut(x_pc, bins=bins, labels=False)
        r2_disc = pd.cut(y_pc, bins=bins, labels=False)

        h_r1 = entropy(np.bincount(r1_disc))
        h_r2 = entropy(np.bincount(r2_disc))

        # Mutual information
        mi = mutual_info_score(r1_disc, r2_disc)

        # Normalized mutual information
        nmi = mi / np.sqrt(h_r1 * h_r2) if h_r1 > 0 and h_r2 > 0 else 0

        # Store result
        mi_records.append({
            'pc': i + 1,
            'nmi': nmi,
            'sim': run
        })

mi_df = pd.DataFrame(mi_records)

# Group and summarize
summary_df = (
    mi_df
    .groupby('pc', as_index=False)
    .agg(
        avg_nmi=('nmi', 'mean'),
        se_nmi=('nmi', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    )
)

summary_df.to_csv(f'data/unequal/avg_PCA_mutualinformation_cs={coupling}.csv', index=False)

##############################################################################################

# # Plot
# 
# # Load and label data
# df_none = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.0.csv")
# df_loose = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.5.csv")
# df_tight = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=1.0.csv")
# 
# df_none['coupling'] = "Coupling=0.0"
# df_loose['coupling'] = "Coupling=0.5"
# df_tight['coupling'] = "Coupling=1.0"
# 
# # Combine and order factors
# mi_df = pd.concat([df_none, df_loose, df_tight], ignore_index=True)
# mi_df['coupling'] = pd.Categorical(
#     mi_df['coupling'],
#     categories=["Coupling=0.0", "Coupling=0.5", "Coupling=1.0"],
#     ordered=True
# )
# 
# # Set up color map
# coupling_levels = mi_df['coupling'].cat.categories
# manual_colors = ['#6baed6', '#2171b5', '#08306b']
# colors = {level: manual_colors[i] for i, level in enumerate(coupling_levels)}
# 
# plt.figure(figsize=(10, 6))
# 
# for coupling in coupling_levels:
#     group = mi_df[mi_df['coupling'] == coupling]
#     x = group['pc'].values
#     y = group['avg_nmi'].values
#     yerr = group['se_nmi'].values
# 
#     color = colors[coupling]
# 
#     plt.plot(x, y, '-', color=color, label=coupling, linewidth=2)
#     plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
# 
# 
# plt.xlabel("PC")
# plt.ylabel(r"Avg. NMI($R_1$, $R_2$)")
# plt.grid(True)
# plt.legend(title=None, loc='upper right')
# plt.tight_layout()
# 
# plt.savefig("plt_figs/PCA_mutualinformation.png", dpi=300)
