import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Process data

# cum_var_df_list = []
# 
# for run in range(10):
#     for strength in ["0.0", "0.5", "1.0"]:
#         r1 = pd.read_csv(f"data/savedata/r1_cs={strength}_r1=500_r2=500_sample=0.5_sim={run}.csv")
#         r2 = pd.read_csv(f"data/savedata/r2_cs={strength}_r1=500_r2=500_sample=0.5_sim={run}.csv")
# 
#         r_comb = pd.concat([r1, r2], axis=1)
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(r_comb)
# 
#         pca = PCA()
#         pca.fit(scaled_data)
# 
#         prop_var_explained = pca.explained_variance_ratio_
#         cum_var_explained = np.cumsum(prop_var_explained)
# 
#         temp_df = pd.DataFrame({
#             'pc': np.arange(1, len(cum_var_explained) + 1),
#             'cumvar': cum_var_explained,
#             'coupling': strength,
#             'sim': run
#         })
# 
#         cum_var_df_list.append(temp_df)
# 
# cum_var_df = pd.concat(cum_var_df_list, ignore_index=True)
# 
# summary_df = (
#     cum_var_df
#     .groupby(['pc', 'coupling'], as_index=False)
#     .agg(
#         avg_cumvar=('cumvar', 'mean'),
#         se_cumvar=('cumvar', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
#     )
# )
# 
# summary_df.to_csv("data/savedata/avg_PCA_variance.csv", index=False)


#################################################################

# Plot

# Load data
cum_var_df = pd.read_csv("data/savedata/avg_PCA_variance.csv")
cum_var_df['coupling'] = cum_var_df['coupling'].astype(str).astype('category')

# Setup color palette
new_labels = {'0.0': 'Coupling=0.0', '0.5': 'Coupling=0.5', '1.0': 'Coupling=1.0'}
cum_var_df['coupling'] = cum_var_df['coupling'].cat.rename_categories(new_labels)
coupling_levels = cum_var_df['coupling'].cat.categories
manual_colors = ['#2171b5', '#6baed6', '#bdd7e7']
colors = {level: manual_colors[i] for i, level in enumerate(coupling_levels)}

plt.figure(figsize=(10, 6))

for coupling in coupling_levels:
    group = cum_var_df[cum_var_df['coupling'] == coupling]
    group = group[group['pc'] <= 50]
    x_vals = group['pc'].values
    y_means = group['avg_cumvar'].values
    y_sems = group['se_cumvar'].values

    color = colors[coupling]

    plt.plot(x_vals, y_means, '-', color=color, label=str(coupling), linewidth=2)
    plt.fill_between(x_vals, y_means - y_sems, y_means + y_sems,
                     color=color, alpha=0.2)


plt.xlim(1, 50)
plt.xlabel("PC")
plt.ylabel("Avg. Cumulative Proportion of Variance Explained")
plt.grid(True)
plt.legend(title=None, loc='upper right', bbox_to_anchor=(1, 0.9))
plt.tight_layout()

plt.savefig(f'plt_figs/PCA_cumulative_explained_var.png', dpi=300)