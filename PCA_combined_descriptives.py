import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# # Load data for first plot
# cum_var_df = pd.read_csv("data/savedata/avg_PCA_variance.csv")
# cum_var_df['coupling'] = cum_var_df['coupling'].astype(str).astype('category')
# 
# new_labels = {'0.0': 'Coupling=0.0', '0.5': 'Coupling=0.5', '1.0': 'Coupling=1.0'}
# cum_var_df['coupling'] = cum_var_df['coupling'].cat.rename_categories(new_labels)
# coupling_levels = cum_var_df['coupling'].cat.categories
# manual_colors = ['#6baed6', '#2171b5', '#08306b']
# colors = {level: manual_colors[i] for i, level in enumerate(coupling_levels)}
# 
# # Load data for second plot
# df_none = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.0.csv")
# df_loose = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.5.csv")
# df_tight = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=1.0.csv")
# 
# df_none['coupling'] = "Coupling=0.0"
# df_loose['coupling'] = "Coupling=0.5"
# df_tight['coupling'] = "Coupling=1.0"
# 
# mi_df = pd.concat([df_none, df_loose, df_tight], ignore_index=True)
# mi_df['coupling'] = pd.Categorical(
#     mi_df['coupling'],
#     categories=["Coupling=0.0", "Coupling=0.5", "Coupling=1.0"],
#     ordered=True
# )
# 
# # Create the figure and two subplots sharing x-axis
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# 
# # Plot 1: Cumulative Variance
# for coupling in coupling_levels:
#     group = cum_var_df[cum_var_df['coupling'] == coupling]
#     group = group[group['pc'] <= 50]
#     x_vals = group['pc'].values
#     y_means = group['avg_cumvar'].values
#     y_sems = group['se_cumvar'].values
#     color = colors[coupling]
# 
#     ax1.plot(x_vals, y_means, '-', color=color, label=str(coupling), linewidth=2)
#     ax1.fill_between(x_vals, y_means - y_sems, y_means + y_sems, color=color, alpha=0.2)
# 
# ax1.set_ylabel("Avg. Cumulative Proportion of Variance Explained")
# ax1.grid(True)
# 
# # Plot 2: Mutual Information
# for coupling in coupling_levels:
#     group = mi_df[mi_df['coupling'] == coupling]
#     x = group['pc'].values
#     y = group['avg_nmi'].values
#     yerr = group['se_nmi'].values
#     color = colors[coupling]
# 
#     ax2.plot(x, y, '-', color=color, label=str(coupling), linewidth=2)
#     ax2.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
# 
# ax2.set_xlabel("PC")
# ax2.set_ylabel(r"Avg. NMI($R_1$, $R_2$)")
# ax2.grid(True)
# 
# # Legend only once on the top plot
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, title=None, loc='upper right', bbox_to_anchor=(1, 0.9))
# 
# # Set x-limits for both plots
# ax1.set_xlim(1, 50)
# 
# plt.tight_layout()
# plt.savefig('plt_figs/PCA_combined_descriptives.png', dpi=300)
# plt.show()



# Load data for first plot
cum_var_df = pd.read_csv("data/savedata/avg_PCA_variance.csv")
cum_var_df['coupling'] = cum_var_df['coupling'].astype(str).astype('category')

new_labels = {'0.0': 'Coupling=0.0', '0.5': 'Coupling=0.5', '1.0': 'Coupling=1.0'}
cum_var_df['coupling'] = cum_var_df['coupling'].cat.rename_categories(new_labels)
coupling_levels = cum_var_df['coupling'].cat.categories
manual_colors = ['#9ecae1', '#2171b5', '#08306b']
colors = {level: manual_colors[i] for i, level in enumerate(coupling_levels)}

# Load data for second plot
df_none = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.0.csv")
df_loose = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=0.5.csv")
df_tight = pd.read_csv("data/savedata/avg_PCA_mutualinformation_cs=1.0.csv")

df_none['coupling'] = "Coupling=0.0"
df_loose['coupling'] = "Coupling=0.5"
df_tight['coupling'] = "Coupling=1.0"

mi_df = pd.concat([df_none, df_loose, df_tight], ignore_index=True)
mi_df['coupling'] = pd.Categorical(
    mi_df['coupling'],
    categories=["Coupling=0.0", "Coupling=0.5", "Coupling=1.0"],
    ordered=True
)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Secondary y-axis
ax2 = ax1.twinx()

# Plot 1: Mutual Information on left y-axis (solid lines, define legend here)
for coupling in coupling_levels:
    group = mi_df[mi_df['coupling'] == coupling]
    x = group['pc'].values
    y = group['avg_nmi'].values
    yerr = group['se_nmi'].values
    color = colors[coupling]

    ax1.plot(x, y, '-', color=color, label=str(coupling), linewidth=2)  # Labels here
    ax1.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

ax1.set_ylabel(r"Avg. NMI($R_1$, $R_2$)", color='black')
ax1.set_ylim(0, 1)

# Shared x-axis label
ax1.set_xlabel("PC")

# Plot 2: Cumulative Variance on right y-axis (dashed lines, no labels)
for coupling in coupling_levels:
    group = cum_var_df[cum_var_df['coupling'] == coupling]
    group = group[group['pc'] <= 50]
    x_vals = group['pc'].values
    y_means = group['avg_cumvar'].values
    y_sems = group['se_cumvar'].values
    color = colors[coupling]

    ax2.plot(x_vals, y_means, '--', color=color, linewidth=2)  # No label here
    ax2.fill_between(x_vals, y_means - y_sems, y_means + y_sems, color=color, alpha=0.2)

ax2.set_ylabel("Avg. Cumulative Variance Explained", color='black')
ax2.set_ylim(0, 1)
ax2.grid(False)
ax2.set_xlim(1, 50)

ax1.grid(True)

# Legends

legend_kwargs = dict(
    fontsize='small',
    borderaxespad=0.5,
    handlelength=2,
    frameon=True,
    fancybox=True,
    edgecolor='gray'
)

handles, labels = ax1.get_legend_handles_labels()
leg1 = ax1.legend(handles, labels, title=None, loc='upper left', bbox_to_anchor=(0.8, 0.8), **legend_kwargs)

ax1.add_artist(leg1)

line_style_handles = [
    mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='NMI'),
    mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Cumulative Var.'),
]

leg2 = ax1.legend(handles=line_style_handles, title=None, loc='upper left', bbox_to_anchor=(0.8, 0.9), **legend_kwargs)

plt.tight_layout()
plt.savefig('plt_figs/PCA_combined_dual_axis.png', dpi=300)

