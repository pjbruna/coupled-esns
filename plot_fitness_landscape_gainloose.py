import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Load data
# 
# df = pd.read_csv('data/fitness_landscape_N=10_noise=0.csv')
# 
# res_size = df['res_size'].tolist()
# train_size = df['train_size'].tolist()
# measure = df['measure'].tolist()
# 
# 
# # Plot fitness landscape
# 
# res_size_unique = sorted(set(res_size))
# train_size_unique = sorted(set(train_size))
# 
# # Fill heatmap grid
# heatmap = np.full((len(train_size_unique), len(res_size_unique)), np.nan)
# for r, t, m in zip(res_size, train_size, measure):
#     i = train_size_unique.index(t)
#     j = res_size_unique.index(r)
#     heatmap[i, j] = m
# 
# # Meshgrid for surface plot
# X, Y = np.meshgrid(res_size_unique, train_size_unique)
# Z = np.nan_to_num(heatmap, nan=0.0)
# 
# # Plotting
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# 
# # Plot the surface
# surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# 
# # Axis labels (z-axis intentionally blank)
# ax.set_xlabel('Reservoir Size (nodes)')
# ax.set_ylabel('Training Size (%)')
# ax.set_zlabel('')  # Hide Z label
# 
# # Remove z-axis ticks and numbers
# ax.set_zticks([])
# ax.set_zticklabels([])
# 
# # Remove grid lines
# ax.grid(False)
# 
# # Colorbar
# fig.colorbar(surf, ax=ax, label='Gain')
# 
# plt.show()
# 
# # plt.tight_layout()


#######################################################################


# Load data
df = pd.read_csv('data/gainloose_fitness_landscape_N=10.csv')

res_size = df['res_size'].tolist()
train_size = df['train_size'].tolist()
measure = df['measure'].tolist()

# Unique sorted values
res_size_unique = sorted(set(res_size))
train_size_unique = sorted(set(train_size))

# Fill heatmap grid
heatmap = np.full((len(train_size_unique), len(res_size_unique)), np.nan)
for r, t, m in zip(res_size, train_size, measure):
    i = train_size_unique.index(t)
    j = res_size_unique.index(r)
    heatmap[i, j] = m

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
im = ax.imshow(heatmap, cmap='viridis', origin='lower', aspect='auto')

# Set axis ticks and labels
ax.set_xticks(np.arange(len(res_size_unique)))
ax.set_yticks(np.arange(len(train_size_unique)))
ax.set_xticklabels(res_size_unique, fontsize=18)
ax.set_yticklabels([f"{v:.2f}" for v in train_size_unique], fontsize=18)

# Axis labels
ax.set_xlabel('Reservoir Size', fontsize=20)
ax.set_ylabel('Training Size', fontsize=20)

# Rotate x tick labels if needed
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$Gain_{loose}$', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(f'plt_figs/2D_fitness_landscape.png')