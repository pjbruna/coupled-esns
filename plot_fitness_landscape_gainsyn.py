import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv('data/gainsyn_fitness_landscape_N=10.csv')

res_size = df['coupling_strength'].tolist()
train_size = df['inequality_strength'].tolist()
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
im = ax.imshow(heatmap, cmap='inferno', vmin=-0.5, vmax=0.05, origin='lower', aspect='auto')

# Set axis ticks and labels
ax.set_xticks(np.arange(len(res_size_unique)))
ax.set_yticks(np.arange(len(train_size_unique)))
ax.set_xticklabels(res_size_unique, fontsize=18)
ax.set_yticklabels([f"{v:.2f}" for v in train_size_unique], fontsize=18)

# Axis labels
ax.set_xlabel('Coupling Strength', fontsize=20)
ax.set_ylabel('Inequality Size', fontsize=20)

# Rotate x tick labels if needed
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$Gain_{emerg}$', fontsize=20)
cbar.ax.tick_params(labelsize=18)

# Adjust colorbar ticks and format
cbar.set_ticks(np.array([0.0 if t == -0.0 else t for t in np.arange(-0.5, 0.1, 0.05)]))
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig(f'plt_figs/2D_fitness_landscape_gainsyn.png', dpi=300)