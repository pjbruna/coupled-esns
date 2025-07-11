import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# Helper functions

def compute_rdm(X, metric='correlation'):
    return squareform(pdist(X, metric=metric))

def rsa_between_rdms(rdm1, rdm2):
    # Use upper triangle (excluding diagonal) to flatten
    idx = np.triu_indices(rdm1.shape[0], k=1)
    vec1 = rdm1[idx]
    vec2 = rdm2[idx]
    return spearmanr(vec1, vec2).correlation

def plot_rdms(rdm1, rdm2, title1=r"$R_1$", title2=r"$R_2$", save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))  # 1 row, 2 cols
    
    sns.heatmap(rdm1, cmap='inferno', square=True, cbar=False, ax=axs[0])
    axs[0].set_title(title1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    sns.heatmap(rdm2, cmap='inferno', square=True, cbar=True, ax=axs[1], cbar_kws={'label': 'Dissimilarity'})
    axs[1].set_title(title2)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    colorbar = axs[1].collections[0].colorbar
    colorbar.set_ticks(np.linspace(0, 1, num=6))
    colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(0, 1, num=6)])

    # left_ax = axs[0].get_position()
    # left_x = left_ax.x0  # x0 is the left edge in figure coords

    # fig.suptitle(fr"$\it{{Coupling={coupling}}}$", fontsize=16, x=left_x, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


#######################################################################

# Condition

coupling = 1.0

# Calculate over 10 runs

rsa_scores = []
all_rdms_1 = []
all_rdms_2 = []

for r in range(10):
    print(f'Calculating simulation {r+1}/10...')

    r1 = pd.read_csv(f'data/savedata/r1_cs={coupling}_r1=500_r2=500_sample=0.5_sim={r}.csv').to_numpy()
    r2 = pd.read_csv(f'data/savedata/r2_cs={coupling}_r1=500_r2=500_sample=0.5_sim={r}.csv').to_numpy()

    rdm_1 = compute_rdm(r1, metric='correlation')
    rdm_2 = compute_rdm(r2, metric='correlation')

    rsa_score = rsa_between_rdms(rdm_1, rdm_2)
    rsa_scores.append(rsa_score)

    all_rdms_1.append(rdm_1)
    all_rdms_2.append(rdm_2)

mean_rsa = np.mean(rsa_scores)
std_rsa = np.std(rsa_scores)

print(f'Mean RSA score (coupling={coupling}): {mean_rsa:.3f} ± {std_rsa:.3f}')


# Plot RDMs

avg_rdm_1 = np.mean(np.stack(all_rdms_1), axis=0)
avg_rdm_2 = np.mean(np.stack(all_rdms_2), axis=0)

plot_rdms(avg_rdm_1, avg_rdm_2, save_path=f'plt_figs/RDMs_cs={coupling}.png')


# Plot RDM difference

rdm_diff = avg_rdm_1 - avg_rdm_2

plt.figure(figsize=(6, 5))
ax = sns.heatmap(rdm_diff, cmap='inferno', center=0, square=True,
                 vmin=0, vmax=1,
                 cbar_kws={'label': 'Δ Dissimilarity'})

# Access the colorbar properly
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.linspace(0, 1, num=6))
colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(0, 1, num=6)])

plt.title(r"$R_1 - R_2$")
plt.xticks([])
plt.yticks([])
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig(f'plt_figs/RDMs_diff_cs={coupling}.png', dpi=300)