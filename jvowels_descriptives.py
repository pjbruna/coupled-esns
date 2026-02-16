import numpy as np
import matplotlib.pyplot as plt
from data_processing import *


# load with preprocessing

X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True, do_print=True) # zscore=True


# plot LPC values

train_concat = np.vstack(X_train)
test_concat = np.vstack(X_test)

n_lpc = train_concat.shape[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].boxplot([train_concat[:, i] for i in range(n_lpc)])
axes[0].set_title("train inputs")
axes[0].set_xlabel("LPC")
axes[0].set_ylabel("value")

axes[1].boxplot([test_concat[:, i] for i in range(n_lpc)])
axes[1].set_title("test inputs")
axes[1].set_xlabel("LPC")

plt.tight_layout()
plt.show()


# distribution

fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(n_lpc):
    axes[i].hist(train_concat[:, i], bins=50, alpha=0.6, label='Train', density=True)
    axes[i].hist(test_concat[:, i], bins=50, alpha=0.6, label='Test', density=True)
    axes[i].set_title(f'LPC {i+1}')
    axes[i].set_xlim(-2.0, 2.0)

# Hide unused subplots
for j in range(n_lpc, len(axes)):
    axes[j].axis('off')

axes[0].legend()
fig.text(0.5, 0.04, 'value', ha='center')
fig.text(0.04, 0.5, 'density', va='center', rotation='vertical')
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()