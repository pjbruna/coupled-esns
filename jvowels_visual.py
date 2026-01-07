import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.datasets import japanese_vowels

# Load dataset
X_train, Y_train, X_test, Y_test = japanese_vowels()

samples_per_class = 30
timesteps_to_plot = 10 
figsize = (8, 6)

# Loop through the 9 classes
for class_idx in range(9):
    sample_idx = class_idx * samples_per_class
    
    plt.figure(figsize=figsize)
    plt.imshow(X_train[sample_idx][:timesteps_to_plot, :].T, vmin=-1.2, vmax=2)
    speaker = np.argmax(Y_train[sample_idx]) + 1
    plt.title(f"speaker {speaker}")
    plt.xlabel("timesteps")
    plt.ylabel("LPC (cepstra)")
    # plt.colorbar()
    
    plt.savefig(f"speaker_figs/class_{class_idx + 1}.png")
    plt.close()
