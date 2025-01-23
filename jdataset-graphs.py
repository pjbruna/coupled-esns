import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.datasets import japanese_vowels

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

X_train = X_train[:90]
Y_train = Y_train[:90]
X_test = X_test[:154]
Y_test = Y_test[:154]

# 30 test exemplars per class (same as training)

del X_test[30]
del Y_test[30]

del X_test[60:65]
del Y_test[60:65]

del X_test[90:]
del Y_test[90:]

xlist = [len(item) for item in X_test]
ylist = [len(item) for item in Y_test]

if xlist == ylist:
    print("lists are identical")
else:
    [i for i, (a, b) in enumerate(zip(xlist, ylist)) if a != b]

# 3-bit one-hot encodings

for i in range(len(Y_train)):
    Y_train[i] = Y_train[i][:,:3]

for i in range(len(Y_test)):
    Y_test[i] = Y_test[i][:,:3]


## PLOT ##

# Train signals

plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Train: Class A', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_train[i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Train: Class B', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_train[30 + i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Train: Class C', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_train[60 + i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# Test signals

plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Test: Class A', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_test[i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Test: Class B', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_test[30 + i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


plt.figure(figsize=(12, 10))
fig, axes = plt.subplots(5, 6, figsize=(12, 10)) 
fig.suptitle('Test: Class C', fontsize=16, y=0.95)

for i in range(30):
    ax = axes[i // 6, i % 6]
    ax.imshow(X_test[60 + i].T, vmin=-1.2, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"Plot {i + 1}", fontsize=8)

# Adjust layout to avoid overlap with the main title
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()