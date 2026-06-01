import numpy as np
import reservoirpy as rpy
from data_processing import *
from cesn_model import *

rpy.verbosity(0)
np.random.seed(42)

# sample data
X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True)

# train model
model = CesnModel_Multi(ensemble_size=1, nnodes=25, in_plink=0.1, rc_plink=0.1, seed=42)
model.train(inputs=X_train, targets=Y_train, teacherfb_sigma=0.2, reset='zero')

# test
outputs = model.test(inputs=X_test, targets=Y_test, condition='polycentric', input_sigma=0.5, reset='zero')
results = model.accuracy(predictions=outputs, targets=Y_test)

print(results)
