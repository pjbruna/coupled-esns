import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from cesn_model import *

rpy.verbosity(0)

### DATA ###

sample = 1000
generate = 750
series = mackey_glass(n_timesteps=2000)

scaler = StandardScaler().fit(series[:sample])
z_series = scaler.transform(series)

X_train = z_series[:sample]
X_test = z_series[sample:sample+generate]
Y_train = z_series[1:sample+1]
Y_test = z_series[sample+1:sample+generate+1]


### RUN ###

simulations = {'coupled': [], 'independent': []}

for run in range(20):
    print(f'Running simulation #{run+1}...')

    model = CesnModel_V3(nnodes=[800,800], in_plink=[0.1,0.1])

    r1state_list = []
    r2state_list = []
    target_list = []

    for t in range(len(X_train)):
        r1state = model.reservoir1.run(X_train[t])
        r2state = model.reservoir2.run(X_train[t])

        if t > 9:
            r1state_list.append(r1state)
            r2state_list.append(r2state)
            target_list.append(Y_train[t, np.newaxis])

    # fit readout layer
    model.readout1.fit(r1state_list, target_list)
    model.readout2.fit(r2state_list, target_list)

    # run
    init_r1 = model.reservoir1.state()
    init_r2 = model.reservoir2.state()
    init_y1 = model.readout1.run(r1state_list[-1])
    init_y2 = model.readout2.run(r2state_list[-1])

    results = {'coupled': [], 'independent': []}

    for c in ('coupled', 'independent'):
        for t in range(len(X_test)):
            if t==0:
                model.reservoir1.reset(to_state=init_r1)
                model.reservoir2.reset(to_state=init_r2)
                pred1 = init_y1
                pred2 = init_y2

            if c=='coupled':
                x = np.mean((pred1, pred2), axis=0)
                rstate1 = model.reservoir1.run(x)
                rstate2 = model.reservoir2.run(x)

            else:
                rstate1 = model.reservoir1.run(pred1)
                rstate2 = model.reservoir2.run(pred2)

            pred1 = model.readout1(rstate1)
            pred2 = model.readout2(rstate2)

            results[c].append((pred1, pred2))
    
    for c in ('coupled', 'independent'):
        simulations[c].append(results[c])


### PLOT ###

def avg_sims(sim_results):
    sims = []
    cum_errs = []

    for sim in sim_results:
        y1, y2 = zip(*sim)
        y1 = np.array(y1).reshape(-1)
        y2 = np.array(y2).reshape(-1)

        ymean = np.mean((y1, y2), axis=0)
        sims.append(ymean)

        cum_errs.append(np.cumsum(np.abs(ytar - ymean)))

    sims = np.stack(sims, axis=0)        
    cum_errs = np.stack(cum_errs, axis=0)

    mean = np.mean(sims, axis=0)
    se = np.std(sims, axis=0, ddof=1) / np.sqrt(sims.shape[0])

    cum_mean = np.mean(cum_errs, axis=0)
    cum_se = np.std(cum_errs, axis=0, ddof=1) / np.sqrt(cum_errs.shape[0])

    return mean, se, cum_mean, cum_se


ytar = Y_test.reshape(-1)

yavg_c, yse_c, yerr_c, yerr_se_c = avg_sims(simulations['coupled'])
yavg_i, yse_i, yerr_i, yerr_se_i = avg_sims(simulations['independent'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t = np.arange(len(ytar))

# predictions
ax1.plot(ytar, label="target", color="blue")

ax1.plot(yavg_c, label="coupled", color="red")
ax1.fill_between(t, yavg_c - yse_c, yavg_c + yse_c, color="red", alpha=0.2)

ax1.plot(yavg_i, label="independent", color="orange")
ax1.fill_between(t, yavg_i - yse_i, yavg_i + yse_i, color="orange", alpha=0.2)

ax1.set_ylabel("mackey-glass (standardized)")
ax1.legend(loc='lower left')
ax1.grid(alpha=0.3)

# error
ax2.plot(yerr_c, label="coupled", color="red")
ax2.fill_between(
    t,
    yerr_c - yerr_se_c,
    yerr_c + yerr_se_c,
    color="red",
    alpha=0.2
)

ax2.plot(yerr_i, label="independent", color="orange")
ax2.fill_between(
    t,
    yerr_i - yerr_se_i,
    yerr_i + yerr_se_i,
    color="orange",
    alpha=0.2
)

ax2.set_xlabel("time step")
ax2.set_ylabel("cumulative error")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()