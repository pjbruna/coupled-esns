import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

class CesnModel:
    def __init__(self, nnodes=None, coupling_strength=None):

        self.coupling_strength = coupling_strength
        self.reservoir1 = Reservoir(units=nnodes, sr=0.9, lr=0.1, activation='tanh') # seed=42
        self.reservoir2 = Reservoir(units=nnodes, sr=0.9, lr=0.1, activation='tanh') # seed=42
        self.readout1 = Ridge(ridge=1e-6)
        self.readout2 = Ridge(ridge=1e-6)


    def train(self, input=None, target=None):
        train_targets = []
        train_states1 = []
        train_states2 = []

        for (x, y) in zip(input, target):
            # final = len(x)-1

            for t in range(len(x)):
                if t==0:
                    input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
                else:
                    input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)

                rstate1 = self.reservoir1.run(input_fb)
                rstate2 = self.reservoir2.run(input_fb)

                train_states1.append(rstate1)
                train_states2.append(rstate2)
                train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            self.reservoir1.reset()
            self.reservoir2.reset()

        self.readout1.fit(train_states1, train_targets)
        self.readout2.fit(train_states2, train_targets)

        return


    def test(self, input=None, target=None, noise=None, print=False, save_reservoir=False):
        Y_pred1 = []
        Y_pred2 = []
        R_states1 = []
        R_states2 = []

        for x in input:
            y1 = np.array([np.zeros(self.readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(self.readout2.output_dim)] * len(x))

            if save_reservoir==True:
                r1 = np.array([np.zeros(self.reservoir1.output_dim)] * len(x))
                r2 = np.array([np.zeros(self.reservoir2.output_dim)] * len(x))

            for t in range(len(x)):
                if t==0:
                    # np.random.seed(42)
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise)), np.array(np.zeros(self.readout1.output_dim))), axis=None)
                    # np.random.seed(42)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise)), np.array(np.zeros(self.readout2.output_dim))), axis=None)
                else:
                    # np.random.seed(42)
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[(1-self.coupling_strength), self.coupling_strength])), axis=None)
                    # np.random.seed(42)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[self.coupling_strength, (1-self.coupling_strength)])), axis=None)

                rstate1 = self.reservoir1.run(input_fb1)
                rstate2 = self.reservoir2.run(input_fb2)

                ypred1 = self.readout1.run(rstate1)
                ypred2 = self.readout2.run(rstate2)

                # store
                y1[t] = ypred1
                y2[t] = ypred2

                if save_reservoir==True:
                    r1[t] = rstate1
                    r2[t] = rstate2

            # store
            Y_pred1.append(y1)
            Y_pred2.append(y2)

            if save_reservoir==True:
                R_states1.append(r1)
                R_states2.append(r2)

            # reset reservoirs
            self.reservoir1.reset()
            self.reservoir2.reset()

        # indiv rmse
        if print==True:
            persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred1)]
            persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred2)]

            print(f'CS: {self.coupling_strength}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')

        return Y_pred1, Y_pred2, R_states1, R_states2
    

    def accuracy(self, pred1=None, pred2=None, target=None, print=False):
        acc1 = [np.argmax(test, axis=1) == np.argmax(pred1, axis=1) for (test, pred1) in zip(target, pred1)]
        acc2 = [np.argmax(test, axis=1) == np.argmax(pred2, axis=1) for (test, pred2) in zip(target, pred2)]
        acc_joint = [np.argmax(test, axis=1) == np.argmax(np.mean((pred1, pred2), axis=0), axis=1) for (test, pred1, pred2) in zip(target, pred1, pred2)]

        if print==True:
            print(f'CS: {self.coupling_strength}; Acc1: {np.mean(np.concatenate(acc1))}; Acc2: {np.mean(np.concatenate(acc2))}; Joint: {np.mean(np.concatenate(acc_joint))}')

        return np.mean(np.concatenate(acc_joint))
