import numpy as np
from scipy.special import softmax
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge


### FUNCTIONS ###

def generate_null_feedback(bits=None, normalize=None):
        x = np.random.uniform(-1, 1, bits)
        if normalize==True:
            x = x / np.sum(x)
        return x


def select_param(range=None, quant=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(range, size=quant, replace=True)


### MODEL ###

class CesnModel:
    def __init__(self, r1_nnodes=None, r2_nnodes=None, coupling_strength=None, is_seed=None):

        self.coupling_strength = coupling_strength

        if is_seed==None:
            self.reservoir1 = Reservoir(units=r1_nnodes, sr=0.9, lr=0.1, activation='tanh')
            self.reservoir2 = Reservoir(units=r2_nnodes, sr=0.9, lr=0.1, activation='tanh')
        else:
            self.reservoir1 = Reservoir(units=r1_nnodes, sr=0.9, lr=0.1, activation='tanh', seed=is_seed[0])
            self.reservoir2 = Reservoir(units=r2_nnodes, sr=0.9, lr=0.1, activation='tanh', seed=is_seed[1])
            
        self.readout1 = Ridge(ridge=1e-6)
        self.readout2 = Ridge(ridge=1e-6)


    def train_r1(self, input=None, target=None, warmup=None, reset=None):
        train_states = []
        train_targets = []

        for (x, y) in zip(input, target):
            for t in range(len(x)):
                if t==0:
                    input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
                else:
                    input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)

                rstate = self.reservoir1.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))

            if reset=='zero':
                self.reservoir1.reset()

        self.readout1.fit(train_states, train_targets)

        return
    

    def train_r2(self, input=None, target=None, warmup=0, reset=None):
        train_states = []
        train_targets = []

        for (x, y) in zip(input, target):
            for t in range(len(x)):
                if t==0:
                    input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
                else:
                    input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)

                rstate = self.reservoir2.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='random':
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

            if reset=='zero':
                self.reservoir2.reset()

        self.readout2.fit(train_states, train_targets)

        return


    def test(self, input=None, target=None, noise_scale=0, reset=None, do_print=False, save_reservoir=False):
        Y_pred1 = []
        Y_pred2 = []
        R_states1 = []
        R_states2 = []

        for (x, y) in zip(input, target):
            y1 = np.array([np.zeros(self.readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(self.readout2.output_dim)] * len(x))

            if save_reservoir==True:
                r1 = np.array([np.zeros(self.reservoir1.output_dim)] * len(x))
                r2 = np.array([np.zeros(self.reservoir2.output_dim)] * len(x))

            for t in range(len(x)):
                if t==0:
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.array(np.zeros(self.readout1.output_dim))), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.array(np.zeros(self.readout2.output_dim))), axis=None)
                else:
                    # input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), y[t]), axis=None) # 'true' feedback
                    # input_fb2 = input_fb1

                    # softmax readouts
                    sftmx1 = softmax(self.readout1.state())
                    sftmx2 = softmax(self.readout2.state())

                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([sftmx1, sftmx2], axis=0, weights=[(1-self.coupling_strength), self.coupling_strength])), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([sftmx1, sftmx2], axis=0, weights=[self.coupling_strength, (1-self.coupling_strength)])), axis=None)

                    # input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[(1-self.coupling_strength), self.coupling_strength])), axis=None)
                    # input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[self.coupling_strength, (1-self.coupling_strength)])), axis=None)

                rstate1 = self.reservoir1.run(input_fb1)
                rstate2 = self.reservoir2.run(input_fb2)

                ypred1 = softmax(self.readout1.run(rstate1))
                ypred2 = softmax(self.readout2.run(rstate2))

                # ypred1 = self.readout1.run(rstate1)
                # ypred2 = self.readout2.run(rstate2)

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
            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

            if reset=='zero':
                self.reservoir1.reset()
                self.reservoir2.reset()

        # indiv rmse
        if do_print==True:
            persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred1)]
            persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred2)]

            print(f'CS: {self.coupling_strength}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')

        return Y_pred1, Y_pred2, R_states1, R_states2


    def test_null(self, input=None, target=None, noise_scale=None, do_print=False, save_reservoir=False):
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
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.array(np.zeros(self.readout1.output_dim))), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.array(np.zeros(self.readout2.output_dim))), axis=None)
                else:
                    null1 = generate_null_feedback(bits=self.readout1.output_dim, normalize=True)
                    null2 = generate_null_feedback(bits=self.readout2.output_dim, normalize=True)

                    null1 = null1.reshape(self.readout1.state().shape)
                    null2 = null2.reshape(self.readout2.state().shape)

                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([self.readout1.state(), null2], axis=0, weights=[(1-self.coupling_strength), self.coupling_strength])), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale)), np.average([null1, self.readout2.state()], axis=0, weights=[self.coupling_strength, (1-self.coupling_strength)])), axis=None)

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
        if do_print==True:
            persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred1)]
            persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred2)]

            print(f'CS: {self.coupling_strength}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')

        return Y_pred1, Y_pred2, R_states1, R_states2
    

    def test_unequal(self, input=None, target=None, noise_scale_1=None, noise_scale_2=None, do_print=False, save_reservoir=False):
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
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale_1)), np.array(np.zeros(self.readout1.output_dim))), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale_2)), np.array(np.zeros(self.readout2.output_dim))), axis=None)
                else:
                    input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale_1)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[(1-self.coupling_strength), self.coupling_strength])), axis=None)
                    input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * noise_scale_2)), np.average([self.readout1.state(), self.readout2.state()], axis=0, weights=[self.coupling_strength, (1-self.coupling_strength)])), axis=None)

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
        if do_print==True:
            persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred1)]
            persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(target, Y_pred2)]

            print(f'CS: {self.coupling_strength}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')

        return Y_pred1, Y_pred2, R_states1, R_states2


    def accuracy(self, pred1=None, pred2=None, target=None):
        acc1 = [np.argmax(test, axis=1) == np.argmax(pred1, axis=1) for (test, pred1) in zip(target, pred1)]
        acc2 = [np.argmax(test, axis=1) == np.argmax(pred2, axis=1) for (test, pred2) in zip(target, pred2)]
        acc_joint = [np.argmax(test, axis=1) == np.argmax(np.mean((pred1, pred2), axis=0), axis=1) for (test, pred1, pred2) in zip(target, pred1, pred2)]

        return np.mean(np.concatenate(acc_joint)), np.mean(np.concatenate(acc1)), np.mean(np.concatenate(acc2))


    def weighted_accuracy(self, pred1=None, pred2=None, target=None):
        acc_list = []

        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            acc_joint = [np.argmax(test, axis=1) == np.argmax(np.average((pred1, pred2), axis=0, weights=[w,1-w]), axis=1) for (test, pred1, pred2) in zip(target, pred1, pred2)]
            acc_list.append(np.mean(np.concatenate(acc_joint)))

        return acc_list
    


#############################################



class CesnModel_V2:
    def __init__(self, nnodes=None, in_plink=None, amp=[1,1], seed=None):
        # amplitude of feedback channels
        self.amp_a = amp[0] # auto
        self.amp_b = amp[1] # allo

        # create reservoirs
        if seed==None:
            self.reservoir1 = Reservoir(units=nnodes[0], input_connectivity=in_plink[0], sr=0.9, lr=0.1, activation='tanh')
            self.reservoir2 = Reservoir(units=nnodes[1], input_connectivity=in_plink[1], sr=0.9, lr=0.1, activation='tanh')
        else:
            self.reservoir1 = Reservoir(units=nnodes[0], input_connectivity=in_plink[0], sr=0.9, lr=0.1, activation='tanh', seed=seed[0])
            self.reservoir2 = Reservoir(units=nnodes[1], input_connectivity=in_plink[1], sr=0.9, lr=0.1, activation='tanh', seed=seed[1])
            
        # create readouts
        self.readout1 = Ridge(ridge=1e-6)
        self.readout2 = Ridge(ridge=1e-6)


    def train_r1(self, input=None, target=None, teacherfb_sigma=0.2, warmup=0, reset='zero'):
        train_states = []
        train_targets = []
        for (x, y) in zip(input, target):
            for t in range(len(x)):

                # create input + feedback
                if t==0:
                    fb_a = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    fb_b = fb_a

                    input_fb = np.concatenate((x[t], fb_a, fb_b), axis=None)
                else:
                    fb_a = self.amp_a * (y[t] + (teacherfb_sigma * np.random.randn(len(y[t])))) # teacher feedback + simulated Gaussian noise: mu=0, sigma=0.2 (default)
                    fb_b = self.amp_b * (y[t] + (teacherfb_sigma * np.random.randn(len(y[t]))))

                    input_fb = np.concatenate((x[t], fb_a, fb_b), axis=None)

                # harvest reservoir states
                rstate = self.reservoir1.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='zero':
                self.reservoir1.reset()

            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))

        # fit readout layer
        self.readout1.fit(train_states, train_targets)

        return
    

    def train_r2(self, input=None, target=None, teacherfb_sigma=0.2, warmup=0, reset='zero'):
        train_states = []
        train_targets = []
        for (x, y) in zip(input, target):
            for t in range(len(x)):

                # create input + feedback
                if t==0:
                    fb_a = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    fb_b = fb_a

                    input_fb = np.concatenate((x[t], fb_a, fb_b), axis=None) 
                else:
                    fb_a = self.amp_a * (y[t] + (teacherfb_sigma * np.random.randn(len(y[t])))) # teacher feedback + simulated Gaussian noise: mu=0, sigma=0.2 (default)
                    fb_b = self.amp_b * (y[t] + (teacherfb_sigma * np.random.randn(len(y[t]))))

                    input_fb = np.concatenate((x[t], fb_a, fb_b), axis=None)

                # harvest reservoir states
                rstate = self.reservoir2.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='zero':
                self.reservoir2.reset()

            if reset=='random':
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

        # fit readout layer
        self.readout2.fit(train_states, train_targets)

        return


    def test(self, input=None, target=None, condition=None, input_sigma=[0,0], reset='zero', save_reservoir=False):
        Y_pred1 = []
        Y_pred2 = []
        R_states1 = []
        R_states2 = []

        for (x, y) in zip(input, target):
            y1 = np.array([np.zeros(self.readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(self.readout2.output_dim)] * len(x))

            if save_reservoir==True:
                r1 = np.array([np.zeros(self.reservoir1.output_dim)] * len(x))
                r2 = np.array([np.zeros(self.reservoir2.output_dim)] * len(x))

            for t in range(len(x)):
                # noise inputs
                noise_1 = (np.random.randn(len(x[t])) * input_sigma[0])
                noise_2 = (np.random.randn(len(x[t])) * input_sigma[1])

                if t==0:
                    fb_a = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    fb_b = fb_a

                    input_fb_1 = np.concatenate(((x[t] + noise_1), fb_a, fb_b), axis=None)
                    input_fb_2 = np.concatenate(((x[t] + noise_2), fb_a, fb_b), axis=None)

                else:
                    if condition=="auto": # autocentric
                        input_fb_1 = np.concatenate(((x[t] + noise_1), self.amp_a * self.readout1.state(), self.amp_b * self.readout1.state()), axis=None)
                        input_fb_2 = np.concatenate(((x[t] + noise_2), self.amp_a * self.readout2.state(), self.amp_b * self.readout2.state()), axis=None)

                    if condition=="allo": # allocentric
                        input_fb_1 = np.concatenate(((x[t] + noise_1), self.amp_a * self.readout2.state(), self.amp_b * self.readout2.state()), axis=None)
                        input_fb_2 = np.concatenate(((x[t] + noise_2), self.amp_a * self.readout1.state(), self.amp_b * self.readout1.state()), axis=None)
                    
                    if condition=="poly_parall": # polycentric (parallel streams)
                        input_fb_1 = np.concatenate(((x[t] + noise_1), self.amp_a * self.readout1.state(), self.amp_b * self.readout2.state()), axis=None)
                        input_fb_2 = np.concatenate(((x[t] + noise_2), self.amp_a * self.readout2.state(), self.amp_b * self.readout1.state()), axis=None)

                    if condition=="poly_integr": # polycentric (integrated streams)
                        avg_fb = np.mean([self.readout1.state(), self.readout2.state()], axis=0)
                        input_fb_1 = np.concatenate(((x[t] + noise_1), self.amp_a * avg_fb, self.amp_b * avg_fb), axis=None)
                        input_fb_2 = np.concatenate(((x[t] + noise_2), self.amp_a * avg_fb, self.amp_b * avg_fb), axis=None)

                # harvest reservoir states + predictions
                rstate1 = self.reservoir1.run(input_fb_1)
                rstate2 = self.reservoir2.run(input_fb_2)

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
            if reset=='zero':
                self.reservoir1.reset()
                self.reservoir2.reset()
                
            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

        return Y_pred1, Y_pred2, R_states1, R_states2
    

    def accuracy(self, pred1=None, pred2=None, target=None):
        acc1 = [np.argmax(test[0]) == np.argmax(np.sum(pred, axis=0)) for (test, pred) in zip(target, pred1)] # integrate readouts over time and select most active node as decision
        acc2 = [np.argmax(test[0]) == np.argmax(np.sum(pred, axis=0)) for (test, pred) in zip(target, pred2)]

        # dyad accuracy (if coupled) // wisdom of the crowd (if uncoupled)
        joint_acc = np.mean([np.argmax(test[0]) == np.argmax(np.sum(np.sum((p1, p2), axis=1), axis=0)) for (test, p1, p2) in zip(target, pred1, pred2)]) # avg readouts for joint decision

        # accuracy of more (and less) sensitive esn in the dyad
        upper_acc = np.max([np.mean(acc1), np.mean(acc2)])
        lower_acc = np.min([np.mean(acc1), np.mean(acc2)])

        # avg esn accuracy
        avg_acc = np.mean([np.mean(acc1), np.mean(acc2)])

        return joint_acc, upper_acc, lower_acc, avg_acc
    


#############################################



class CesnModel_V3:
    def __init__(self, nnodes=None, in_plink=None, rc_plink=[0.1,0.1], seed=None):
        # create reservoirs
        if seed==None:
            self.reservoir1 = Reservoir(units=nnodes[0], input_connectivity=in_plink[0], rc_connectivity=rc_plink[0], sr=0.9, lr=0.1, activation='tanh')
            self.reservoir2 = Reservoir(units=nnodes[1], input_connectivity=in_plink[1], rc_connectivity=rc_plink[1], sr=0.9, lr=0.1, activation='tanh')
        else:
            self.reservoir1 = Reservoir(units=nnodes[0], input_connectivity=in_plink[0], rc_connectivity=rc_plink[0], sr=0.9, lr=0.1, activation='tanh', seed=seed[0])
            self.reservoir2 = Reservoir(units=nnodes[1], input_connectivity=in_plink[1], rc_connectivity=rc_plink[1], sr=0.9, lr=0.1, activation='tanh', seed=seed[1])
            
        # create readouts
        self.readout1 = Ridge(ridge=1e-6)
        self.readout2 = Ridge(ridge=1e-6)


    def train_r1(self, input=None, target=None, teacherfb_sigma=0.2, warmup=0, reset='zero'):
        train_states = []
        train_targets = []
        for (x, y) in zip(input, target):
            for t in range(len(x)):
                # create input + feedback
                if t==0:
                    fb = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    input_fb = np.concatenate((x[t], fb), axis=None)
                else:
                    fb = y[t] + (teacherfb_sigma * np.random.randn(len(y[t]))) # teacher feedback + simulated Gaussian noise: mu=0, sigma=0.2 (default)
                    input_fb = np.concatenate((x[t], fb), axis=None)

                # harvest reservoir states
                rstate = self.reservoir1.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='zero':
                self.reservoir1.reset()

            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))

        # fit readout layer
        self.readout1.fit(train_states, train_targets)

        return
    

    def train_r2(self, input=None, target=None, teacherfb_sigma=0.2, warmup=0, reset='zero'):
        train_states = []
        train_targets = []
        for (x, y) in zip(input, target):
            for t in range(len(x)):
                # create input + feedback
                if t==0:
                    fb = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    input_fb = np.concatenate((x[t], fb), axis=None) 
                else:
                    fb = y[t] + (teacherfb_sigma * np.random.randn(len(y[t]))) # teacher feedback + simulated Gaussian noise: mu=0, sigma=0.2 (default)
                    input_fb = np.concatenate((x[t], fb), axis=None)

                # harvest reservoir states
                rstate = self.reservoir2.run(input_fb)

                if t >= warmup:
                    train_states.append(rstate)
                    train_targets.append(y[t, np.newaxis])

            # reset reservoirs
            if reset=='zero':
                self.reservoir2.reset()

            if reset=='random':
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

        # fit readout layer
        self.readout2.fit(train_states, train_targets)

        return


    def test(self, input=None, target=None, condition=None, input_sigma=[0,0], reset='zero', save_reservoir=False):
        Y_pred1 = []
        Y_pred2 = []
        R_states1 = []
        R_states2 = []

        for (x, y) in zip(input, target):
            y1 = np.array([np.zeros(self.readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(self.readout2.output_dim)] * len(x))

            if save_reservoir==True:
                r1 = np.array([np.zeros(self.reservoir1.output_dim)] * len(x))
                r2 = np.array([np.zeros(self.reservoir2.output_dim)] * len(x))

            for t in range(len(x)):
                # noise inputs
                noise1 = (np.random.randn(len(x[t])) * input_sigma[0])
                noise2 = (np.random.randn(len(x[t])) * input_sigma[1])

                if t==0:
                    fb = np.array(np.zeros(len(y[t]))) # no feedback on first timestep
                    input_fb1 = np.concatenate(((x[t] + noise1), fb), axis=None)
                    input_fb2 = np.concatenate(((x[t] + noise2), fb), axis=None)

                else:
                    if condition=="auto": # autocentric fb
                        input_fb1 = np.concatenate(((x[t] + noise1), self.readout1.state()), axis=None)
                        input_fb2 = np.concatenate(((x[t] + noise2), self.readout2.state()), axis=None)

                    if condition=="allo": # allocentric fb
                        input_fb1 = np.concatenate(((x[t] + noise1), self.readout2.state()), axis=None)
                        input_fb2 = np.concatenate(((x[t] + noise2), self.readout1.state()), axis=None)

                    if condition=="poly": # polycentric fb
                        avg_fb = np.mean([self.readout1.state(), self.readout2.state()], axis=0)
                        input_fb1 = np.concatenate(((x[t] + noise1), avg_fb), axis=None)
                        input_fb2 = np.concatenate(((x[t] + noise2), avg_fb), axis=None)

                # harvest reservoir states + predictions
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
            if reset=='zero':
                self.reservoir1.reset()
                self.reservoir2.reset()
                
            if reset=='random':
                self.reservoir1.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir1.output_dim))
                self.reservoir2.reset(to_state=np.random.uniform(-1, 1, size=self.reservoir2.output_dim))

        return Y_pred1, Y_pred2, R_states1, R_states2
    

    def accuracy(self, pred1=None, pred2=None, target=None):
        acc1 = [np.argmax(test[0]) == np.argmax(np.sum(pred, axis=0)) for (test, pred) in zip(target, pred1)] # integrate readouts over time and select most active node as decision
        acc2 = [np.argmax(test[0]) == np.argmax(np.sum(pred, axis=0)) for (test, pred) in zip(target, pred2)]

        # dyad accuracy (if coupled) // wisdom of the crowd (if uncoupled)
        joint_acc = np.mean([np.argmax(test[0]) == np.argmax(np.sum(np.sum((p1, p2), axis=1), axis=0)) for (test, p1, p2) in zip(target, pred1, pred2)]) # avg readouts for joint decision

        # accuracy of more (and less) sensitive esn in the dyad
        upper_acc = np.max([np.mean(acc1), np.mean(acc2)])
        lower_acc = np.min([np.mean(acc1), np.mean(acc2)])

        # avg esn accuracy
        avg_acc = np.mean([np.mean(acc1), np.mean(acc2)])

        return joint_acc, upper_acc, lower_acc, avg_acc
