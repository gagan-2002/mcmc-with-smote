# !/usr/bin/python


# MCMC Random# !/usr/bin/python


# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

# Data (Sunspot and Lazer). Taken' Theorem used for Data Reconstruction (Dimension = 4, Timelag = 2).
# Data procesing file is included.

# RMSE (Root Mean Squared Error)

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk


# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra



# Reference for publication for this code
# [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()
        self.lrate = learn_rate

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        #self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        #self.B2 += (-1 * self.lrate * out_delta)
        #self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        #self.B1 += (-1 * self.lrate * hid_delta)

        layer = 1  # hidden to output
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        for y in range(0, self.Top[layer + 1]):
            self.B2[y] += -1 * self.lrate * out_delta[y]

        layer = 0  # Input to Hidden
        for x in range(0, self.Top[layer]):
            for y in range(0, self.Top[layer + 1]):
                self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        for y in range(0, self.Top[layer + 1]):
            self.B1[y] += -1 * self.lrate * hid_delta[y]

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, depth):
            for i in range(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self. Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated

    def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[i] = self.out

        return fx



# --------------------------------------------------------------------------

# -------------------------------------------------------------------


class MCMC:
    def __init__(self,  use_langevin_gradients , l_prob,  learn_rate,  samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata 
        self.use_langevin_gradients  =  use_langevin_gradients 

        self.l_prob = l_prob # likelihood prob

        self.learn_rate =  learn_rate
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        #loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq

        n = y.shape[0]

        loss =( -(n/2) * np.log(2 * math.pi * tausq)) -( (1/(2*tausq)) * np.sum(np.square(y - fx)))
        return [loss, fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self, w_limit, tau_limit):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self, w_limit, tau_limit):

        # ------------------- initialize MCMC
        initialtraindata = self.traindata
        samples = self.samples
        X, y = initialtraindata[:, :-1], initialtraindata[:, -1]
        counter = Counter(y)
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        y = y.reshape(X.shape[0],1)
        self.traindata = np.append(X,y,axis=1)
        self.sgd_depth = 1
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)
        data = self.traindata
        x = 1
        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print(y_train.size)
        print(y_test.size)

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        #step_w = 0.05;  # defines how much variation you need in changes to w
        #step_eta = 0.2; # exp 0


        step_w = w_limit  # defines how much variation you need in changes to w
        step_eta = tau_limit #exp 1
        # --------------------- Declare FNN and initialize
         
        neuralnet = Network(self.topology, self.traindata, self.testdata, self.learn_rate)
        print('evaluate Initial w')

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        #sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        #np.fill_diagonal(sigma_diagmat, step_w)

        delta_likelihood = 0.5 # an arbitrary position


        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print(likelihood, ' Initial likelihood')

        naccept = 0

        langevin_count = 0
        auc = np.zeros(samples-1)
        f11 = np.zeros(samples-1)
        f12 = np.zeros(samples-1)
        f13 = np.zeros(samples-1)


        for i in range(samples - 1):
            if(i%(samples/10)==0):
                X, y = initialtraindata[:, :-1], initialtraindata[:, -1]
                counter = Counter(y)
                oversample = SMOTE()
                X, y = oversample.fit_resample(X, y)
                y = y.reshape(X.shape[0],1)
                data1 = np.append(X,y,axis=1)
                data = np.concatenate((data,data1),axis = 1)
                x+=1
                print('oversampled for',i)
            q = random.randint(1,x)
            self.traindata = data[:,(q-1)*initialtraindata.shape[1]:q*initialtraindata.shape[1]]
            neuralnet = Network(self.topology, self.traindata, self.testdata, self.learn_rate)
            pred_train = neuralnet.evaluate_proposal(self.traindata, w)
            y_train = self.traindata[:, netw[0]]

            lx = np.random.uniform(0,1,1)

            if (self.use_langevin_gradients is True) and (lx< self.l_prob):  
                w_gd = neuralnet.langevin_gradient(self.traindata, w.copy(), self.sgd_depth) # Eq 8
                w_proposal = np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = neuralnet.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth) 
                #first = np.log(multivariate_normal.pdf(w , w_prop_gd , sigma_diagmat)) 
                #second = np.log(multivariate_normal.pdf(w_proposal , w_gd , sigma_diagmat)) # this gives numerical instability - hence we give a simple implementation next that takes out log 

                wc_delta = (w- w_prop_gd) 
                wp_delta = (w_proposal - w_gd )

                sigma_sq = step_w *step_w

                first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq

            
                diff_prop =  first - second  
                langevin_count = langevin_count + 1

                

            else:
                diff_prop = 0
                w_proposal = np.random.normal(w, step_w, w_size)

             
            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro) 

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients


            diff_prior = prior_prop - prior_current

            diff_likelihood = likelihood_proposal - likelihood

            #mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))

            try:
                mh_prob = min(1, math.exp(diff_likelihood+diff_prior+ diff_prop))

            except OverflowError as e:
                mh_prob = 1



            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position 
                naccept += 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = w_proposal
                eta = eta_pro
                if i%1000 ==0:
                    print(i,likelihood, prior_current, diff_prop, rmsetrain, rmsetest, 'accepted')
                 

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]
            auc[i] = roc_auc_score(y_test, pred_test)
            pred_test=(pred_test>0.5)
            f11[i] = f1_score(y_test, pred_test, average='weighted')
            f12[i] = f1_score(y_test, pred_test, average='macro')
            f13[i] = f1_score(y_test, pred_test, average='micro')

        print(naccept, ' num accepted')
        print(naccept / (samples * 1.0)*100, '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100
        print(langevin_count, ' langevin_count')
        burnin=5000
        auc_afterburnin=auc[burnin:-1]
        f11_afterburnin=f11[burnin:-1]
        f12_afterburnin=f12[burnin:-1]
        f13_afterburnin=f13[burnin:-1]
        auc1 = np.mean(auc_afterburnin)
        print('AUC(MEAN): %.3f' % auc1)
        pred_test=(pred_test>0.5)
        f111=np.mean(f11_afterburnin)
        f121=np.mean(f12_afterburnin)
        f131=np.mean(f13_afterburnin)
        print('f1 score weighted(MEAN):%.3f' % f111)
        print('f1 score macro(MEAN):%.3f' % f121)
        print('f1 score micro(MEAN):%.3f' % f131)
        auc2 = roc_auc_score(y_train, pred_train)
        print('AUC(TRAIN): %.3f' % auc2)
        print('Precision: %.3f' % precision_score(y_test, pred_test))
        print('Recall: %.3f' % recall_score(y_test, pred_test))
        print('Accuracy: %.3f' % accuracy_score(y_test,  pred_test))
        x = np.array(range(1, samples))
        y = np.array(auc)
        plt.title("Line graph")
        plt.xlabel("samples")
        plt.ylabel("AUC(TEST)")
        plt.plot(x, y, color ="green")
        plt.show()
        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
    hidden = 11
    input = 10  #
    output = 1
    w_limit =  0.025 # step size for w
    tau_limit = 0.2 # step size for eta
    ecoli = pd.read_csv('abalone.data',sep=',')
    name	= "abalone"
    ecolidat = pd.DataFrame(ecoli)
    ecolidata = pd.get_dummies(ecolidat)
    traindata = ecolidata.sample(frac=0.6, random_state=25)
    testdata = ecolidata.drop(traindata.index)
    traindata = traindata.to_numpy()
    testdata = testdata.to_numpy()
    col1 = np.zeros(traindata.shape[0])
    for i in range(0,traindata.shape[0]):
        if traindata[i,7] <=10:
            col1[i] = 0
        if 10 < traindata[i,7]:
            col1[i] = 1
    traindata=np.delete(traindata,7,1)
    col1=col1.reshape(traindata.shape[0],1)
    traindata = np.append(traindata,col1,axis=1)
    cl1 = np.zeros(testdata.shape[0])
    for i in range(0,testdata.shape[0]):
        if testdata[i,7] <=10:
            cl1[i] = 0
        if 10 < testdata[i,7]:
            cl1[i] = 1
    testdata=np.delete(testdata,7,1)
    cl1=cl1.reshape(testdata.shape[0],1)
    testdata = np.append(testdata,cl1,axis=1)
    topology = [input, hidden, output]
    random.seed(time.time())
    numSamples = 10000 # need to decide yourself
    use_langevin_gradients  = True
    l_prob = 0.5
    learn_rate = 0.01
    timer = time.time() 
    mcmc = MCMC( use_langevin_gradients , l_prob,  learn_rate, numSamples, traindata, testdata, topology)  # declare class
    [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler(w_limit, tau_limit)
    print('sucessfully sampled')
    burnin = 0.5 * numSamples  # use post burn in samples
    
    timer2 = time.time()
    timetotal = (timer2 - timer) /60
    print((timetotal), 'min taken')
    pos_w = pos_w[int(burnin):, ]
    pos_tau = pos_tau[int(burnin):, ]
    fx_mu = fx_test.mean(axis=0)
    fx_high = np.percentile(fx_test, 95, axis=0)
    fx_low = np.percentile(fx_test, 5, axis=0)
    fx_mu_tr = fx_train.mean(axis=0)
    fx_high_tr = np.percentile(fx_train, 95, axis=0)
    fx_low_tr = np.percentile(fx_train, 5, axis=0)
    pos_w_mean = pos_w.mean(axis=0) 
    rmse_tr = np.mean(rmse_train[int(burnin):])
    rmsetr_std = np.std(rmse_train[int(burnin):])
    rmse_tes = np.mean(rmse_test[int(burnin):])
    rmsetest_std = np.std(rmse_test[int(burnin):])
    print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)

 
        #outres_db = open('result.txt', "a+")

        #np.savetxt(outres_db, ( use_langevin_gradients ,    learn_rate, rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio, timetotal), fmt='%1.5f')


        # ytestdata = testdata[:, input]
        # ytraindata = traindata[:, input]

        # plt.plot(x_test, ytestdata, label='actual')
        # plt.plot(x_test, fx_mu, label='pred. (mean)')
        # plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        # plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        # plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
        # plt.legend(loc='upper right')

        # plt.title("Prediction  Uncertainty ")
        # plt.savefig('mcmcrestest.png') 
        # plt.clf()
        # # -----------------------------------------
        # plt.plot(x_train, ytraindata, label='actual')
        # plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        # plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        # plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        # plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        # plt.legend(loc='upper right')

        # plt.title("Prediction  Uncertainty")
        # plt.savefig('mcmcrestrain.png') 
        # plt.clf()

        # mpl_fig = plt.figure()
        # ax = mpl_fig.add_subplot(111)

        # ax.boxplot(pos_w)
        # ax.set_xlabel('[W1] [B1] [W2] [B2]')
        # ax.set_ylabel('Posterior')
        # plt.legend(loc='upper right')
        # plt.title("Boxplot of Posterior W (weights and biases)")
        # plt.savefig('w_pos.png')
         
        # plt.clf()


if __name__ == "__main__": main()  
