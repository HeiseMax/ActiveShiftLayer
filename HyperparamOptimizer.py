import numpy as np
import matplotlib.pyplot as plt

import sobol
import optunity
import GPy
from GPy import models


class HyperparamOptimizer():
    def __init__(self, train_function, parameter_range, device):
        self.train_function = train_function
        self.parameter_range = parameter_range
        self.num_params = len(parameter_range.keys())
        self.device = device

        self.gpmodel = None

        self.points = np.zeros((0, self.num_params))
        self.values = np.zeros((0, 1))
        self.gp_predictions = np.zeros(0)

    def GPPredict(self, **kwargs):
        return self.gpmodel.predict_noiseless(np.array([[kwargs[param] for param in kwargs]]))[0]

    def maxVarGP(self, **kwargs):
        return self.gpmodel.predict_noiseless(np.array([[kwargs[param] for param in kwargs]]))[1]

    def get_sobol_samples(self, num_samples):
        first_index = len(self.values)
        self.values = np.append(
            self.values, np.zeros((num_samples, 1)), axis=0)
        self.points = np.append(self.points, np.zeros(
            (num_samples, self.num_params)), axis=0)
        self.gp_predictions = np.append(
            self.gp_predictions, np.zeros(num_samples))
        self.gp_predictions[first_index:] = None

        parameterUpperLimits = np.array(
            [self.parameter_range[i][1] for i in self.parameter_range])
        parameterLowerLimits = np.array(
            [self.parameter_range[i][0] for i in self.parameter_range])

        for i in range(num_samples):
            paramters = {}
            parameter_list = sobol.i4_sobol(self.num_params, i)[
                0] * (parameterUpperLimits - parameterLowerLimits) + parameterLowerLimits
            for j, param in enumerate(self.parameter_range):
                paramters[param] = parameter_list[j]
            self.values[i +
                        first_index] = self.train_function(paramters, self.device)
            self.points[i + first_index] = parameter_list

    def GPR_optim(self, num_iterations, end_var_optim=30):
        first_index = len(self.values)
        self.values = np.append(
            self.values, np.zeros((num_iterations, 1)), axis=0)
        self.points = np.append(self.points, np.zeros(
            (num_iterations, self.num_params)), axis=0)
        self.gp_predictions = np.append(
            self.gp_predictions, np.zeros(num_iterations))

        for i in range(num_iterations):
            self.gpmodel = models.GPRegression(
                self.points, self.values, GPy.kern.Matern52(self.num_params, ARD=True))

            if i % 2 == 0 or i >= end_var_optim:
                pars, details, _ = optunity.maximize(
                    self.GPPredict, **self.parameter_range)
            else:
                pars, details, _ = optunity.maximize(
                    self.maxVarGP, **self.parameter_range)

            # q = np.array([pars['lr'], pars['momentum'],
            #              pars["p_randomTransform"]])

            q = np.array([pars[i] for i in pars])

            self.gp_predictions[i + first_index] = self.GPPredict(**pars)
            self.points[i + first_index] = q
            self.values[i +
                        first_index] = self.train_function(pars, self.device)

    def get_best_parameters(self, n_best):
        configs = []
        ind = np.flip(np.argsort(self.values, axis=0))[:n_best]
        params = self.points[ind]
        for i in range(n_best):
            best_params = {}
            best_params["acc"] = self.values[ind][i][0][0]
            for j, param in enumerate(self.parameter_range):
                best_params[param] = params[i][0][j]
            configs.append(best_params)

        return configs
