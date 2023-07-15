# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import torch
import gpytorch
import numpy as np
from GPBase import BaseGP
from itertools import chain, repeat
from datetime import datetime
from gpytorch.constraints.constraints import Interval
import os
import math

# Define the base multi-task model:
class MVGPDKL(gpytorch.models.ExactGP):
    """
    Model in use. Multitask model with deep kernel learning
    
    """
    def __init__(self, train_x, train_y, likelihood, layer_sizes=None, deltas=False, num_tasks=2):
        super(MVGPDKL, self).__init__(train_x, train_y, likelihood)
        """
        Args:
            train_x (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            train_y (1d torch.tensor): free energy calculation
            likelihood (gpytorch.liklihood) : gaussian likelihood function for conditional distribution
            layer sizes (list): list of layer sizes ie: [40,20]
            deltas (boolean): don't know what this is
            num_tasks (int): number of fidelities
            
        """
        print(train_x[0].type(), train_y.type(), "TRAIN TYPES")

        if layer_sizes is None:
            layer_sizes = []

        # Set up the NN layers preceding the GP
        self.lin_layers = []
        for i, size_i in enumerate(layer_sizes):
            if i == 0:
                input_size = train_x[0].shape[1]
            else:
                input_size = layer_sizes[i-1]
            self.lin_layers.append(torch.nn.Linear(input_size, size_i))
        self.lin_layers = torch.nn.ModuleList(self.lin_layers)
        for i, lin_layer in enumerate(self.lin_layers):
            torch.nn.init.eye_(lin_layer.weight.data)
            if i == 0 and deltas:
                print('Initializing with delta structure!')
                # Assume that the predictor columns have been ordered such that
                # the input is mutant_(feature_0), ..., mutant(feature_k),
                # wt_(feature_0), ... wt(feature_k), i.e., that the
                input_size = lin_layer.weight.shape[1]
                assert input_size == 2 * math.floor(input_size/2.0)  # I.e., it's even
                for colpos in range(math.floor(input_size/2.0)):
                    colneg = colpos + math.floor(input_size/2.0)
                    # print(colpos, colneg)
                    lin_layer.weight.data[:, colneg] = \
                        -1.0 * lin_layer.weight.data[:, colpos]
            lin_layer.weight.data = 0.1 * lin_layer.weight.data + 0.005 * torch.randn_like(lin_layer.weight.data)

        if layer_sizes == []:
            layer_sizes = [train_x[0].shape[1]]

        self.num_tasks = num_tasks

        # Set up the GP on top
        self.mean_module = gpytorch.means.ConstantMean()

        if self.num_tasks > 1:
            self.covar_module = gpytorch.kernels.RBFKernel()

            # We learn an IndexKernel for num_tasks tasks
            self.task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=num_tasks, rank=num_tasks-1)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.task_covar_module = gpytorch.kernels.RBFKernel()  # Shouldn't do anything

    def forward(self, x, i):
        """
        Args:
            x (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            i (1d torch.tensor): task indices
            
        """
        for idxi, lli in enumerate(self.lin_layers):
            if idxi == len(self.lin_layers):
                x = lli(x)
            else:
                x = torch.tanh(lli(x))

        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        if self.num_tasks > 1:
            # Get task-task covariance
            covar_i = self.task_covar_module(i)
            # Multiply the two together to get the covariance we want
            covar = covar_x.mul(covar_i) 
        else:
            covar = covar_x

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Attempt to pump some more data in the model
        
        Args:
            inputs (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            target (1d torch.tensor): targers
                    
        """
        try:
            model_out = super(MVGPDKL, self).get_fantasy_model(inputs, targets, **kwargs)
        except RuntimeError:
            warnings.warn('Guarding failed get_fantasy_model call', RuntimeWarning)
            model_out = deepcopy(self)
        return model_out


class MultitaskVanillaGPModelDKL(BaseGP):
    """
    Deep kernel learning + hadamard multitask
    
    Args:
        :attr:`name` (optional, string):
            model name
        :attr:`num_iters` (int):
            number of iterations for Gaussian Process
        :attr:`learning_rate` (int):
            learning rate for conjugate gradient. recommended around .1 or .01
        :attr:`noise_covar` (float):
            hyperparamter, noise assumed in the data
       :attr:`lengthscale` (float):
            hyperparameter, magnitude relative to assumed correlation in data
       :attr:`output_scale` (optional, float):
            scaling parameter
            
    """

    def __init__(self, name='MultitaskVanillaDKL', num_iters=50, learning_rate=1e-1,
                 noise_covar=1.0, length_scale=100.0, output_scale=1.0, load_path=None):
        super().__init__(name)
        
        """
        Args:
            num_iters, learning_rate: These are training parameters that decide how long model should train, at what rate.
            noise_covar, lenght_scale, outputscale: direct model "hyper" parameters that get passed into kernel. 
        """

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood( noise_constraint=Interval(0.0001, 1.0)).float() 

        self.training_iter = num_iters
        self.learning_rate = learning_rate
        self.noise_covar = noise_covar
        self.length_scale = length_scale
        self.output_scale = output_scale
        self.load_path = load_path
        


    def fit(self, X, y, **kwargs):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            y (1d torch.tensor): training data in list form, (nsameples) for each response
        """
     
        X = X.float()
        y = y.float() 
        i_vec = kwargs["i"]        
        

        layer_sizes = [40, 10]
        coordinate_descent_cycles = 20
        
        
        i_vec[0] = 0
        i_vec[1] = 1
        i_vec[2] = 2
        i_vec[3] = 3
        
        i_count = len(np.unique(i_vec))
        

        self.model = MVGPDKL((X, i_vec), y, self.likelihood, layer_sizes=layer_sizes, num_tasks=i_count, deltas=True).float() 
        if self.load_path:
            self.load_model(X, y, **kwargs)
        # add training phases: (num_iters, params to optimize, learning rate)
        training_phases = []
        for cycle in range(coordinate_descent_cycles):
            training_phases.append(
                (5, chain(
                    self.model.mean_module.parameters(),
                    self.model.covar_module.parameters(),
                    self.model.task_covar_module.parameters(),
                    self.likelihood.parameters()), 0.1),
            )
            training_phases.append(
            (10, self.model.lin_layers.parameters(), 0.005)
            )
        

        training_phases = training_phases + [
            (50, chain(
             self.model.mean_module.parameters(),
             self.model.covar_module.parameters(),
             self.model.task_covar_module.parameters(),
             self.likelihood.parameters()), 0.1),
            (200, self.model.parameters(), 0.01),
            (200, self.model.parameters(), 0.03)
        ]
        
        # DEBUG
        # training_phases = training_phases[0:1]
       

        # #############
        # ### Train ###
        # #############

        # Find the optimal hyperparameters
        self.model.train()
        self.likelihood.train()
        
        if hasattr(self, "writer"):
            recording = True

        # "Loss" for GPs - the marginal log likelihood
        counter = 0 # counter for logging TT curve
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for phaseidx, (train_iters, optimset, lr) in enumerate(training_phases):
            optimizer = torch.optim.SGD([
                {'params': optimset}
            ], lr=lr)
            print('{}: Beginning phase {} training'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), phaseidx))
            for i in range(train_iters):
                #if i%10 == 0:
                 #   print_parameters(self.model, self.likelihood)
                optimizer.zero_grad()
                output = self.model(X, i_vec)
                loss = -mll(output, y)
                loss.backward()
                print('%s: Phase %d iter %d/%d - Loss: %.3f' % (datetime.now().strftime('%Y%m%d_%H%M%S'), phaseidx, i + 1, train_iters, loss.item()))
                if hasattr(self, "writer")  and recording:
                    self.writer.add_scalar("-mll", loss.item(), counter)
                optimizer.step()
                counter += 1
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.model(X, i_vec)
                loss = -mll(output, y)
                print('Final training loss: %.3f' % (loss.item()))
            print('{}: Phase {} training complete.'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), phaseidx))
        if hasattr(self, "writer") and recording: 
            self.writer.flush()

    def predict(self, X, **kwargs):
        """ 
        Predict using trained model
        
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
        """
        X = X.float() 

        return_std = False
        if 'return_std' in kwargs.keys():
            return_std = kwargs['return_std']
            
        i_vec = kwargs["i"]
        i_count = int(max(i_vec))
        
        # put in eval mode
        self.model.eval()
        self.likelihood.eval()

        # make predictions using likelihood
        with torch.no_grad():
            observed_pred = self.likelihood(self.model(X,i_vec))
            pred_y = observed_pred.mean

        if return_std:
            lower, upper = observed_pred.confidence_region()
            return pred_y, lower, upper
        else:
            return pred_y
        

    def load_model(self, X, y, **kwargs):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            y (1d torch.tensor): training data in list form, (nsameples) for each response
        """
        
        X = X.float() 
        y = y.float() 
        i_vec = kwargs["i"]        
        layer_sizes = [40, 10]
        
        i_vec[0] = 0
        i_vec[1] = 1
        i_vec[2] = 2
        i_vec[3] = 3
        i_count = len(np.unique(i_vec))
        if kwargs["ntasks"]:   
            i_count = kwargs["ntasks"]

        self.model = MVGPDKL((X, i_vec), y, self.likelihood, layer_sizes=layer_sizes, num_tasks=i_count, deltas=True).float() 
        try:
            state_dict = torch.load(self.load_path)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print("exeception from loading state dictionary for MVGPDKL")
            print("Are you loading single fidelity data into mutlifidelity model?")
            raise e
        print("LOADED FROM : ", self.load_path)

        return 

    def save_model(self, path):
        """
        save a trained model
        Parameters
        ----------- 
        path: str
            Filepath specifying where model should be saved.
        """
        if self.name not in path:
            path = os.path.join(path, self.name)
        if torch.cuda.is_available():
            state_dict = self.model.cpu().state_dict()
        else:
            state_dict = self.model.state_dict()
        if 'task_covar_module.covar_factor' in state_dict and len(state_dict['task_covar_module.covar_factor']) > 1:
            iter_dict = state_dict.copy()
            for key in iter_dict:
                if "covar_module" in key:
                    new_key = key.replace("covar_module", "covar_module.module")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            torch.save(iter_dict, path)
        else:
            #print(method.model.state_dict()['covar_module.raw_lengthscale'])
            state_dict['task_covar_module.covar_factor'] =  torch.Tensor([1])
            torch.save(state_dict, path)


        
cols = [
        'NumAcidicAcidic', 'NumAcidicAliphatic', 'NumAcidicAmidic',
        'NumAcidicAromatic', 'NumAcidicBasic', 'NumAcidicHydroxilic',
        'NumAcidicSulfurous', 'NumAliphaticAliphatic', 'NumAliphaticAmidic',
        'NumAliphaticAromatic', 'NumAliphaticBasic', 'NumAliphaticHydroxilic',
        'NumAliphaticSulfurous', 'NumAmidicAmidic', 'NumAmidicAromatic',
        'NumAmidicBasic', 'NumAmidicHydroxilic', 'NumAmidicSulfurous',
        'NumAromaticAromatic', 'NumAromaticBasic', 'NumAromaticHydroxilic',
        'NumAromaticSulfurous', 'NumBasicBasic', 'NumBasicHydroxilic',
        'NumBasicSulfurous', 'NumHydroxilicHydroxilic',
        'NumHydroxilicSulfurous', 'NumSulfurousSulfurous', 'NumLargeLarge',
        'NumLargeMedium', 'NumLargeSmall', 'NumLargeVeryLarge',
        'NumLargeVerySmall', 'NumMediumMedium', 'NumMediumSmall',
        'NumMediumVeryLarge', 'NumMediumVerySmall', 'NumSmallSmall',
        'NumSmallVeryLarge', 'NumSmallVerySmall', 'NumVeryLargeVeryLarge',
        'NumVeryLargeVerySmall', 'NumVerySmallVerySmall', 'WT_NumAcidicAcidic',
        'WT_NumAcidicAliphatic', 'WT_NumAcidicAmidic', 'WT_NumAcidicAromatic',
        'WT_NumAcidicBasic', 'WT_NumAcidicHydroxilic', 'WT_NumAcidicSulfurous',
        'WT_NumAliphaticAliphatic', 'WT_NumAliphaticAmidic',
        'WT_NumAliphaticAromatic', 'WT_NumAliphaticBasic',
        'WT_NumAliphaticHydroxilic', 'WT_NumAliphaticSulfurous',
        'WT_NumAmidicAmidic', 'WT_NumAmidicAromatic', 'WT_NumAmidicBasic',
        'WT_NumAmidicHydroxilic', 'WT_NumAmidicSulfurous',
        'WT_NumAromaticAromatic', 'WT_NumAromaticBasic',
        'WT_NumAromaticHydroxilic', 'WT_NumAromaticSulfurous',
        'WT_NumBasicBasic', 'WT_NumBasicHydroxilic', 'WT_NumBasicSulfurous',
        'WT_NumHydroxilicHydroxilic', 'WT_NumHydroxilicSulfurous',
        'WT_NumSulfurousSulfurous', 'WT_NumLargeLarge', 'WT_NumLargeMedium',
        'WT_NumLargeSmall', 'WT_NumLargeVeryLarge', 'WT_NumLargeVerySmall',
        'WT_NumMediumMedium', 'WT_NumMediumSmall', 'WT_NumMediumVeryLarge',
        'WT_NumMediumVerySmall', 'WT_NumSmallSmall', 'WT_NumSmallVeryLarge',
        'WT_NumSmallVerySmall', 'WT_NumVeryLargeVeryLarge',
        'WT_NumVeryLargeVerySmall', 'WT_NumVerySmallVerySmall']

def print_parameters(model, likelihood):
    """
    Helper funtion that takes in core model and prints model parameters. Lifted from ABAG ML repo.
    Args:
        model (gpytorch.model)
        likeliehood (gpytorch.likelihoods)
        
    """
    # Print out the learned model parameters:
    print('\n--------')
    print('Learned noise parameters:')
    for ni, parami in likelihood.named_parameters():
        print('{}: {}'.format(ni, parami))
    try:
        lnc = likelihood.noise
        print('Noise: {}'.format(lnc))
    except:
        pass
    print('------')
    print('Learned model parameters:')
    for ni, parami in model.named_parameters():
        print('{}: {}'.format(ni, parami))

    try:
        tcm = model.task_covar_module.covar_matrix.evaluate()
        print('Task covariance matrix:')
        print(tcm)
    except:
        pass

    try:
        lengthscales = model.covar_module.lengthscale
        print('Lengthscales in the kernel:')
        print(lengthscales)
    except:
        pass
    print('------')

    