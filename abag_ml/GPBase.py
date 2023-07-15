# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import torch
import sys
import os



class BaseGP(object):
    """
    Abstract class for a base ML predictor

    Parameters
    ----------- 
    name: string
        Name of the method.
    
    Attributes
    ----------
    name: str
        Name of method.

    Methods
    -------
    fit(X, y, **kwargs)
        Fits parameters of model to data.
    predict(X, **kwargs)
        Use model to predict ddG values using X features.
    load_model(X, y, **kwargs)
        Load model parameters into model.
    save_model(path)
        Save model parameters to a path.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, data_descriptor = 'No data descriptor provided'):  # , full_name):
        self.name = name
        self.recording = False
        self.num_tasks = None
        self.train_x = None
        self.train_y = None
        self.train_i = None
        self.data_descriptor = data_descriptor

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        Parameters
        ----------- 
        X:  2d torch.tensor
            training data in 2d array form (nsamples,nfeatures)
        y: 1d torch.tensor 
            training data in list form, (nsameples) for each response
        i: optional 1d torch.tensor
            fidelity
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """         
        Parameters
        ----------- 
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            i (optional 1d torch.tensor): fidelity
        """
        pass

    @abstractmethod
    def load_model(self, X, y, **kwargs):
        """
        Parameters
        ----------- 
        X:  2d torch.tensor
            training data in 2d array form (nsamples,nfeatures)
        y: 1d torch.tensor 
            training data in list form, (nsameples) for each response
        i: optional 1d torch.tensor
            fidelity
        """
        pass

    @abstractmethod
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
            iter_dict = state_dict.copy()
            for key in iter_dict:
                if "covar_module" in key:
                    new_key = key.replace("covar_module", "covar_module.module")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            torch.save(iter_dict, path)

        else:
            #print(method.model.state_dict()['covar_module.raw_lengthscale'])
            torch.save(self.model.state_dict(), path)



    @abstractmethod
    def get_covar(self, **kwargs):
        return self.model.get_covar().evaluate()

    @abstractmethod
    def get_train_targets(self, **kwargs):
        return self.model.train_targets

    @abstractmethod
    def get_likelihood_noise(self, **kwargs):
        return self.likelihood.noise
    
    @abstractmethod
    def get_train_inputs(self, **kwargs):
        if len(self.model.train_inputs) == 1:
            return (self.model.train_inputs[0],None)
        else:
            return self.model.train_inputs

    @abstractmethod
    def add_train_data(self, new_x, new_y, new_i,  **kwargs):
        if len(new_x.shape) < 2:
            new_x = new_x.unsqueeze(1)

        curr_inputs = self.get_train_inputs()
        if curr_inputs[1] is not None:
            inputs = (torch.cat([curr_inputs[0], new_x], dim=-2) ,
                    torch.cat([curr_inputs[1], new_i.reshape((-1, 1))], dim=-2))
        else:
            inputs = torch.cat([curr_inputs[0], new_x], dim=-2)

        targets = torch.cat([self.get_train_targets(), new_y], dim=-1)

        self.model.set_train_data(inputs=inputs,targets=targets,strict=False)

        return inputs, targets
        

    @abstractmethod
    def get_predictive_posterior(self, X, mvn_only=False, **kwargs):
        self.model.eval()
        self.likelihood.eval()
        # TODO: this can cause problems...can there be a better solution
        # you could always allow forward passes to take kwargs?
        # then just pass in i as a kwarg
        with torch.no_grad():
            if "Multitask" in str(type(self.model)):
                joint_mvn = self.model(X, kwargs['i'])
            else:
                joint_mvn = self.model(X)
        if mvn_only:
            return joint_mvn
        res = joint_mvn.covariance_matrix.detach()
        mean = joint_mvn.mean.detach()
        return mean, res