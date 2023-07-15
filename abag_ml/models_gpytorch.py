# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
A module for extending gpytorch's Gaussian Process model classes

"""

from copy import deepcopy
import math
import numpy as np
import gpytorch
import torch
import warnings


def estimate_task_covariance_from_data(train_x, train_y, num_tasks):
    """
    Given a set of training data, get a quick-and-dirty estimate of the task covariance

    :param train_x:
    :param train_y:
    :param num_tasks:
    :return:
    """

    x = train_x[0].detach().clone()
    idx = train_x[1].detach().clone()
    x_unique = x.unique(dim=0)
    tuples = []
    for i in range(x_unique.shape[0]):
        # find all rows that are the same by features:
        match_rows_x_logical = torch.all(np.equal(x, np.vstack([x_unique[i] for j in range(x.shape[0])])), axis=1)
        # find the first row that is the same and has each of the tasks:
        fail = False
        y_temp = []
        for tidx in range(num_tasks):
            if fail:
                continue
            match_rows_idx = idx == tidx
            try:
                first_both = int(np.argwhere(np.logical_and(match_rows_x_logical, match_rows_idx)).reshape((-1, ))[0])
                y_temp.append(train_y[first_both])
            except (IndexError, ValueError):
                fail = True
        # Copy over the data to the list of tuples if all present:
        if fail:
             continue
        tuples.append(y_temp)

    # Get the empirical covariance matrix from numpy
    a = np.cov(np.array([[aij for aij in ai] for ai in tuples]), rowvar=False)

    return torch.tensor(a)


def low_rank_and_diag_decomp(covariance_matrix, rank=None):
    """
    Obtain a low-rank + diagonal decomposition from partial eigendecomposition

    :param covariance_matrix: torch.tensor, assumed to be positive semi-definite
    :param rank: int or None, giving the rank of the low-rank component.
    :return: torch.tensor giving the largest rank magnitude eigenvectors
        (ntasks x rank) and a torch.tensor giving the residual of the covariance
         matrix's diagonal vs. low_rank * low_rank^T
    """
    if rank is None:
        rank = covariance_matrix.shape[0] - 1

    eigenvals, eigenvecs = covariance_matrix.detach().clone().eig(eigenvectors=True)

    # Sort both the eigenvectors and eigenvalues by the MAGNITUDE of the eigenvalues
    # Note that the eigenvalues should be >= real part, since the covariance
    # matrix is symmetrical.
    if any(eigenvals[:,1] > 0.0001):
        raise ValueError('Non-real eigenvalues!')
    _, indices = torch.sum(torch.pow(eigenvals, 2), dim=1).sort(descending=True)
    eigenvecs = eigenvecs[:, indices]
    eigenvals = eigenvals[indices, :]

    # This would have to be changed in the case that we had some imaginary or negative eigenvalues.
    truncated_eigvecs = torch.matmul(
        eigenvecs[:, :rank], torch.diag(torch.sqrt(eigenvals[:rank,0]))
    )
    # Compute the residual between the low-rank reconstruction and the original
    # covariance matrix on the diagonal
    residual_diag = (
            covariance_matrix
            - torch.matmul(
        truncated_eigvecs,
        truncated_eigvecs.transpose(-2, -1))
    ).diag()

    return truncated_eigvecs.float(), residual_diag.float()


# Define the multi-task model:
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 3 tasks
        # (so we'll actually learn 3x3=9 tasks with correlations)  # TD: ????
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks,
                                                              rank=num_tasks-1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


# Define the multi-task model:
class MultitaskVanillaGPModelDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, layer_sizes=None, deltas=False, num_tasks=2):
        super(MultitaskVanillaGPModelDKL, self).__init__(train_x, train_y, likelihood)

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
            covar = covar_x.mul(covar_i)  # TODO: Switch to ProductStructureKernel?
        else:
            covar = covar_x

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        try:
            model_out = super(MultitaskVanillaGPModelDKL, self).get_fantasy_model(inputs, targets, **kwargs)
        except RuntimeError:
            warnings.warn('Guarding failed get_fantasy_model call', RuntimeWarning)
            model_out = deepcopy(self)
            # TODO: Don't I want a model_out.set_train_data here?

        return model_out


class MultitaskLinPlusStationaryDKLGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, layer_sizes_lin=None, layer_sizes_stationary=None, deltas=False, num_tasks=2):
        super(MultitaskLinPlusStationaryDKLGP, self).__init__(train_x, train_y, likelihood)

        # Input handling:
        if layer_sizes_lin is None:
            layer_sizes_lin = []

        if layer_sizes_stationary is None:
            layer_sizes_stationary = []

        # Set up the linear transformations going into the linear kernel
        self.lin_layers_lin = []
        for i, size_i in enumerate(layer_sizes_lin):
            if i == 0:
                input_size = train_x[0].shape[1]
            else:
                input_size = layer_sizes_lin[i-1]
            self.lin_layers_lin.append(torch.nn.Linear(input_size, size_i))
        self.lin_layers_lin = torch.nn.ModuleList(self.lin_layers_lin)

        # Set up the linear transformations going into the stationary kernel
        self.lin_layers_stationary = []
        for i, size_i in enumerate(layer_sizes_stationary):
            if i == 0:
                input_size = train_x[0].shape[1]
            else:
                input_size = layer_sizes_stationary[i - 1]
            self.lin_layers_stationary.append(torch.nn.Linear(input_size, size_i))
        self.lin_layers_stationary = torch.nn.ModuleList(self.lin_layers_stationary)

        # Initialize all of the weights in the linear layers
        for lin_layer_group in [self.lin_layers_lin, self.lin_layers_stationary]:
            for i, lin_layer in enumerate(lin_layer_group):
                torch.nn.init.eye_(lin_layer.weight.data)
                if i == 0 and deltas:
                    print('Initializing with delta structure!')
                    # Assume that the predictor columns have been ordered such that
                    # the input is mutant_(feature_0), ..., mutant(feature_k),
                    # wt_(feature_0), ... wt(feature_k), i.e., that the
                    input_size = lin_layer.weight.shape[1]
                    assert input_size == 2 * math.floor(
                        input_size / 2.0)  # I.e., it's even
                    lin_layer.weight.data[:, :math.floor(input_size / 2.0)] = \
                        lin_layer.weight.data[:, torch.randperm(
                            math.floor(input_size / 2.0)
                        )]
                    lin_layer.weight.data[:, math.floor(input_size / 2.0):] = \
                        -1.0 * lin_layer.weight.data[:, :math.floor(input_size / 2.0)]
                lin_layer.weight.data = 0.05 * lin_layer.weight.data + 0.005 * torch.randn_like(
                    lin_layer.weight.data)
                lin_layer.zero_grad()

        # Set the number of tasks
        self.num_tasks = num_tasks

        # Set up the (dual kernel) GP on top
        self.mean_module = gpytorch.means.ConstantMean()

        # Initialize the dimension lists for input to the kernels
        if not layer_sizes_stationary:
            layer_sizes_stationary = [train_x[0].shape[1]]
        stationary_dims = [i for i in range(layer_sizes_stationary[-1])]

        if not layer_sizes_lin:
            layer_sizes_lin = [train_x[0].shape[1]]
        linear_dims = [i + stationary_dims[-1] + 1 for i in range(layer_sizes_lin[-1])]

        if self.num_tasks > 1:
            self.covar_module = \
                gpytorch.kernels.RBFKernel(
                    active_dims=torch.tensor(stationary_dims)
                ) +\
                gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(linear_dims)
                )

            # We learn an IndexKernel for num_tasks tasks
            self.task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=num_tasks, rank=num_tasks-1)

        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    active_dims=torch.tensor(stationary_dims)
                ) + \
                gpytorch.kernels.LinearKernel(
                    active_dims=torch.tensor(linear_dims)
                )
            )
            self.task_covar_module = gpytorch.kernels.RBFKernel()  # Shouldn't do anything

    def initialize_task_covar(self):
        est_covar_matrix = estimate_task_covariance_from_data(
            self.train_inputs[0], self.train_targets, self.num_tasks)
        truncated_eigvecs, diag_residual = low_rank_and_diag_decomp(
            est_covar_matrix, rank=self.num_tasks - 1
        )
        self.task_covar_module.raw_var = torch.nn.Parameter(
            torch.log(torch.exp(diag_residual) - 1.0)
        )
        self.task_covar_module.covar_factor = torch.nn.Parameter(
            truncated_eigvecs
        )

    def forward(self, x, i):
        xlin = x.clone()
        for idxi, lli in enumerate(self.lin_layers_lin):
            if idxi == len(self.lin_layers_lin):
                xlin = lli(xlin)
            else:
                xlin = torch.tanh(lli(xlin))

        xstationary = x.clone()
        for idxi, lli in enumerate(self.lin_layers_stationary):
            if idxi == len(self.lin_layers_stationary):
                xstationary = lli(xstationary)
            else:
                xstationary = torch.tanh(lli(xstationary))

        xall = torch.cat([xstationary, xlin], dim=1)

        mean_x = self.mean_module(xall)

        # Get input-input covariance
        covar_x = self.covar_module(xall)
        if self.num_tasks > 1:
            # Get task-task covariance
            covar_i = self.task_covar_module(i)
            # Multiply the two together to get the covariance we want
            covar = covar_x.mul(covar_i)  # TODO: Switch to ProductStructureKernel?
        else:
            covar = covar_x

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        try:
            model_out = \
                super(MultitaskLinPlusStationaryDKLGP,self).get_fantasy_model(
                    inputs, targets, **kwargs
                )
        except RuntimeError:
            warnings.warn('Guarding failed get_fantasy_model call',
                          RuntimeWarning)
            model_out = deepcopy(self)
            # TODO: Don't I want a model_out.set_train_data here?
        return model_out
