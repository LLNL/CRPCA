# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import sklearn.metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import torch
#from sklearn.metrics import ndcg_score



def rmse(pred, R):
    """
    calculate l2 norm of predicted and actual free energy
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = np.sqrt((1 / len(rvals)) * np.sum((predvals - rvals) ** 2))
    return err

def r2(pred, R):
    """
    calculate coefficient of determination
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.r2_score(rvals, predvals)
    return err

def explained_variance_score(pred, R):
    """
    calculate explained variance 
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    inds = np.where(~np.isnan(R))
    err = 0
   
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.explained_variance_score(rvals.reshape(-1,1), predvals.reshape(-1,1))
    return err

def mae(pred, R):
    """
    calculate mean absolute error
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    inds = np.where(~np.isnan(R))
    err = 0
    if len(inds[0]) > 0:
        rvals = R[inds]
        predvals = pred[inds]
        err = sklearn.metrics.mean_absolute_error(rvals, predvals)
    return err

def spearman(pred, R):
    """
    calculate spearman correlation
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    return stats.spearmanr(R, pred)

def spearman_only(pred, R):
    """
    calculate spearman correlation
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    return stats.spearmanr(R, pred)[0]

def spearman_only_torch(pred, R):
    """
    calculate spearman correlation for TORCH
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    return torch.Tensor([stats.spearmanr(R, pred)[0]])

def pearson(pred, R):
    """
    calculate spearman correlation
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    return stats.pearsonr(R, pred)

def regret(y_vals, selected_inds):
    """
    calculate simple regret abs(argmin(y) - y[selected_inds] )
    Args:
        pred: vector of predicted free energy
        R: actual vector of free energy
    """
    # optimal_set = np.argmin(y_vals)
    # return np.sum(np.abs(y_vals[optimal_set] - y_vals[selected_inds]))
    return np.abs(np.min(y_vals) - np.min(y_vals[selected_inds]))

def ppl(y_rep, R, k=1):
    """
    calculate Gelfand and Ghosh Posterior predictive loss criterion
    Args:
        y_reps: matrix of responses drawn from the posterior predictive distribution
        R: actual vector of free energy
        k: penalty to the goodness of fit term, k=0 ignores the goodness of fit.
    """
    # import pdb; pdb.set_trace()
    g = np.sum((y_rep.mean(axis=0)-R)**2)
    p = np.sum(y_rep.std(axis=0))
    ppl = ((k/(k+1))*g)+p
    return ppl