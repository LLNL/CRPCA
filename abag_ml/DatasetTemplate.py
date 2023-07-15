# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from abc import ABCMeta, abstractmethod


class DatasetTemplate(Dataset):
    """
    Base Class for Dataset

    Parameters
    ----------- 
    name: string
        name for the dataset
    size: int
        size of dataset
    train_test_split: float (0,1)
        percentage to train and test

    Attributes
    ----------
    name: str
        Total # of training pts.
    size: int
        Total # of testing pts.
    train_test_split: int
        Total size of dataset.

    Methods
    -------
    prepare_data()
        Prepares all data for all specified partitions by calling x_y_i_from_partition. 
    shuffle_and_split()
        Selects random subsets of partitions based on partition sizes. Sets these values
        to be self.X_train or self.i_test for example. (see class attr. above)
    
    """
    
    __metaclass__ = ABCMeta

    
    def __init__(self, name, size, train_test_split):
        self.name = name
        self.size  = size
        self.train_test_split = train_test_split 
        
    @abstractmethod
    def __len__(self):
        pass 
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
        
    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def shuffle_and_split(self):
        pass

    def remove_acquired(self, inds):
        X_test = self.X_test[~inds]
        y_test = self.y_test[~inds]
        i_test = self.i_test[~inds]
        return X_test, y_test, i_test