# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import sys
import os
from DBDataset import DBDataset
from MultitaskVanillaGPModelDKL import MultitaskVanillaGPModelDKL
import utils
from experimenter import ModelTraining
import shutil


if __name__ == '__main__':

  db_path = "**REDACTED**"
  # dataset = DBDataset(db_path, 20000, .2, name='REDACTED', use_cached=True)
  dataset = DBDataset(db_path, 200, .2, name='example_data', use_cached=True)

  dataset.prepare_data()
  methods = [MultitaskVanillaGPModelDKL(num_iters=1)]

  metrics = ['rmse', 'mae', 'spearman_only']
  exp_folder = os.path.basename(__file__.replace('.py',''))
  exp = ModelTraining(exp_folder)
  exp.execute(dataset, methods, metrics, nruns=1)
  exp.generate_report()

  # saves model and data to location on filesysytem
  # utils.export_method_and_data(methods[0], exp_folder, save_name='variant_model')




    