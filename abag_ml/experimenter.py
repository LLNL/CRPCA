# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import os
import types
import shutil
import json
import pandas as pd
from torch.utils import data
import seaborn as sns
import functools
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import copy
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'figure.autolayout': True})
import performance_metrics
import utils
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import pickle
from copy import deepcopy
from time import time
 

class ModelTraining(object):
    """
    class for model training
    steps:
        1. initialize with experiment name
        2. execute experiements
        3. report
    """
    def __init__(self, name, starting_run=None, clear_prev_dir=True):
        assert isinstance(name, str)
        self.name = name
        self.dataset = None
        self.methods = None
        self.metrics = None
        if starting_run :
            self.starting_run=starting_run
        else:
            self.starting_run = 0
        self.clear_prev_dir = clear_prev_dir

    def execute(self, dataset, methods, metrics, reports=[], nruns=1, validate_only=False):  # , report_only=False):
        """
        Args:
            dataset (DatasetTemplate): dataset class with prepare, shuffle methods
            methods (list): list of GPs exetending base class with fit and predict
            metrics (list): list of strings matching metrics in UTILs
            nruns (int): number of times to run all the models
        These parameters are checked and enforced
        """
        self.__check_inputs(dataset, methods, metrics, nruns, reports)
        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics
        self.reports = reports
        self.ntasks = None
        # this could cause upstream issues if anything is cast to float
        torch.set_default_tensor_type(torch.FloatTensor)
        self.nb_runs = nruns + self.starting_run 
        self.path_to_output = 'outputs'

        # set experiment output directory
        directory = os.path.join(self.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.islink(directory):
            os.unlink(directory)
        if os.path.exists(directory) and self.clear_prev_dir:
            shutil.rmtree(directory)
        # make a new directory with experiment name
        # print(os.getcwd(), "CURR DIR", directory)
        os.makedirs(directory,exist_ok=True)
        self.directory = directory

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        results_runs = dict()
        for r_i in range(self.starting_run , self.nb_runs):

            # shuffle and re-split the data between training and test
            self.dataset.shuffle_and_split()
            if hasattr(dataset, "max_tasks"):
                self.ntasks = dataset.max_tasks
                print(f"MAX TASKS : {self.ntasks}")

            run_directory = os.path.join(directory, 'run_{}'.format(r_i + 1))

            results_runs['run_{}'.format(r_i + 1)] = dict()
            # execute all methods passed through 'methods' attribute
            for method in self.methods:
                
                print("############# RUNNING METHOD: " + str(method.name) + "  #############")
                ################### FIT and PREDICT ###########################
                start = time()
                # check types
                utils.assertTorchCPU(self.dataset.X_train), utils.assertTorchCPU(self.dataset.y_train),  
                utils.assertTorchCPU(self.dataset.i_train)
                if not validate_only:
                    method.fit(self.dataset.X_train, self.dataset.y_train, i=self.dataset.i_train, ntasks=self.ntasks, X_corpus=self.dataset._X_test, \
                    i_corpus = self.dataset._i_test)
                else:
                    method.load_model(self.dataset.X_train, self.dataset.y_train, i=self.dataset.i_train, ntasks=self.ntasks)
                fit_end = time()

                # PREDICT
                y_pred, lower, upper = method.predict(dataset.X_test, i=dataset.i_test, y=dataset.y_test, ntasks=self.ntasks, return_std=True)
                end = time()
               
                ################### EVALUATE ##################################
                result_method = {}
                timing = {"total": end - start, "fit": fit_end - start, "predict": end - fit_end}
                print(timing)
                # cache data so we can make reports later
                self._cacheData(dataset, y_pred, lower, upper, timing ,method, r_i, directory)
                for met in self.metrics:
                    # checking return types
                    utils.assertTorchCPU(y_pred), utils.assertTorchCPU(dataset.y_test)
                    if "ppl" in met:
                        # get sample size and k from metrics 
                        met, post_samples, k = met.split("_")
                        post_samples = int(post_samples)
                        k = int(k)
                        model_posterior = method.get_predictive_posterior(X=dataset.X_test, i=dataset.i_test, mvn_only=True)
                        y_rep = model_posterior.sample_n(post_samples)
                        result_method[met] = metric_func[met](y_rep=y_rep.numpy(), R=dataset.y_test.numpy(), k=1)
                    result_method[met] = metric_func[met](y_pred.numpy(), dataset.y_test.numpy()).item()
                    results_runs['run_{}'.format(r_i + 1)][method.__str__()] = result_method
                    
        with open(os.path.join(directory, 'performance.json'), 'w') as fp:
            json.dump(results_runs, fp, indent=4, sort_keys=True)

    def __check_inputs(self, dataset, methods, metrics, nb_runs, reports):
        """
        Helper function to verify object types. Throws assert error if not
        
        Args:
            dataset (DatasetTemplate): pass in one of our cookie cuttter dataset templates to handle data
            methods (list): string list of methods to use, currently not checking they are valid, could be though
            metrics (list): string list of metrics to use, currently not checking they are valid, could be though
            nb_runs (int): number of times to run each method

        """
        # make sure all inputs have expected values and types

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)
        assert len(metrics) > 0

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            # handle ppl method case
            if "ppl" in metric:
                metric, _, _ = metric.split("_")
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert nb_runs > 0

    def generate_report(self):
        """
        There is a lot happening in this function and it is maybe even more important than 
        the main experiment loop. Pseudocode:
        1. first print out some basic metrics for the runs
        2. generate a report pdf 
        3. performance boxplot onto pdf (using UTIL/performance_report)
        4. example scatter plot onto pdf (using UTIL/performance_report)
        5. performance and plot from ABAGML (how necessary is this, and if so we need to clean up PDF)
        6. fantasyModelTest, testing how fast it is to impute based on new info. 
        Using custom GPytorch code definitely need to revisit this one
        7. time prediction elements (need to only call this in a certain case, 
        
        The idea is that this function can generate the whole report using python code tucked away in UTILS
        
        """
        # read results from experiment folder and store it into a dataframe
        df = self.__read_experiment_results()
        if hasattr(self.dataset, 'abbind_structure_filter') and self.dataset.abbind_structure_filter:
            df = self.add_benchmarks_to_abbind_df(df)
        print(df[['Method', 'Metric', 'Value']].groupby(['Method', 'Metric']).agg(['mean', 'std']))

        # set output pdf name
        pdf_filename = os.path.join(self.path_to_output,
                                    self.name,
                                    '{}_report.pdf'.format(self.name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        # call several plot functions
        ##################################
        # insert plot functions here     #
        ##################################
        for report_name in self.reports:
            for method in self.methods:
                report = getattr(performance_report, report_name) # trick to make string of report an actual function
                if report_name is not 'performance_boxplots':
                    report(self, pdf, method)
        if 'performance_boxplots' in self.reports:
            self.performance_boxplots(pdf, df)

        # close pdf file
        pdf.close()

    def performance_boxplots(self, pdf, df, **kwargs):
        """ Create boxplot with performance of all methods per metric.

        Args:
            df (pandas.DataFrame): Dataframe with runs/models/performance
            pdf (obj): pdf object to save the plot on
            
        """
        fig = plt.figure()
        for metric in self.metrics:
            df_a = df[df['Metric'] == metric].copy()
            g = sns.barplot(x="Method", y="Value", data=df_a)
            plt.title(metric)
            plt.ylabel("Error")
            plt.xticks(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.clf()
            
    def __read_experiment_results(self):
        """ Read in performance results from the json file.
        
        Returns:
            pandas.DataFrame: runs, methods, metrics, and perf in a DF
            
        """
        experiment_dir = os.path.join(self.path_to_output, self.name)
        with open(os.path.join(experiment_dir, 'performance.json'), 'r') as fh:
            data = json.load(fh)

        perf_list = list()
        for run in data.keys():
            for method in data[run].keys():
                for metric in data[run][method].keys():
                    value = data[run][method][metric]
                    perf_list.append([run, method, metric, value])
        column_names = ['Run', 'Method', 'Metric', 'Value']
        df = pd.DataFrame(perf_list, columns=column_names)
        return df
         
    def _cacheData(self, dataset, predictions, lower, upper, timing, method, run_num, directory):
        """
        Stores data in pkl file for later retrieval
        Args:
            dataset (DatasetTemplate): common datatset object to be expected
            predictions (np.array): 1d array of predictions, now back on cpu in numpy
            timing (dict): should be a dict of start and stop times
            method (GPBase): baseGP object, we really just save the name
            run_num (int): what run was this?
            directory (string); filepath-- where do we want to save this?
            
        DICTIONARY/PKL KEYS Reference 
            X_train
            y_train
            i_train
            X_test
            y_test
            i_test
            y_pred
            method
            run_num
            timing
            modelClass
            --- optionally ---
            train partition 
            test partition
            
        """
        res = {'X_train': dataset.X_train, 'y_train': dataset.y_train, 'i_train': dataset.i_train, \
               "X_test": dataset.X_test, "y_test": dataset.y_test, "i_test": dataset.i_test, \
               "y_pred": predictions, "method": method.name, "run_num": run_num, "timing": timing,\
               "modelClass": str(type(method.model).__name__)}
        # TODO: refactor out later
        res['lower'] = lower
        res['upper'] = upper

        if hasattr(dataset, "train_partition"):
            res['train_partition'] = dataset.train_partition
            res['test_partition'] = dataset.test_partition
            res['train_partition_sizes'] = dataset.train_sizes
            res['test_partition_sizes'] = dataset.test_sizes
            res['struct_partition'] = dataset.struct_partition

        allruns = os.path.join(directory, "data")
        if not os.path.exists(allruns):
            os.makedirs(allruns)
        run_directory = os.path.join(allruns, "run" + str(run_num))
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)
        filename = os.path.join(run_directory, str(method) + '.pkl')
        with open(filename, 'wb') as pkl_file:
            pickle.dump(res,pkl_file )
        try:
            method.save_model(run_directory)
        except AttributeError:
            # TODO: this is not the job of utils
            utils.fakeSaveModel(method, run_directory)
            print("not saving : {} is it a benchmark?? if so ... OK".format(str(method)))
        