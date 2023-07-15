# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT


from improvwf.utils import get_history, get_history_db, yaml_safe_dump_with_lock, \
    append_single_record_to_db
from improvwf.db_interface.utils_sina import reformat_dict 
import os
from abag_ml.utils import  get_history_tensors_from_history
from abag_ml.constants import EXPERIMENTAL_COLUMN_NAMES, \
    EXPERIMENTAL_TYPES_INT_CODES, EXPERIMENTAL_TYPES_STR
import torch
import numpy as np
import csv
import pandas as pd
import copy
from DatasetTemplate import DatasetTemplate
import utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import pickle


predictor_columns=[
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


class DBDataset(DatasetTemplate):
    def __init__(self, DB_path, size, train_test_split, name=None, nrandom=None, max_cpu=32, shuffle=True, \
                normalize=False, use_cached=True, struct_hashes=[],start=0):
        """
        Attributes
        ----------
            name (string): name for the dataset
            size (int): size of dataset
            train_test_split (float (0,1): percentage to train and test
            filename (string): looks for this filename in tensors folder, no postfix for prefix filepath

        """
        self.DB_path = DB_path
        if not name:
            name = os.path.basename(self.DB_path)
            name = name.replace('.my.cnf','') 

        self.nrandom = nrandom
        self.max_cpu = max_cpu
        self.use_cached = use_cached
        self.struct_hashes = struct_hashes
        self.start = start


        super().__init__(name, size, train_test_split)
        if "datasets" in os.getcwd(): # testing from db dir
            prefix = os.path.join("tensors",name)
        else: # actually used from root dir
            prefix = os.path.join("datasets","tensors",name)
        self.prefix = prefix
        self.filepath_x = os.path.join(prefix, name + "_x.pt")
        self.filepath_y = os.path.join(prefix, name + "_y.pt")
        self.filepath_i =  os.path.join(prefix, name + "_i.pt")
        self.X = None
        self.y = None
        self.i = None
        self.size = size
        self.run_num = 0
        self.shuffle = shuffle
        self.normalize = normalize
        
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    @staticmethod
    def filter_for_struct(struct_hashes, history):
        inds = []
        final_history = copy.deepcopy(history)
        for entry in history['history']:
            curr_hash = history['history'][entry]['study_parameters']['STRUCTURE_HASH'][0][0] 
            if curr_hash not in struct_hashes:
                del final_history['history'][entry]
        return final_history

    def db_pull(self):
        type_col='StudyType'
        os.makedirs(self.prefix, exist_ok=True)

        basedir = self.prefix
        history_seqname_path = os.path.join(self.prefix, self.name + "_seqname.csv")
        history_structname_path = os.path.join(self.prefix, self.name + "_structname.csv")

        history = get_history_db(self.DB_path, nrandom=self.nrandom, max_cpu=1)
        # get history out of database format into format we get from yaml
        history = reformat_dict(history)
        if len(self.struct_hashes):
            print("filtering for : ", self.struct_hashes)
            history = self.filter_for_struct(self.struct_hashes, history)

        history_features_tensor, history_tasks, history_obs_val, history_ids, history_structids = \
        get_history_tensors_from_history(
                history, predictor_columns, EXPERIMENTAL_COLUMN_NAMES,
                EXPERIMENTAL_TYPES_INT_CODES, EXPERIMENTAL_TYPES_STR, type_col,
                include_seqid_structid=True
            )

        torch.save(history_features_tensor, self.filepath_x)
        torch.save(history_tasks, self.filepath_i)
        torch.save(history_obs_val, self.filepath_y)

        # ignores tensor dir
        with open(os.path.join(basedir, '.gitignore'), 'w') as f:
            f.write('*')

        with open(history_seqname_path, 'w', newline='') as f:
            w = csv.writer(f, delimiter=' ')
            for li in history_ids:
                w.writerow([li])

        with open(history_structname_path, 'w', newline='') as f:
            w = csv.writer(f, delimiter=' ')
            for li in history_structids:
                w.writerow([li])

        # Calculate and print some statistics
        # Unique (feature, task) tuples
        # Unique (feature, task, ID) tuples
        tmp = pd.DataFrame(data=history_features_tensor.numpy())
        cols_from_features = [ci for ci in tmp.columns]
        tmp['ID'] = history_ids
        tmp['structID'] = history_structids
        tmp['i'] = history_tasks.numpy()
        tmp['y'] = history_obs_val.numpy()

        # First, get just one row per combination of 'ID', 'structID', and 'i';
        # that is, task, structure, and sequence

        # Take the mean value over repetitions of the exact same study:
        print('Initially, there are {} retrieved history records.'.format(tmp.shape[0]))
        tmp = tmp.groupby(by=cols_from_features + ['i', 'ID', 'structID']).mean()

        # Assert that grouping by ['i', 'ID', 'structID'] gives counts of 1
        # only, that is, i, ID, structID ==> unique x.
        assert (tmp.reset_index().groupby(by=['i', 'ID', 'structID'])['y'].count() == 1).all(), \
            'Only one x may be present for a given (task, ID, structID) tuple!'


        # Goal: find "improper" collisions, that is, those where different (seq)
        # IDs collide, with or without the same structure.
        # First, squish out differences in structID:
        print('After grouping by (x, i, ID, structID) (where x is redundant), and '
                'prior to groupby x, i, ID, there are {} rows in '
                'tmp.'.format(tmp.shape[0]))
        tmp = tmp.reset_index().drop(columns=['structID']).groupby(
            by=cols_from_features + ['i', 'ID']).mean()
        print('After groupby x, i, ID, there are {} rows in tmp.'.format(tmp.shape[0]))

        # Now, count collisions in (x, i) where there are distinct IDs.
        tmp_grpd = tmp.reset_index().groupby(by=cols_from_features + ['i'])

        stats = pd.DataFrame().from_dict(data={
            'nIDstructID': tmp_grpd.count()['y'].values,
            'mean': tmp_grpd.mean()['y'].values,
            'std': tmp_grpd.std()['y'].values
        })
        print('Finally, there are {} unique combinations of feature vector and '
                'task ID'.format(stats.shape[0]))

        stats.to_csv(os.path.join(os.path.split(history_seqname_path)[0], 'collision_info.csv'))

    def prepare_data(self):
        if not self.use_cached or not os.path.isfile(self.filepath_x) or not os.path.isfile(self.filepath_y):
            print("Dataset not found .... pulling data")
            self.db_pull()
            
        self.X = torch.load(self.filepath_x)
        self.y = torch.load(self.filepath_y)
        self.i  = torch.load(self.filepath_i)

        self.X = self.X[self.start:self.start + self.size,:]
        self.y = self.y[self.start:self.start + self.size]
        self.i = self.i[self.start:self.start + self.size]
        
        if self.normalize:
            print("nomalizing data...")
            self.normalize_data()

    def normalize_data(self):
        self.y = torch.from_numpy(normalize(self.y.reshape(-1,1))[:,0]).to(dtype=torch.float)

    def shuffle_and_split(self):
        indices = range(len(self.X))
        self.X_train, self.X_test, self.y_train, self.y_test, indices_train, indices_test = \
        train_test_split(self.X,self.y,indices,test_size=self.train_test_split, random_state = 1, shuffle=self.shuffle)
        mapping = utils.mapVecsToTask([self.i])
        self.i_train = self.i[indices_train] #utils.iVecToTasks(self.i[indices_train], mapping)
        self.i_test = self.i[indices_test] #utils.iVecToTasks(self.i[indices_test], mapping)
        print('train :', len(self.X_train), 'test : ', len(self.X_test))
        
        self.run_num += 1
        print(np.unique(self.i_train)," --> task inds")
        print(len(np.unique(self.i_train))," --> num tasks in train")
        print(len(np.unique(self.i_train))," --> num tasks in test")
    
        self._X_train, self._y_train, self._i_train = self.X_train, self.y_train, self.i_train
        self._X_test, self._y_test, self._i_test = self.X_test, self.y_test, self.i_test
        

if __name__ == '__main__':
    #db_path = ***REDACTED***
    dataset = DBDataset(db_path, 1200, .2, use_cached=True)
    dataset.prepare_data()
