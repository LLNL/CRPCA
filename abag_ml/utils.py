# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""

A module containing utility functions used by the abag_ml package

"""
from datetime import datetime, timedelta
import re
import yaml
import time
import os
import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser

from improvwf.utils import read_run_descriptors_history

from abag_agent_setup.expand_allowed_mutant_menu import \
    write_selected_study_abag as write_selected_study
from abag_agent_setup.expand_allowed_mutant_menu import \
     consolidate_studies_to_write
from abag_agent_setup.decision_making import \
    get_feature_representation_from_history_or_menu

from abag_ml.constants import PRED_FEATURE_SCALING

from vaccine_advance_core.featurization.assay_utils import \
    calculate_dG_from_octet_Kd

_SEQID_COL = 'AntigenID'
_STRUCTID_COL = 'Complex'

def print_parameters(model, likelihood):
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


def get_layer_sizes(state_dict):
    """
    Using a torch state_dict, obtain the lin_layers sizes

    :param state_dict: PyTorch state_dict
    :return: dictionary of lists of layer sizes (ints)
    """

    # TODO: Update for the case where we have two separate streams of linear layers

    re_matcher = '(lin_layers[_a-zA-Z]*)\.([0-9]+)\.bias'
    layers = {}
    for ki, vi in state_dict.items():
        v = re.fullmatch(re_matcher, ki)
        if not v:
            continue
        if not v.group(1) in layers.keys():
            layers[v.group(1)] = {}
        layers[v.group(1)][int(v.group(2))] = vi.shape

    layer_sizes = {
        ki: [vi[j][0] for j in sorted(list(vi.keys()))]
        for ki, vi in layers.items()
    }
    return layer_sizes


def get_num_tasks(state_dict):
    """
    Using a torch state_dict, obtain the size of the task covariance kernel

    :param state_dict:
    :return: int, giving the number of tasks
    """
    return int(state_dict['task_covar_module.covar_factor'].shape[0])


def compress_history_observations(history_obs_df):
    """
    Take several history observation types and compress to a single column

    :param history_obs_df: DataFrame with k columns, where each column has a
        single type of historical observation or none
    :return: np.array containing a single column, where if the study is pending
        or failed, the value is NaN.
    """

    history_obs_val = history_obs_df.values  # Convert to np.array
    # Squish the values into one column, where target type is no longer encoded;
    # df['StudyType'] will be the kernel input encoding this relationship.
    # Verify that one and only one output is present per row:
    check = np.all(np.less_equal(
        (history_obs_val.shape[1] - 1) * np.ones((history_obs_val.shape[0],), dtype=np.int),
        np.sum(np.isnan(history_obs_val), axis=1)
    ))
    if not check:
        raise ValueError('The history has multiple target types for a single row/study in the history!')
    return np.nansum(history_obs_val, axis=1)



def get_history_tensors_from_history(
        history, predictor_columns, experimental_column_names,
        experimental_types_int_codes, experimental_types_str, type_col,
        include_seqid_structid=False):
    if history['history']:  # i.e., if there are any history entries:
        history_df, _ = get_feature_representation_from_history_or_menu(history,
                                                                        cpx_dictionary=None)
        history_obs_val = get_target_values(
            history_df,
            desired_targets=experimental_column_names
        )

        if history_obs_val.shape[1] == 0:
            print('All observations pending! Writing out empty tensors.')
            history_tasks = torch.empty((0,), dtype=torch.long)
            history_features_tensor = torch.empty((0, 1),
                                                  dtype=torch.float)  # The horizontal dimension needs to be changed
            history_obs_val = torch.empty((0,), dtype=torch.float)
            history_seqid = []
            history_structid = []

        else:
            print('Some observations found.')
            # Compress observations to a single column
            history_obs_val = compress_history_observations(
                history_obs_val).reshape((-1, 1)).astype(np.double)

            # The above is a np.array with a single column.  Use it to delete rows
            # corresponding to pending or failed observations. Note that this means
            # that we will be treating these as if they did not exist. In principle,
            # this is wrong, since knowledge of concurrently running experiments
            # should be useful to avoid redundancy. However, this avoids the case of
            # dealing with zombie studies and does not require disambiguating (at
            # this point, rather than at an earlier point in the history loading)
            # between RUNNING and FAILED studies.
            # TODO: Use RUNNING studies' information, disambiguating from FAILED
            print(
                'Prior to removing running studies, the shape of history_obs_val is:')
            print(history_obs_val.shape)
            history_df = history_df.loc[
                np.logical_not(np.isnan(history_obs_val.reshape((-1,))))]
            history_obs_val = history_obs_val[
                np.logical_not(np.isnan(history_obs_val))]

            print('After removal, the shape of history_obs_val is:')
            print(history_obs_val.shape)
            history_obs_val = torch.tensor(history_obs_val,
                                           dtype=torch.float).reshape((-1,))
            history_df[type_col] = [
                experimental_types_int_codes[experimental_types_str.index(vi)]
                for vi in history_df[type_col].values
            ]
            # Tensor-ize tasks for use below
            history_tasks = torch.tensor(
                history_df[type_col].values.astype(np.double), dtype=torch.long)
            history_features_tensor = \
                PRED_FEATURE_SCALING * torch.tensor(
                    history_df[predictor_columns].values,
                    dtype=torch.float)
            history_seqid = [id for id in history_df[_SEQID_COL]]
            history_structid = [id for id in history_df[_STRUCTID_COL]]
    else:
        history_tasks = torch.empty((0,), dtype=torch.long)
        history_features_tensor = torch.empty((0, 1),
                                              dtype=torch.float)  # The horizontal dimension needs to be changed
        history_obs_val = torch.empty((0,), dtype=torch.float)
        history_seqid = []
        history_structid = []

    if include_seqid_structid:
        return history_features_tensor, history_tasks, history_obs_val, history_seqid, history_structid
    else:
        return history_features_tensor, history_tasks, history_obs_val

def get_target_values(history_df, desired_targets=None):
    """
    For each row in history_df, obtain the specified target values

    :param history_df: n x m dataframe containing the features and result
        values for the Maestro studies already run.
    :param target_values: list of k desired columns, str, or None; if none,
        defaults to the empty list. If a str, becomes a 1-element list with
        that str.
    :return: DataFrame (n x k) with columns corresponding to the listed
        target_values, with one row corresponding to each input row
    """

    if desired_targets is None:
        desired_targets = []
    elif isinstance(desired_targets, str):
        desired_targets = [desired_targets]
    if not isinstance(desired_targets, list):
        raise ValueError('desired_targets must be a list!')

    tv_out = []
    for tvi in desired_targets:
        # Switching conditions
        try:
            if tvi.lower() == 'foldx_interface_ddg':
                col = history_df['FoldXInterfaceDG'] \
                      - history_df['WT_FoldXInterfaceDG']  # pd.Series
            elif tvi.lower() == 'statium_ddg':
                col = history_df['inter'] \
                      + history_df['intra'] \
                      - history_df['master_inter'] \
                      - history_df['master_intra']  # pd.Series
            elif tvi.lower() == 'octet_ddg':
                col = pd.Series(
                    calculate_dG_from_octet_Kd(history_df['KD (M)'].values),
                    name=tvi, index=history_df.index)  # TODO: Untested: debug
            elif tvi.lower() == 'rosetta_fl_ddg':
                col = history_df['RosettaFlexDDGAvg']
            else:
                raise ValueError('Unrecognized observation type {}!'.format(tvi))
            # We've recognized the name and calculated the value
        except (ValueError, KeyError):
            print('Unable to compute observation type {}!'.format(tvi))
            col = pd.Series(np.nan * np.ones((history_df.shape[0], ), dtype=np.float), name=tvi, index=history_df.index)
        col.name = tvi
        tv_out.append(col)

    if tv_out:
        return pd.concat(tv_out, axis=1)
    else:
        return pd.DataFrame().reindex_like(history_df).drop(columns=history_df.columns)


def remove_from_menu_with_return_removed(expanded_menu, list_of_history_studies):
    """
    Remove previously-executed studies from expanded, discrete-valued menu
    :argument expanded_menu: The dictionary containing the set of all allowed
        studies (for this worker) in its "studies" field.
    :argument list_of_history_studies: list containing previously executed
        studies, which will be removed from expanded_menu.
    :return expanded_menu: the input menu, with the previously-executed
        studies removed.
    :return expanded_menu_keep: a boolean list of which items by index have
        been removed (False) or kept (True)
    """
    expanded_menu_keep = [True for mi in expanded_menu["studies"]]
    for si in list_of_history_studies:
        # run_parameters = {ki: si[ki] for ki in si
        #                   if ki not in ["request_id", "status", "result"]}
        run_descriptors = read_run_descriptors_history(si)
        for j, sj in enumerate(expanded_menu["studies"]):
            if expanded_menu_keep[j] is False:
                continue
            if run_descriptors == sj:
                expanded_menu_keep[j] = False
    
    expanded_menu["studies"] = [emi for emi, ki in
        zip(expanded_menu["studies"], expanded_menu_keep) if ki]
    
    return expanded_menu, expanded_menu_keep

def write_all_selected_studies(studies_to_write, studies, improv_inbox, fastas_directory):
    """
    Consolidate and write out studies selected in the decision-making process

    :param studies_to_write: list of studies (dictionaries) as they appear in
        the expanded menu.
    :param studies: a dictionary of template studies, keyed by description:name
        (i.e., study_type), obtained via get_studies(studies_path).
    :param improv_inbox: str, giving the path to the improv inbox to which the
        completed studies will be written
    :param fastas_directory: str, giving the path to which the required fastas
        will be written
    """
    studies_to_write = consolidate_studies_to_write(
        studies_to_write, study_types=['statium'], must_match=['structure_hash']
    )

    for study_dict_i in studies_to_write:
        time.sleep(1.0)
        write_selected_study(
            study_dict_i['parameters'],
            studies,
            study_dict_i['request_id'],
            output_path=improv_inbox,
            fasta_path=os.path.abspath(os.path.join(
                fastas_directory, 'mutants_{}.fasta'.format(
                    study_dict_i['request_id']
                )))
        )

def parse_slurm_allocation_duration(allocation_duration_string):
    """
    Parse a SLURM-formatted allocation walltime string

    See man sbatch and the --time argument.

    :param allocation_duration_string: str, formatted as one of:
        "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours",
        "days-hours:minutes", or "days-hours:minutes:seconds"
    :return: datetime.timedelta equivalent to the string

    >>> parse_slurm_allocation_duration('4:00:05')
    datetime.timedelta(0, 14409)
    >>> parse_slurm_allocation_duration('0:05')
    datetime.timedelta(0, 5)
    >>> parse_slurm_allocation_duration('5')
    datetime.timedelta(0, 300)
    >>> parse_slurm_allocation_duration('5-20')
    datetime.timedelta(5, 72000)
    """

    if not '-' in allocation_duration_string:
        # No days present: allowed formats are  "minutes",
        # "minutes:seconds" "hours:minutes:seconds"
        _ = re.split(':', allocation_duration_string)
        if len(_) == 1:
            _ = _ + ['00'] # Shift from minutes to minutes:seconds
        _.reverse()
        allocation_duration = {ki: float(vi) for vi, ki in zip(_, ['seconds', 'minutes', 'hours'])}
    else:
        # Days are present, specified as a leading dd-HH:MM:SS
        # Allowed formats: "days-hours", "days-hours:minutes" and
        # "days-hours:minutes:seconds"
        _ = re.split('[-:]', allocation_duration_string)
        allocation_duration = {ki: float(vi) for vi, ki in zip(_, ['days', 'hours', 'minutes', 'seconds'])}
    return timedelta(**allocation_duration)


def check_allocation_suff_rem_time(timing, min_time_remaining_to_start):
    if timing and min_time_remaining_to_start:
        if isinstance(min_time_remaining_to_start, str):
            min_time_remaining_to_start = \
                parse_slurm_allocation_duration(min_time_remaining_to_start)

        # Load the timing file
        with open(os.path.abspath(timing), 'r') as f:
            timing_dict = yaml.safe_load(f)

        allocation_start_time = datetime.strptime(timing_dict['ALLOCATION_START_TIME'], '%Y%m%d_%H%M%S')
        # Above: Created in the invocation of this worker
        allocation_duration = timing_dict['ALLOCATION_DURATION']
        # Above: SLURM-formatted string: see parse_slurm_allocation_duration
        allocation_duration = parse_slurm_allocation_duration(allocation_duration)

        time_remaining = allocation_start_time + allocation_duration - datetime.now()

        # Compare the time_remaining with the configured min_time_remaining;
        # if less than min_time_remaining_to_start, terminate decision study

        if time_remaining < min_time_remaining_to_start:
            print('The time remaining in the allocation is less than the '
                  'minimum allowed time for new study submission. The decision'
                  ' study will therefore decline to submit new studies.')
            return False
        else:
            print('Time remaining {} exceeds the minimum allowed ({}); '
                  'attempting to select studies.'.format(
                str(time_remaining), str(min_time_remaining_to_start)
            ))
            return True
    return True

def setup_argparser(database=True):
    """
    Set up an argparser and return it

    Expected arguments are:
    -y $(IMPROV_HISTORY)
    -b $(IMPROV_DATABASE)
    -m $(IMPROV_MENU)
    -s $(IMPROV_STUDIES)
    -i $(IMPROV_INBOX)
    -f <path to fasta directory>
    -n <number of studies to submit>
    -c: <path to a configuration file>
    :return: parser
    """

    parser = ArgumentParser()
    parser.add_argument('-y', '--history', type=str,
                        help='Path to history file')
    if(database):
        parser.add_argument("-b", "--database_url", type=str,
                            help="URL of a study requests database.")
    parser.add_argument('-m', '--menu', type=str,
                        help='Path to the menu')
    parser.add_argument('-s', '--studies', type=str,
                        help='Path to studies')
    parser.add_argument('-i', '--inbox', type=str,
                        help='Path to the inbox')
    parser.add_argument('-f', '--fastas', type=str,
                        help='Path to the FASTA directory')
    parser.add_argument('-n', '--nsubmit', type=str,
                        help='Integer number of studies to submit')
    parser.add_argument('-t', '--timing', type=str,
                        help='Path to the timing file, containing information '
                             'on when the allocation starts',
                        default='')
    # Everything else comes from the config file
    # These arguments are:
    # nmutpermaster
    # model_state_path
    # predictor_columns
    # type_col
    # target_ab_master_and_structures
    # previous_model_data
    parser.add_argument('-c', '--config', type=str,
                        help='Path to a .yaml configuration file')
    return parser

def setup_argparser_minimal():
    """
    Set up an minimal argparser and return it

    Expected arguments are:
    -h $(IMPROV_HISTORY)
    -f <path to fasta directory>
    -n <number of studies to submit>
    -c: <path to a configuration file>

    :return: parser
    """

    parser = ArgumentParser()
    parser.add_argument('-y', '--history', type=str,
                        help='Path to history file')
    parser.add_argument('-n', '--nsubmit', type=str,
                        help='Integer number of antigens to submit')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to a .yaml configuration file')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to the output fasta file or directory')
    return parser

def get_parsed_args_from_dict(args_dict):

    arg_list = [
        '-m', args_dict['menu'],
        '-s', args_dict['studies'],
        '-i', args_dict['inbox'],
        '-f', args_dict['fastas'],
        '-n', args_dict['iterations'],
        '-c', args_dict['config']
    ]

    if 'database_url' in args_dict:
        arg_list.append('-b')
        arg_list.append(args_dict['database_url'])
    elif 'history' in args_dict:
        arg_list.append('-y')
        arg_list.append(args_dict['history'])
    else:
        raise ValueError('Must provide either a history yaml or database URL/file to arg parser!')

    print("\nParsing arguments:")
    for i in range(0,len(arg_list),2):
        print(arg_list[i] + " " + arg_list[i+1])

    parser = setup_argparser()
    return parser.parse_args(arg_list)


def assertTorchCPU(tensor):
    assert (torch.is_tensor(tensor)), "data and predictions should be tensors, maybe it is a numpy array?"
    assert (str(tensor.device) == "cpu"), "prediction tensor needs to be returned in cpu mode not " + str(tensor.device)

def fakeSaveModel(method, path):
    """
    fake save a trained model (benchmark)
    Args:
        method : benchmark model 
        path (str): path to save
    """
    
    path = os.path.join(path, method.name)
    f = open(path, 'w')
    f.write("Blank")
    f.close()

def mapVecsToTask(vecs):
    cat_vec = torch.cat(vecs)
    keys = torch.unique(cat_vec)
    mapped_vals = torch.Tensor(range(len(torch.unique(cat_vec)))).long()
    mapping = dict(zip(keys, mapped_vals))
    return mapping

def iVecToTasks(i_vec, mapping):
    """
    Helper function for having fidelities like [0,0,0,3,3,3] --> [0,0,0,1,1,1] 
    """
    for real_ind in mapping:
        i_vec = torch.where(i_vec == real_ind, mapping[real_ind] , i_vec)
    return i_vec


def export_method_and_data(method, exp_name, run=0 , source_path=None, save_path=None, save_name=None):
    """
    method used to save model in training  playground so it can be used by 
    CLS

    Args:
        method : benchmark model 
        exp_name (str):  name of experimenter object
        run (int): number of training run
        source_path (str, optional): path of model being saved
        save_path (str, optional): path to save to, default is outputs
        save_name (str, optional): name to save as, default is model name

    """
    # method.save_model('./') # 10052021_rosetta_model.pth
    if source_path:
        load_path = f'{source_path}/{method.name}.pkl'
    else:
        load_path = f'outputs/{exp_name}/data/run{run}/{method.name}.pkl'
    with open(load_path, 'rb') as f:
      prev_data = pickle.load(f)
    # convert to format to be read by abag_ml # TODO: adapt this format so this conversion not necessary
    curr_data = {}
    curr_data['train_x'] = prev_data['X_train']
    curr_data['test_x'] = prev_data['X_test']
    curr_data['train_y'] = prev_data['y_train']
    curr_data['train_i'] = prev_data['i_train']
    curr_data['test_y'] = prev_data['y_test']
    curr_data['test_i'] = prev_data['i_test']
    curr_data['test_x'] = prev_data['X_test']
    if not save_name:
        save_name = exp_name

    if save_path:
        pkl_save_path = os.path.join(save_path, f'{save_name}_{method.name}.pkl')
    else:
        pkl_save_path = f'saved_models/saved_model_data/{save_name}_{method.name}.pkl'
    
    with open(pkl_save_path, 'wb') as f:
      pickle.dump(curr_data, f)

    # copy model over
    if source_path:
        src_path = f'{source_path}/{method.name}'
    else:
        src_path = f'outputs/{exp_name}/data/run{run}/{method.name}'

    if save_path:
        dest_path = os.path.join(save_path, f'{save_name}_{method.name}')
    else:
        dest_path = f'saved_models/{save_name}_{method.name}'
    

    shutil.copyfile(src_path, dest_path)