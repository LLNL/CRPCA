# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""

A Gaussian process, information-gain + exploitation method that seeks to
maximize a target-fidelity response function

Overall, needs troubleshooting/improvement in the following areas:

1. Robustness, e.g. column naming conventions, etc.
2. Speed: repeated data type conversions are probably a big slow-down.
3. Improved models (performance)
4. Use of more advanced GP models (speed & data size).
5. Calculation of the MVN posteriors is slow and/or unstable in many cases; the
    explore score is sometimes NaN or negative, which should never occur.

"""
import sys
from datetime import datetime, timedelta
from copy import deepcopy
import os
import time
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.constraints import Interval
import re
import warnings

# Import some improv tools
from improvwf.utils import get_menu, get_studies, remove_from_menu, \
    remove_from_menu_db

# Import the abag_agent_setup tools for menu handling
from abag_agent_setup.expand_allowed_mutant_menu import \
    expand_menu_study_params_master_antigen_structures_mutations \
        as expand_menu_study_params

from abag_agent_setup.expand_allowed_mutant_menu import get_history, \
    get_history_db, study_query_data_for_expanded_menu
from abag_agent_setup.decision_making import \
    get_feature_representation_from_history_or_menu, \
    get_target_features_from_history_or_menu

# Import the vaccine_advance_core tools for various manipulations on models and
# data
from vaccine_advance_core.featurization.utils import get_datetime
from vaccine_advance_core.featurization.distance_penalty import compute_pam30_penalties
from vaccine_advance_core.featurization.seq_to_features import diff_seqs

# Import model classes and tools for loading model parameters
from abag_ml.utils import get_layer_sizes, get_num_tasks, \
    print_parameters, remove_from_menu_with_return_removed
from abag_ml.models_gpytorch import MultitaskVanillaGPModelDKL, \
    MultitaskLinPlusStationaryDKLGP
from abag_ml.utils import setup_argparser, write_all_selected_studies, \
    get_history_tensors_from_history, check_allocation_suff_rem_time
from abag_ml.constants import PRED_FEATURE_SCALING, \
    TRIM_PRETRAIN_SIZELIM_HIST_PRETRAIN, PREDICTOR_COLUMNS, \
    EXPERIMENTAL_COLUMN_NAMES, EXPERIMENTAL_TYPES_INT_CODES, \
    EXPERIMENTAL_TYPES_STR




def _remove_for_idx_geq_num_tasks(x, idx, y, num_tasks, name=None):
    """
    Given a set of tensors, remove those elements for which idx >= num_tasks

    :param x: torch.tensor, shape (n, d), containing features
    :param idx: torch.tensor of dtype long, shape (n, ), containing task indices
    :param y: torch.tensor, shape (n, ) containing response values
    :param num_tasks: int, giving the number of tasks the model expects
    :return: x, idx, y tensors, censored to remove places idx >= num_tasks.
    """
    if name is None:
        name = 'Incognito'

    if (idx >= num_tasks).any():

        print(
            'In tensor set {}, eliminating {} observations from the '
            'previous_model_data because the observation types are not '
            'currently compatible with the model!'.format(
                name, (idx >= num_tasks).sum().item()
            )
        )
        x = x[idx < num_tasks, :]
        y = y[idx < num_tasks]
        idx = idx[idx < num_tasks]

    return x, idx, y


def compute_conditional_mi(covariance_matrix_with_study_first, noisevar=0.0):
    """
    Computes mutual information between study and targets

    Joint distribution given as a GPyTorch MultivariateNormal in which row 0 is
    the study and all other rows are targets.

    :param covariance_matrix_with_study_first: covariance matrix from a
        MultivariateNormal of size > 1. Gives the joint distribution of the
        candidate observation and its "cognate" targets.
    :param noisevar: float giving the variance of the noise.
    :return: float giving the conditional mutual information between the
        candidate study and the targets.
    """

    if isinstance(noisevar, torch.Tensor):
        noisevar = noisevar.clone().detach().float()

    prior_var = torch.max(torch.zeros((1,)), covariance_matrix_with_study_first[0,0]) + noisevar  # What if cov matrix has negatives on diagonal; shouldn't, but I've seen it happen
    try:
        posterior_var = prior_var - \
                        gpytorch.inv_quad(
                            covariance_matrix_with_study_first[1:, 1:],
                            covariance_matrix_with_study_first[0, 1:]
                        )
        inf = -0.5 * (posterior_var.log() - prior_var.log()).float()  # there is some connection between target and candidate study
    except RuntimeError:
        # This is necessary because the (Cholesky) decomp can fail if the
        # covariance matrix is ~ singular
        warnings.warn('Guarding failed calculation of menu/target mutual information!', RuntimeWarning)
        print('Covariance matrix in failed CMI calculation:')
        print(covariance_matrix_with_study_first)
        inf = torch.tensor(0.0, dtype=torch.float)
        # Assume failure is problematic, and we should not give exploratory
        # value to this observation

    # Protect against numerical problems via the following limits:
    # TODO: Fix ranging; sometimes getting NEGATIVES or NaNs; upper lim in case variances are wrong (e.g., negative)?
    inf_upper_lim = -0.5 * (torch.log(noisevar) - prior_var.log()).float()  # target determines candidate study results up to the observation noise
    inf_lower_lim = 0.0  # target and candidate study are conditionally independent
    v = np.min([np.max([inf, inf_lower_lim]), inf_upper_lim])
    if v < 0:
        print('v < 0!')
    return v


def get_target_rows(target_df, study_antigen_sequence, target_antibody_ids):
    """
    Given an antigen, find the target-fideilty 'studies' that correspond with it
    :param target_df:  DataFrame, indexed by AntigenSequence and AntibodyID,
        that contains the target-fidelity feature vectors.
    :param study_antigen_sequence: str, giving the antigen sequence.
    :param target_antibody_ids: list of n str, where each is the AntibodyID for a
        single, target antibody.
    :return: DataFrame containing only the n rows corresponding to the targets.
    """

    rows = []
    for abi in target_antibody_ids:
        rows.append(target_df.loc[[(study_antigen_sequence, abi)]])  # putting the tuple inside [] means that we return a DataFrfame
        assert not isinstance(rows[-1], pd.Series), 'Must return 1xk DataFrames'

        # Above: could use to_frame().transpose(), but would coerce data types.
        # Doing .to_frame().transpose() would ensure that the concatenated
        # objects are 1-row, k-column DataFrames, yielding a n-row,
        # k-column DataFrame below
    return pd.concat(rows, axis=0)


def get_dec_rule_val(row_i, joint_mvn_mean, joint_mvn_covariance, type_col, lincoeffs_target_means_by_ab, noisevar, cost_dict):

    # Get the weighted exploitation score:
    exploit = joint_mvn_mean[1:].dot(torch.tensor(lincoeffs_target_means_by_ab,
                                                  dtype=torch.float))  # TODO: check syntax
    # pred_mean.append(joint_mvn.mean[0])
    # pred_std.append(joint_mvn.stddev[0])

    # Get the weighted exploration score:
    # print('Joint covariance matrix of study of type {} and target:'.format(
    #     row_i.iloc[0][type_col]))
    # print(joint_mvn.covariance_matrix)
    cmi = compute_conditional_mi(joint_mvn_covariance, noisevar)
    # print('Calculated CMI value of {}'.format(cmi))
    explore = cmi / float(cost_dict[row_i.iloc[0][type_col]])

    return explore, exploit, joint_mvn_mean[0], np.sqrt(joint_mvn_covariance[0, 0])


def compute_menu_scores_target_performance_and_cmi_monolithic(
        menu_df, target_df, model, predictor_cols, type_col, noisevar,
        target_ab_ids=None, lincoeffs_target_means_by_ab=None, cost_dict=None
):
    rows = []
    target_rows = []
    antigen_sequences = []
    pam30_penalty_values = []
    for idx in range(menu_df.shape[0]):
        row_i = menu_df.iloc[[idx]]
        # Use brackets to ensure the returned slice is a DF
        # Get the antigen sequence from row_i
        # Assumes that the menu_df is NOT indexed by this value, and instead,
        # it's in the row
        antigen_sequence = row_i.iloc[0]['AntigenSequence']
        antigen_sequences.append(antigen_sequence)
        pam30_penalty_values.append(row_i.iloc[0]['PAM30Penalties'])
        # Retrieve the target rows
        target_rows_i = get_target_rows(target_df, antigen_sequence,
                                        target_ab_ids)

        rows.append(row_i)
        target_rows.append(target_rows_i)

    # pam30_penalty_values = compute_pam30_penalties(antigen_sequences)
    # Sign flip: now in range -1.0 to 0.0
    pam30_penalty_values = [-1.0 * p30i for p30i in pam30_penalty_values]

    start_rows = [0]
    all_rows = []
    for ri, tri in zip(rows, target_rows):
        start_rows.append(start_rows[-1] + len(ri) + len(tri))
        all_rows.append(ri)
        all_rows.append(tri)
    end_rows = start_rows[1:]
    start_rows = start_rows[:-1]

    # print('Starts and ends for slices:')
    # print(start_rows)
    # print(end_rows)

    # prepare the inputs to the model
    pred_x = torch.tensor(
        pd.concat(all_rows, axis=0, sort=False)[
            predictor_cols].values,
        dtype=torch.float
    )
    pred_i = torch.tensor(
        pd.concat(all_rows, axis=0, sort=False)[
            type_col].values.astype(np.double),
        dtype=torch.long
    )

    print('Obtaining single joint MVN at {} ...'.format(
        str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
    with torch.no_grad(), gpytorch.settings.max_root_decomposition_size(100000):
        joint_mvn = model(PRED_FEATURE_SCALING * pred_x, pred_i)
    print('Done at {}!'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))

    exploit = []
    explore = []
    pred_mean = []
    pred_std = []
    for row_i, si, ei in zip(rows, start_rows, end_rows):
        explore_i, exploit_i, pred_mean_i, pred_std_i = get_dec_rule_val(
            row_i, joint_mvn.mean[si:ei], joint_mvn.covariance_matrix[si:ei, si:ei],
            type_col, lincoeffs_target_means_by_ab,
            noisevar, cost_dict
        )
        exploit.append(exploit_i)
        explore.append(explore_i)
        pred_mean.append(pred_mean_i)
        pred_std.append(pred_std_i)
    print('Done at {}.'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
    # scores = [er_i + ei_i for er_i, ei_i in zip(explore, exploit)]
    scores = [er_i + ei_i for er_i, ei_i in zip(explore, exploit)]
    for i, p30i in enumerate(pam30_penalty_values):
        if not np.isnan(p30i):
            scores[i] = scores[i] + p30i
    for er_i, ei_i, p30i, si in zip(explore, exploit, pam30_penalty_values, scores):
        # print('Explore: {}, Exploit: {}, Score: {}'.format(er_i, ei_i, si ))
        print('Explore: {}, Exploit: {}, Pam30Penalty: {}, Score: {}'.format(er_i, ei_i, p30i, si))

    return scores, pred_mean, pred_std


def compute_menu_scores_target_performance_and_cmi(
        menu_df, target_df, model, predictor_cols, type_col, noisevar,
        target_ab_ids=None, lincoeffs_target_means_by_ab=None, cost_dict=None
):
    """
    For each study in the menu dataframe, compute its desirability.

    Scores are a function of 1. the conditional mutual information between the
    observation associated with the menu_df row and the corresponding target
    rows in target_df; 2. the cost for a particular study type; 3. the predicted
    performance associated with the target experiments, weighted by the
    antibody's importance.

    :param menu_df:
    :param target_df:
    :param model:
    :param predictor_cols:
    :param type_col:
    :param noisevar:
    :param target_ab_ids:
    :param lincoeffs_target_means_by_ab:
    :param cost_dict
    :return:
    """

    scores = []
    pred_mean = []
    pred_std = []
    for idx in range(menu_df.shape[0]):
        row_i = menu_df.iloc[[idx]]
        # Use brackets to ensure the returned slice is a DF
        # Get the antigen sequence from row_i
        # Assumes that the menu_df is NOT indexed by this value, and instead,
        # it's in the row
        antigen_sequence = row_i.iloc[0]['AntigenSequence']

        # Retrieve the target rows
        target_rows_i = get_target_rows(target_df, antigen_sequence,
                                      target_ab_ids)
        # prepare the inputs to the model
        pred_x = torch.tensor(
            pd.concat([row_i, target_rows_i], axis=0, sort=False)[
                predictor_cols].values,
            dtype=torch.float
        )
        pred_i = torch.tensor(
            pd.concat([row_i, target_rows_i], axis=0, sort=False)[
                type_col].values.astype(np.double),
            dtype=torch.long
        )

        # Get the joint prediction
        print('Obtaining individual joint MVN at {} ...'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            joint_mvn = model(PRED_FEATURE_SCALING * pred_x, pred_i)

        explore_i, exploit_i, pred_mean_i, pred_std_i = get_dec_rule_val(
            row_i, joint_mvn.mean, joint_mvn.covariance_matrix,
            type_col, lincoeffs_target_means_by_ab,
            noisevar, cost_dict
        )
        print('Done at {}.'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))

        scores.append(explore_i + exploit_i)
        pred_mean.append(pred_mean_i)
        pred_std.append(pred_std_i)

        # print('Computed score {} from explore and exploit contributions {} and '
        #       '{}'.format(scores[-1], explore_i, exploit_i))

    return scores, pred_mean, pred_std


def compute_mei_score(mvn_dist, best_so_far):
    """
    Given an MVN, compute the MEI score function for all elements

    Goal is to select the maximizer of the score, where the score is higher for
    inputs likely to be more negative than the best so far.

    Appropriate in a context where we only care about optimizing a particular
    fidelity, without reference to others.

    :param mvn_dist: gpytorch.MultivariateNormal, with attributes mean and
        stddev.
    :param best_so_far: numeric, giving the value of the (estimated) best
        (i.e., most negative) response so far seen; estimate may be necessary
        for noisy case.
    :return: torch.tensor of scores, corresponding to the values in mvn_dist
    """

    a = torch.distributions.Normal(0.0, 1.0)

    # Compute the input to the .cdf and .pdf (limits for extremes)
    input_to_cdf_and_pdf = (best_so_far - mvn_dist.mean) / mvn_dist.stddev
    input_to_cdf_and_pdf = torch.min(input_to_cdf_and_pdf,  6.0 * torch.ones_like(input_to_cdf_and_pdf))
    input_to_cdf_and_pdf = torch.max(input_to_cdf_and_pdf, -6.0 * torch.ones_like(input_to_cdf_and_pdf))

    # Scores: higher score for a more negative mvn_dist.mean value, for wider
    # std_dev.
    scores = (best_so_far - mvn_dist.mean) * a.cdf(input_to_cdf_and_pdf) \
          + mvn_dist.stddev * torch.exp(a.log_prob(input_to_cdf_and_pdf))

    return scores


def compute_menu_scores_mei(menu_df, model, predictor_cols, type_col, best_so_far):
    """
    For each study in the menu dataframe, compute its desirability

    This desirability is solely a function of the MEI with respect to the
    single performance metric, where the type_col value effectively acts as a
    selectable context. No PAM30 penalty is applied.
    :param menu_df:
    :param model:
    :param predictor_cols:
    :param type_col:
    :param best_so_far: numeric value for the best (i.e., most positive)
        observation seen
    """

    # Obtain the set of features to go into the model, assumed to take both
    # the description of the interface (from predictor_cols) and the type (from
    # type_col):
    pred_x = torch.tensor(menu_df[predictor_cols].values, dtype=torch.float)
    pred_i = torch.tensor(menu_df[type_col].values.astype(np.double), dtype=torch.long)  # Is this double-retyping necessary?

    # Pass the inputs to the model
    print('Obtaining single joint MVN at {} ...'.format(
        str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
    with torch.no_grad(), gpytorch.settings.max_preconditioner_size(100):
        joint_mvn = model(PRED_FEATURE_SCALING * pred_x, pred_i)
    print('Done at {}!'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))

    # Use the MVN output to compute the score
    scores = compute_mei_score(joint_mvn, best_so_far)

    # Convert to lists and print
    scores_out = [vi for vi in scores]
    pred_mean_out = [mi for mi in joint_mvn.mean]
    pred_std_out = [stdi for stdi in joint_mvn.stddev]

    for si, mi, stdi in zip(scores_out, pred_mean_out, pred_std_out):
        print('Mean: {}, Std: {}, MEI Score: {}'.format(mi, stdi, si))

    print('Of the {} computed std dev values, {} are nans.'.format(pred_x.shape[0], np.sum(np.isnan(pred_std_out))))


    return scores_out, pred_mean_out, pred_std_out


def main_from_parsed_args(args):

    # Load the configuration information:
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        print('Loading config from {}'.format(config_path))
        # Extract any keywords from it
        kwargs = yaml.safe_load(f)

    # For every field of args (except config) add this kwarg
    for ki, vi in vars(args).items():
        if ki == 'config':
            continue
        kwargs[ki] = vi

    print('Running main with the following kwargs:')
    print(kwargs)

    main(**kwargs)


def _eliminate_studies_for_unhandled_study_types(menu, num_tasks):

    """
    Eliminate studies from menu corresponding to unhandled study types.

    Remove anything where the study_type in EXPERIMENTAL_TYPES_STR 
    matches an element of EXPERIMENTAL_TYPES_INT_CODES >= num_tasks

    :param menu: expanded menu
    :param num_tasks: the amount of study types we know about
    :return menu

    """
    if menu and menu['studies']:
        orig_len = len(menu['studies'])

        menu['studies'] = [
            mi for mi in menu['studies']
            if mi['study_type'] in EXPERIMENTAL_TYPES_STR and
                EXPERIMENTAL_TYPES_INT_CODES[
                    EXPERIMENTAL_TYPES_STR.index(mi['study_type'])
                ] < num_tasks
        ]
        if orig_len != len(menu['studies']):
            print(
                'Eliminated {} studies from expanded menu for having '
                'disallowed study types (model cannot predict)!'.format(
                    orig_len - len(menu['studies'])
                )
            )
    return menu

def _compute_pam30_for_menu_df(menu_df):
    """
    Compute the PAM30 scores and save into the menu_df

    :param menu_df: expanded menu dataframe
    :return menu_df: menu dataframe with PAM30 scores included
    """

    print('Computing the PAM30 penalty function at {} ...'.format(
        get_datetime()))
    antigen_sequences = list(menu_df.reset_index()['AntigenSequence'])
    pam30_penalty_values = compute_pam30_penalties(antigen_sequences)
    menu_df['PAM30Penalties'] = pam30_penalty_values
    print('... done at {}!'.format(get_datetime()))
    return menu_df



def main(history=None, menu=None, studies=None,
         inbox=None, fastas=None, database_url=None,
         stale_history=None,
         nsubmit=1, nmutpermaster=10,
         model_state_path=None, predictor_columns=None, type_col=None,
         target_ab_master_and_structures=None,
         target_ab_ids=None,
         target_ab_weights_np=None,
         previous_model_data=None,
         timing=None, min_time_remaining_to_start=None,
         nlocationsmutatepoissexpect=2,
         save_score_computation_inputs=False,
         mut_generator_conditional_transitions=None,
         mut_generator_log_mixture_weights=None,
         override_mutant_sequence=None,
         score_function_type=None,
         override_master_with_mutant_each_round=False,
         percent_keep_orig_menu=10,
         percent_keep_round=90,
         override_mutant_num_locations=2,
         nrandom=None, seq=None, seq_dist=None,
         **kwargs):
    """
    Select a study and parameters.

    Read the history_file and the pre-trained model, update the model with
    history, select a study and a set of parameters for
    that study from the menu_file, and write this to the registry_file

    :param history: Path to .yaml file containing the history of
        selected study types and parameters.
    :param menu: Path to .yaml file containing the requested studies and
        their allowed parameters.
    :param studies: Path to the studies directory, containing the
        template studies.
    :param inbox: str, Path to directory where the requested studies will
        be placed
    :param fastas: str, path to directory where FASTAS will be written
    :param database_url: str or None, url of sina-managed database of study
        history.
    :param stale_history: str, path to a .yaml file (which may not exist) which
        contains a "stale" copy of the global history; use if unable to obtain
        global history.
    :param nsubmit: int, str: integer number of studies to submit in
        this decision cycle [Default: 1]
    :param nmutpermaster: int, str;  integer number of studies to enumerate
        from the menu's specification, per master antigen [Default: 10]
    :param model_state_path: str, giving path to the .pth file with the PyTorch
        state_dictionary in it.
    :param type_col: str with the name of the DataFrame column containing the
        study type information (initially, strings; converted to ints).
    :param predictor_columns: list of str, giving the ordered list of columns in
        the dataframe corresponding to the predictive features to be input to
        the regressor.
    :param target_ab_master_and_structures: dictionary containing information
        on the targets; decisions are made with respect to the target
        antibodies, structures, and contained sequences, where the experimental
        type is 'octet_ddg'. # TODO: Reconcile with target_ab_... below
    :param target_ab_ids: list of str, giving the IDs associated with the
        antibodies of interest
    :param target_ab_weights_np: None, list, or np.array of weights associated
        with the listed antibodies
    :param previous_model_data: str, path to .pkl file containing the tensors of
        the (training) data.
    :param timing: str, giving path to timing .yaml file, created on invocation
        of the worker. Contains two elements, 'ALLOCATION_START_TIME' and
        'ALLOCATION_DURATION'.
    :param min_time_remaining_to_start: str, formatted like a SLURM WALLTIME
        value (see utils.parse_slurm_allocation_duration) giving the minimum time
        remaining in the allocation at which the decision-maker is allowed to
        submit new studies; should closely or conservatively match the duration
        of a study. The intention is to avoid the scheduler killing the improv
        daemon while studies are in mid-flight, which would clutter up the
        global history file and confuse subsequent decision-making.
    :param nlocationsmutatepoissexpect: int or None, controls the
        EXPECTED number of locations that will be mutated from the master.
        expand_menu_study_params_master_antigen_structures_mutations uses this
        parameter to scale a poisson distribution over number of locations; a
        random set of mutations of the drawn size is then selected. If None, ALL
        locations are mutated, giving uniform sampling over the product of the
        individual components of each study's
        si['study_parameters'][AllowedMutationsKey]['AllowedMutations'].
    :param save_score_computation_inputs: bool, controls whether the first
        iteration model state is saved to a pickle in the current directory.
        False by default.
    :param mut_generator_conditional_transitions: list of n dictionaries that
        return a conditional distribution over the replacement residue, given
        the current residue.
    :param mut_generator_log_mixture_weights: list of floats, giving log mixture
        weights associated with the generator components in
        mut_generator_conditional_transitions.
    :param override_mutant_sequence: str or None, giving an alternative center
        point to mutant generation in expand_menu..., as opposed to the master
        sequence. Note that this is specified once, as opposed to the menu's
        separate specification of individual master sequences.
    :param score_function_type: str or None; either 'mei' or 'cmi_monolithic'.
        Selects the type of score computation executed by the agent. If None,
        defaults to 'cmi_monolithic'.
    :param override_master_with_mutant_each_round: bool; turn ON/OFF overriding 
        master sequence every round.
        defaults to False
    :param percent_keep_orig_menu: int; integer percentage of top 
        scoring studies from original menu to keep forever from first round.
        default: 10 (%)
    :param percent_keep_round: int; integer percentage of top 
        scoring new studies to keep each round (see example below)
        default: 90 (%)
    :param override_mutant_num_locations: int; Number of locations for 
        mutant to mutate when param override_master_with_mutant_each_round=True 
        (analogous to nlocationsmutatepoissexpect)
        default: 2
    :return 0 if successful
    """
    # Note: Reduce genetic drift from original menu with
    # 'percent_keep_orig_menu' and 'percent_keep_round'
    #
    # For example when:
    #   nmutpermaster=1000 (number of mutants to generate)
    #   percent_keep_orig_menu = 50 
    #   percent_keep_round = 50
    # subsequent rounds after the 1st will have the following dist:
    #   500 best original mutant studies
    #   250 best new mutant studies from last round
    #   250 new mutants to generate this round
    # 
    assert percent_keep_round < 100, "percent_keep_round should be less than 100%"
    assert percent_keep_orig_menu < 100, "percent_keep_orig_menu should be less than 100%"
    # Save the menu file path for re-reading later
    menu_file = menu

    # TODO: evaluate other specification of override_mutant_sequence;
    #  e.g., dictionary, specifying study type and master ID, etc.

    # ############################
    # ### Check time remaining ###
    # ############################

    # If either the information on the allocation (timing) or
    # min_time_remaining_to_start is missing (None or ''), this condition
    # cannot be checked.
    suff_time = check_allocation_suff_rem_time(
        timing, min_time_remaining_to_start
    )
    if not suff_time:
        return 0

    # ##############
    # ### Set up ###
    # ##############

    # We either didn't specify a minimum time, or we have at least that much
    # time left. Set up and select some studies.

    if score_function_type is None:
        score_function_type = 'cmi_monolithic'

    if target_ab_weights_np is None:
        target_ab_weights_np = -1.0 * np.ones((len(target_ab_ids), ), dtype=np.float)
    elif isinstance(target_ab_weights_np, list):
        target_ab_weights_np = np.array(target_ab_weights_np)

    if previous_model_data is None:
        raise ValueError('Some previous model data must be supplied!')
    elif isinstance(previous_model_data, str):
        with open(previous_model_data, 'rb') as f:
            previous_model_data = pickle.load(f)
    if not isinstance(previous_model_data, dict):
        raise ValueError('previous_model_data was neither a str nor dict!')
    
    orig_best_scoring_menu = {}

    # Note regarding types of studies:
    # The values below define the allowed studies and their types
    # (EXPERIMENTAL_COLUMN_NAMES; keying into the appropriate combination of
    # columns to calculate the score), how these are
    # translated to the integer-valued type indices that will appear in the
    # 'i' (type index) tensors fed into the models
    # (EXPERIMENTAL_TYPES_STR, EXPERIMENTAL_TYPES_INT_CODES, where these
    # together provide a dictionary for converting string values in the
    # study_type column into the integers) and how these experimental costs
    # will be compared in the CMI decision rule (experimental_costs).

    experimental_costs = [3.0*60.0, 5.0, 10000.0, 5.0*60.0]
    experimental_costs = [0.00001 * eci / experimental_costs[0] for eci in experimental_costs]
    # Above: Want exploration component to be O(10^1) and that MI between FoldX
    # and experiment will be <= O(1e-1), going down to O(1e-4), and that the
    # range of predicted "exploit" values will be O(1).
    # TODO: Scale experimental costs using the prior MI of the fidelities
    experimental_costs_dict = {et_int: ec for et_int, ec in zip(EXPERIMENTAL_TYPES_INT_CODES, experimental_costs)}

    # ##############################################
    # ### Acquire the history, menu, and studies ###
    # ##############################################

    if database_url is not None:
        if history is not None:
            print('WARNING: both database_url and history (path) supplied; '
                  'using database_url and ignoring history (path).')
        history = get_history_db(database_url, nrandom=nrandom, seq=seq, seq_dist=seq_dist)  # enable all get_history_db parameters aside from study_type
    else:
        history = get_history(history, stale_history=stale_history)
    menu = get_menu(menu)
    studies = get_studies(studies)

    # ### Use the history to remove elements from the menu ###
    if 'sampling_frequency_csv' in kwargs.keys():
        # acquire the dataframe from the csv (no further formatting is required)
        sampling_frequency_df = pd.read_csv(kwargs['sampling_frequency_csv'])
        maxmuts = min(8, min([
            len(si['study_parameters']['AllowedMutations']['AllowedMutations'])
            for si in menu['studies']
        ]))
        # TODO: check sampling_frequency_df to figure out how many locations
        #  are available; maxmuts must be no more than that
        menu = expand_menu_study_params(
            menu, n_mutants_per_master=nmutpermaster,
            generator_type='linear_mutant_generator',
            singlePointMutationDataWithSampleWeights=sampling_frequency_df,
            minNumLocationsToMutate=1,
            maxNumLocationsToMutate=maxmuts,
            override_mutant_sequence=override_mutant_sequence
        )
    else:
        menu = expand_menu_study_params(
            menu, n_mutants_per_master=nmutpermaster,
            n_locations_expected=nlocationsmutatepoissexpect,
            generator_type='weighted_mixture_of_conditionals',
            conditional_transitions=mut_generator_conditional_transitions,
            log_mixture_weights=mut_generator_log_mixture_weights,
            override_mutant_sequence=override_mutant_sequence
        )
    # Above: expand_menu_study_params assumes that the menu is  discrete;
    # continuous-range specification of menus is not yet implemented.

    if database_url is not None:
        # Construct study query data
        query_data_for_studies = study_query_data_for_expanded_menu(menu)

        # Have improv remove studies from menu that already exist in DB
        menu = remove_from_menu_db(database_url, query_data_for_studies, menu)
    else:
        menu = remove_from_menu(menu, list(history["history"].values()))
    # Above: remove_from_menu also assumes that the menu is discrete.
    # Removing previously-run objects from the menu assumes experiments
    # are deterministic (no information is gained by repetition), and further,
    # that there is no intrinsic "reward" for repeating "good" experiments;
    # thus there is no reason to repeat an identical experiment.

    # ######################################################
    # ### Prepare features and model for active learning ###
    # ######################################################

    # ### Load the model's state (weights) from file
    # Load the prepared model parameters from file:
    print('Loading old model state...')
    state_dict = torch.load(model_state_path)  # Note that the data is not included here, but that the likelihood.nose_covar.raw noise is.
    layer_sizes = get_layer_sizes(state_dict)
    num_tasks = get_num_tasks(state_dict)  # uses the task_covar_module tensor shape
    print('... done!')

    # Remove the studies for unhandled study_types ( >= num_tasks)
    menu = _eliminate_studies_for_unhandled_study_types(menu, num_tasks)

    # ######################################
    # ### Check if the menu is exhausted ###
    # ######################################
    # If we have run everything on the menu, do not submit a study.
    if not menu or not menu['studies']:
        print('Menu empty!')
        return 0

    # ### Obtain the dataframes for the menu and history ###
    # Featurize menu:
    # A single row in menu_df corresponds to a single element in menu['studies']
    menu_df, menu_cpx_dictionary = get_feature_representation_from_history_or_menu(menu, cpx_dictionary=None)
    menu_df[type_col] = [
        EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index(vi)]
        for vi in menu_df[type_col].values
    ]  # Convert to integers

    # Compute the PAM30 scores and save into the menu_df
    menu_df = _compute_pam30_for_menu_df(menu_df)

    # Below: history_tasks is now of type torch.long, containing the integer
    # codes for the study types (see EXPERIMENTAL_TYPES_INT_CODES)
    history_features_tensor, history_tasks, history_obs_val = \
        get_history_tensors_from_history(
            history, predictor_columns, EXPERIMENTAL_COLUMN_NAMES,
            EXPERIMENTAL_TYPES_INT_CODES, EXPERIMENTAL_TYPES_STR, type_col
        )

    print('Done preparing history_... tensors.')
    target_df, target_cpx_dict = get_target_features_from_history_or_menu(
        menu, target_ab_master_and_structures=target_ab_master_and_structures,
        target_study_type='octet_ddg', cpx_dictionary=None
    )
    target_df.set_index(['AntigenSequence', 'AntibodyID'], inplace=True)  # sort?
    target_df[type_col] = [
        EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index(vi)]
        for vi in target_df[type_col].values
    ]  # Convert to integers


    print('Setting up model with previous data...')
    try:
        x = previous_model_data['x'].clone().detach().float()
        idx = previous_model_data['i'].clone().detach().long()
        y = previous_model_data['y'].clone().detach().float()
    except KeyError:
        x = previous_model_data['train_x'].clone().detach().float()
        idx = previous_model_data['train_i'].clone().detach().long()
        y = previous_model_data['train_y'].clone().detach().float()

    # ### Check previous_model_data and history for unhandled study_type values
    # Previous model data:
    x, idx, y = _remove_for_idx_geq_num_tasks(
        x, idx, y, num_tasks, name='previous_model_data'
    )

    # History
    history_features_tensor, history_tasks, history_obs_val = \
        _remove_for_idx_geq_num_tasks(
            history_features_tensor, history_tasks, history_obs_val, num_tasks,
            name='history'
        )

    # Downsample the (STATIUM) data to control the combined pretrain and history data size:
    if x.shape[0] + history_features_tensor.shape[0] > TRIM_PRETRAIN_SIZELIM_HIST_PRETRAIN:
        n_statium_pretrain = torch.sum(
            idx == EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index('statium_batch_eval')]
        )
        current_overflow = x.shape[0] + history_features_tensor.shape[0] \
                           - TRIM_PRETRAIN_SIZELIM_HIST_PRETRAIN
        n_statium_pretrain_to_remove = np.max([0, np.min([current_overflow, n_statium_pretrain])])
        if n_statium_pretrain_to_remove == n_statium_pretrain:
            # Eliminate all of them
            keep = idx != EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index('statium_batch_eval')]
            x = x[keep, :]
            y = y[keep]
            idx = idx[keep]
        elif n_statium_pretrain_to_remove > 0:
            # Randomly select
            keep = idx != EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index('statium_batch_eval')]
            # Modify keep with randomly keeping some statium rows
            flip = np.random.permutation([i for i, ki in enumerate(keep) if not ki])
            flip = flip[:(len(flip) - n_statium_pretrain_to_remove)]
            for fi in flip:
                keep[fi] = True
            # Apply the logical screen
            x = x[keep, :]
            y = y[keep]
            idx = idx[keep]

    if history_features_tensor.shape[0] == 0:
        history_features_tensor = torch.empty((0, x.shape[-1]), dtype=torch.float)
    x = torch.cat([x, history_features_tensor], dim=-2)
    idx = torch.cat([idx, history_tasks], dim=-1)
    y = torch.cat([y, history_obs_val], dim=-1)

    print('Current data tensor shape is : {}'.format(x.shape))
    # Now, if the tensor is too big, cut it down a bit.
    # This is done for memory management reasons
    if x.shape[0] > 30000:
        keep = torch.zeros(idx.shape, dtype=torch.bool)
        flip = np.random.permutation([i for i, ki in enumerate(keep)])
        flip = flip[:30000]
        for fi in flip:
            keep[fi] = True
        x = x[keep, :]
        y = y[keep]
        idx = idx[keep]
        print('Reduced data tensor shape to {}.'.format(x.shape))

    if set(layer_sizes.keys()) == {'lin_layers'}:  # We have a MultitaskVanillaGPModelDKL model
        model = MultitaskVanillaGPModelDKL(
            (x, idx), y,
            layer_sizes=layer_sizes['lin_layers'],
            num_tasks=num_tasks,
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=Interval(0.0001, 1.0)
            )
        )
    elif set(layer_sizes.keys()) == {'lin_layers_lin', 'lin_layers_stationary'}:
        model = MultitaskLinPlusStationaryDKLGP(
            (x, idx), y,
            layer_sizes_lin=layer_sizes['lin_layers_lin'],
            layer_sizes_stationary=layer_sizes['lin_layers_stationary'],
            num_tasks=num_tasks,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=Interval(0.0001, 1.0)
            )
        )
    print('Model initialized; applying provided model state (weights)...')
    model.load_state_dict(state_dict)
    print('... done!')

    print_parameters(model, model.likelihood)

    # Run ~some~ predictions to initialize for fantasy model:
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _ = model(x, idx)

    # ####################################
    # ####################################
    # ### Select studies from the menu ###
    # ####################################
    # ####################################
    
    # If there remain elements in the menu, submit element(s).
    submitted_this_round = 0
    to_submit_this_round = int(nsubmit)
    if to_submit_this_round < 0:
        raise ValueError('to_submit_this_round is {} but must be a non-'
                         'negative integer!'.format(to_submit_this_round))
    studies_to_write = []

    while submitted_this_round < to_submit_this_round and menu_df.shape[0] > 0:
        print('Beginning round {}...'.format(submitted_this_round + 1))

        if save_score_computation_inputs and submitted_this_round == 0:
            # pickle up everything
            save_dict = {
                'menu_df': menu_df, # updated every round
                'target_df': target_df, # updated every round
                'predictor_columns': predictor_columns,
                'type_col': type_col,
                'noisevar': model.likelihood.noise,
                'target_ab_ids': target_ab_ids,
                'lincoeffs_target_means_by_ab': target_ab_weights_np,
                'cost_dict': experimental_costs_dict,
                'model_state_dict': state_dict,  # use get_num_tasks layer_sizes
                'model_inputs_x': model.train_inputs[0],
                'model_inputs_i': model.train_inputs[1],  # 2D tensor
                'model_targets_y': model.train_targets  # 1D tensor
            }
            with open('score_computation_inputs.pkl', 'wb') as f:
                pickle.dump(save_dict, f)

        # Note, regarding study_types: In both functions below, the requirement
        # is that the integers that appear in menu_df[type_col] be in the range
        # accepted by the model.

        if score_function_type == 'mei':
            print('Computing the MEI score function at {} ...'.format(
                str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
            best_seen_so_far = 0.0
            best_seen_so_far = np.min([best_seen_so_far, model.train_targets.min()])

            scores, pred_mean, pred_std = compute_menu_scores_mei(
                menu_df, model, predictor_columns, type_col, best_seen_so_far
            )
            print('done at {} ...'.format(
                str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        elif score_function_type == 'cmi_monolithic':
            print('Computing the score function in monolithic at {} ...'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
            scores, pred_mean, pred_std = compute_menu_scores_target_performance_and_cmi_monolithic(
                menu_df, target_df, model,
                predictor_columns,
                type_col,
                noisevar=model.likelihood.noise,
                target_ab_ids=target_ab_ids,
                lincoeffs_target_means_by_ab=target_ab_weights_np,
                cost_dict=experimental_costs_dict
            )
            print('done at {} ...'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        else:
            raise ValueError('score_function_type is {}; must be one of mei or cmi_monolithic.')

        # print('Computing the score function serially at {} ...'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        # scores_check, pred_mean_check, pred_std_check = compute_menu_scores_target_performance_and_cmi(
        #     menu_df, target_df, model,
        #     predictor_columns,
        #     type_col,
        #     noisevar=model.likelihood.noise,
        #     target_ab_ids=target_ab_ids,
        #     lincoeffs_target_means_by_ab=target_ab_weights_np,
        #     cost_dict=experimental_costs_dict
        # )
        # print(' done at {}!'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        #
        # assert all([si == sic for si, sic in zip(scores, scores_check)]), 'Not all scores are equal!'
        # assert all([pmi == pmic for pmi, pmic in zip(pred_mean, pred_mean_check)]), 'Not all scores are equal!'
        # assert all([psi == psic for psi, psic in zip(pred_std, pred_std_check)]), 'Not all scores are equal!'

        # Sort studies with best first (reverse argsort which is ascending)
        sorted_studies = np.argsort(np.nan_to_num(scores))[::-1].tolist()
        
        # Store top scoring sequences 
        sequences_by_score = []
        for idx in sorted_studies:
            sequences_by_score.append(menu['studies'][idx]['study_parameters']['AntigenSequence'])

        # Select best scoring study
        selected_study_idx = sorted_studies[0]

        print("Selected best study idx=" + str(selected_study_idx) + ", round=" + 
              str(submitted_this_round+1) + ",  score=" + str(scores[selected_study_idx]))

        # Use to min_pred_mean_study to select future mutants
        override_mutant_idx = np.nanargmin(pred_mean)

        # Print out where in the menu we have selected our study to indicate
        # how mutated it may be. 
        print('Selected study is [' + str(selected_study_idx) + ' of ' + 
        str(menu_df.shape[0]) + '] - round ' + str(submitted_this_round + 1))

        # Obtain the model predictions
        # TODO: Case: study yields more than one type of observation?
        mean_selected = pred_mean[selected_study_idx]  # TODO: Check index type
        std_selected = pred_std[selected_study_idx]

        # Obtain the necessary representations for feeding back into the model
        selected_observation_training_I = menu_df.iloc[selected_study_idx, :][
            type_col
        ]  # The (single) integer value of the study type
        #  Obtain the study from the menu
        selected_observation_training_X = PRED_FEATURE_SCALING * menu_df.iloc[selected_study_idx, :][
            predictor_columns
        ].to_numpy(dtype=float).reshape(1, -1)  # TODO: remove all scaling factors and incorporate into the model directly

        # Recast as torch.tensors
        selected_observation_training_X = torch.tensor(selected_observation_training_X, dtype=torch.float)
        selected_observation_training_I = torch.full((1,), selected_observation_training_I, dtype=torch.long)
        selected_observation_training_Y = torch.full((1,), mean_selected, dtype=torch.float)

        si = menu['studies'][selected_study_idx]  # menu['studies'][selected_study_idx]['study_type'] gives the menu-derived string of the study type
        override_si = menu['studies'][override_mutant_idx] # save study for mutant menu (next round)

        # Generate a request_id (Unique to avoid collisions as the history is
        # updated; below will generate a request_id which is precise to
        # microsecond level. Using the hostname can also be helpful.)
        request_id = "request_" + str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))  # timestamp with YYYYmmdd_HHMMSS_(microseconds)

        # Write out the study to the improv inbox
        # Below, si is an element from menu["studies"]
        studies_to_write.append({'parameters': si, 'request_id': request_id})

        # Add observation to the "training data" for the model
        print('Updating the model with the pseudoobservations at {}... '.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.max_root_decomposition_size(100000):
            # model = model.get_fantasy_model(
            #     inputs=[
            #         selected_observation_training_X,
            #         selected_observation_training_I
            #     ],
            #     targets=selected_observation_training_Y
            # )

            # Had some numerical problems with the get_fantasy_model method above
            model.set_train_data(inputs=[
                torch.cat([
                    model.train_inputs[0],
                    selected_observation_training_X
                ], dim=-2),
                torch.cat([
                    model.train_inputs[1], # Stored as a 2D tensor
                    selected_observation_training_I.reshape((-1, 1))
                ], dim=-2)
            ],
                targets=torch.cat([
                    model.train_targets, # Stored as a 1D tensor
                    selected_observation_training_Y
                ], dim=-1),
                strict=False
            )
        print('... done at {}!'.format(str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))

        # Remove this item from the menu
        print('Removing the selected study from the menu... ')
        
        # Note that if menu has more than one copy of the same study, it's removed more than once, and we're hosed.
        menu['studies'] = [sj for j, sj in enumerate(menu['studies']) if j != selected_study_idx]
        menu_df = menu_df.iloc[
                  [True if i not in [selected_study_idx] else False for i in range(menu_df.shape[0])], :
                  ]  # the right row
        sorted_studies = sorted_studies[1:] # Remove first study

        # Decrement any indicies > selected_study_idx
        sorted_studies = [i-1 if i > selected_study_idx else i for i in sorted_studies]
        print('... done!')
        

        print('Done with iteration {}!'.format(submitted_this_round + 1))

        if(submitted_this_round + 1 < to_submit_this_round and override_master_with_mutant_each_round):

            # Print out where in the menu we have selected our mutant to indicate
            # how mutated it may be.
            print('Override mutant selected is [' + str(override_mutant_idx) + ' of ' + 
                  str(menu_df.shape[0]) + '] - round ' + str(submitted_this_round + 1))

            # Print out the diff between our selected mutant to be 
            # used for subsequent rounds and the master antigen 
            override_mutant_sequence_rounds = override_si['study_parameters']['AntigenSequence']
            master_seq = override_si['study_parameters']['MasterAntigenSequence']
            print('seq diff for override_mutant_sequence from master \n' + 
                  str(diff_seqs(override_mutant_sequence_rounds, master_seq)))

            ### Create or append new menu/menu_df/target_df for subsequent rounds
            num_orig_studies_to_keep = int(nmutpermaster * (percent_keep_orig_menu / 100.0))
            print("Keep original studies: " + str(num_orig_studies_to_keep))
            
            if submitted_this_round == 0:
                # First round carve out original studies to keep forever, the rest are new mutants.
                num_studies_to_keep = num_orig_studies_to_keep
                num_mutants_to_generate = nmutpermaster - num_studies_to_keep
            else:
                # Take remaining studies available and apply percent to them to keep.
                num_studies_to_keep = int((nmutpermaster - num_orig_studies_to_keep) *
                                          (percent_keep_round / 100.0))
                print("Keeping mutant studies: " + str(num_studies_to_keep))
                num_mutants_to_generate = nmutpermaster - num_orig_studies_to_keep - num_studies_to_keep
            
            print("Generating new mutants: " + str(num_mutants_to_generate))
            assert num_mutants_to_generate > 0, str("Not generating any mutants." +
                " Check percent_keep_round and percent_keep_orig_menu less than 100%")

            highest_scoring_ind = sorted_studies[0:num_studies_to_keep]

            # Select only the top studies to keep in the menu and menu_df
            menu['studies'] = [sj for j, sj in enumerate(menu['studies']) if j in highest_scoring_ind]
            menu_df = menu_df.iloc[
                [True if i in highest_scoring_ind else False for i in range(menu_df.shape[0])], :
            ]

            # Validate that the menu_df sequence indices match the menu
            menu_sequences = [sj['study_parameters']['AntigenSequence'] for sj in menu['studies']]
            menu_sequences_df = menu_df['AntigenSequence'].tolist()
            assert menu_sequences == menu_sequences_df, "menu sequences not equal or aligned"

            # Validate that the top scoring sequences are indeed in the menu
            sequences_by_score = sequences_by_score[1:len(menu['studies']) + 1] #offset by 1 for best study removed
            assert sorted(sequences_by_score) == sorted(menu_sequences), "best scored seq's not in menu"
            
            if submitted_this_round > 0 and orig_best_scoring_menu is not None:
                # Remove any studies from original menu from new menu this round
                menu, menu_keep = remove_from_menu_with_return_removed(menu, orig_best_scoring_menu['studies'])
                menu_df = menu_df.iloc[
                    [True if menu_keep[i] else False for i in range(menu_df.shape[0])], :
                ]
                # Remove any selected studies that could come from the expanded menu we are keeping
                orig_best_scoring_menu, menu_keep_orig = \
                    remove_from_menu_with_return_removed(orig_best_scoring_menu, [s['parameters'] for s in studies_to_write])
                orig_best_scoring_menu_df = orig_best_scoring_menu_df.iloc[
                    [True if menu_keep_orig[i] else False for i in range(orig_best_scoring_menu_df.shape[0])], :
                ]
            
            assert menu_df.shape[0] == len(menu['studies'])
            
            # Generate the best mutant's expanded menu
            mutant_menu = get_menu(menu_file)

            if 'sampling_frequency_csv' in kwargs.keys():
                #re-read sampling df since it gets modified by linear mutant generator
                sampling_frequency_df = pd.read_csv(kwargs['sampling_frequency_csv'])
                num_locations_to_mutate_rounds = min(override_mutant_num_locations, maxmuts)
                mutant_menu = expand_menu_study_params(
                    mutant_menu, 
                    n_mutants_per_master=num_mutants_to_generate,
                    generator_type='linear_mutant_generator',
                    singlePointMutationDataWithSampleWeights=sampling_frequency_df,
                    minNumLocationsToMutate=1,
                    maxNumLocationsToMutate=num_locations_to_mutate_rounds,
                    override_mutant_sequence=override_mutant_sequence_rounds
                )
            else:
                mutant_menu = expand_menu_study_params(
                    mutant_menu, 
                    n_mutants_per_master=num_mutants_to_generate,
                    n_locations_expected=override_mutant_num_locations,
                    generator_type='weighted_mixture_of_conditionals',
                    conditional_transitions=mut_generator_conditional_transitions,
                    log_mixture_weights=mut_generator_log_mixture_weights,
                    override_mutant_sequence=override_mutant_sequence_rounds #override with mutant w/ best score
                )

            # Remove previously-executed studies, dataframes not yet generated so don't have to
            # worry about them.
            if database_url is not None:
                # Construct study query data
                query_data_for_studies = study_query_data_for_expanded_menu(mutant_menu)
                # Have improv remove studies from menu that already exist in DB
                mutant_menu = remove_from_menu_db(database_url, query_data_for_studies, mutant_menu)
            else:
                mutant_menu = remove_from_menu(mutant_menu, list(history["history"].values()))

            mutant_menu = remove_from_menu(mutant_menu, menu['studies'])
            if submitted_this_round > 0 and len(orig_best_scoring_menu) > 0:
                mutant_menu = remove_from_menu(mutant_menu, orig_best_scoring_menu['studies'])
            
            # Remove studies that we have already selected
            mutant_menu = remove_from_menu(mutant_menu, [s['parameters'] for s in studies_to_write])

            # Remove the studies for study_types < num_tasks
            mutant_menu = _eliminate_studies_for_unhandled_study_types(mutant_menu, num_tasks)

            try:
                mutant_menu_df, menu_cpx_dictionary = get_feature_representation_from_history_or_menu(mutant_menu,
                                                        cpx_dictionary=menu_cpx_dictionary)
            except KeyError:
                mutant_menu_df, menu_cpx_dictionary = get_feature_representation_from_history_or_menu(mutant_menu)
            
            mutant_menu_df[type_col] = [
                EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index(vi)]
                for vi in mutant_menu_df[type_col].values
            ]  # Convert to integers

            # Compute the PAM30 scores and save into the menu_df
            mutant_menu_df = _compute_pam30_for_menu_df(mutant_menu_df)

            print('(Mutant) Done preparing addendum menu dataframe.')
            try:
                mutant_target_df, target_cpx_dict = get_target_features_from_history_or_menu(
                    mutant_menu, target_ab_master_and_structures=target_ab_master_and_structures,
                    target_study_type='octet_ddg', cpx_dictionary=target_cpx_dict
                )
            except KeyError:
                mutant_target_df, target_cpx_dict = get_target_features_from_history_or_menu(
                    mutant_menu, target_ab_master_and_structures=target_ab_master_and_structures,
                    target_study_type='octet_ddg'
                )
            mutant_target_df.set_index(['AntigenSequence', 'AntibodyID'], inplace=True)  # sort?
            mutant_target_df[type_col] = [
                EXPERIMENTAL_TYPES_INT_CODES[EXPERIMENTAL_TYPES_STR.index(vi)]
                for vi in mutant_target_df[type_col].values
            ]  # Convert to integers

            # Get all the sequences in the menu_df so we can pull them out of target_df
            antigen_sequences = list(menu_df['AntigenSequence'])

            # Make sequences unique indices for DataFrame.loc call below. 
            # Note: pandas-1.1.2 (as compared to pandas-1.0.4) no longer allows 
            # for repeated indicies to be present in DataFrame.loc function.
            antigen_sequences = list(set(antigen_sequences)) 
 
            # Trim down target_df to match the new menu_df. This will pull out
            # all the rows with antigen_sequence matches regardless of the AntibodyID.
            target_df = target_df.loc[antigen_sequences, :]
            
            if(submitted_this_round == 0):
                # Save off this best scoring menu because it is part of the original
                orig_best_scoring_menu = menu.copy()
                orig_best_scoring_menu_df = menu_df.copy()
                orig_best_scoring_target_df = target_df.copy()

            # Add the new mutant menu to remaining top scoring studies we are keeping
            menu['studies'] = menu['studies'] + mutant_menu['studies'] + orig_best_scoring_menu['studies']

            len_before = len(menu['studies'])
            # Remove studies that we have already selected
            menu = remove_from_menu(menu, [s['parameters'] for s in studies_to_write])
            len_after = len(menu['studies'])
            assert len_before == len_after, "Duplicate study found in menu!"

            # Concatenate the dataframes for top scoring studies with new mutants
            if orig_best_scoring_menu_df.empty:
                menu_df = pd.concat([menu_df, mutant_menu_df], axis=0)
                target_df = pd.concat([target_df, mutant_target_df], axis=0)
            else:
                menu_df = pd.concat([menu_df, mutant_menu_df, orig_best_scoring_menu_df], axis=0)
                target_df = pd.concat([target_df, mutant_target_df, orig_best_scoring_target_df], axis=0)
            
            # Remove the duplicate target dataframes (if any)
            target_df.reset_index(inplace=True)
            target_df.drop_duplicates(subset=['AntigenSequence', 'AntibodyID'], inplace=True)
            target_df.set_index(['AntigenSequence', 'AntibodyID'], inplace=True)

        submitted_this_round += 1
        
    # ######################################
    # ### Write out the selected studies ###
    # ######################################

    write_all_selected_studies(
        studies_to_write, studies, improv_inbox=inbox,
        fastas_directory=fastas
    )

    return 0


if __name__ == "__main__":
    # args = sys.argv[1:]
    # main(*args)

    # Set up some dummy configuration information
    # Filter annoying PDBConstructionWarning
    # import warnings
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    warnings.filterwarnings('ignore', category=PDBConstructionWarning, module='Bio.PDB')

    cli = True

    if not cli:
        pass

    if cli:
        parser = setup_argparser()
        parsed_args = parser.parse_args()
        with open('gpt_decision_maker_stdout.log', 'w') as f0:
            sys.stdout = f0
            with open('gpt_decision_maker_stderr.log', 'w') as f1:
                sys.stderr = f1
                try:
                    main_from_parsed_args(parsed_args)
                except Exception as e:
                    print('Encountered an error! Error text follows:')
                    print(e)
                    print(e, file=sys.stderr)
                    print('Error encountered; flushing print buffers!', flush=True)
                    raise e
