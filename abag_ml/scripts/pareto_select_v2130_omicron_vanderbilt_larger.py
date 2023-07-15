# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

import os
import re
import time
import datetime
import random
import copy

import numpy as np
import pandas as pd

# Fix random seeds in both np.random and random
np.random.seed(0)
random.seed(0)

from improvwf.utils import get_menu
from vaccine_advance_core.featurization.bio_sequence_manipulation import \
    recast_as_seqrecord, recast_sequence_as_str
# from vaccine_advance_core.featurization.utils import get_datetime
from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    fasta_from_list_of_seqrecords, list_of_seqrecords_from_fasta
from vaccine_advance_core.featurization.seq_to_features import diff_seqs, mutate_seq
import abag_ml.pareto_selection as ps
from abag_agent_setup.expand_allowed_mutant_menu import AllowedMutationsKey, derive_idstr_from_seq
from abag_ml.scripts.prepare_fasta_from_csv_m396_Aug20_iteration3 import \
    check_new_nxst_vs_master

conventional_id_col = ps.conventional_id_col
conventional_seq_col = ps.conventional_seq_col
conventional_mutations_col = ps.conventional_mutations_col  # contains the text form of the mutation as tuples

_FEP_COLUMN_NAME = 'VARIANT_Ab_FEP_ddG_stability'
_G446S_CONTACT_SET = [104, 105, 118, 185, 186] # , 189]
_Q498R_CONTACT_SET = [166, 189]
_underrepresented_position_set = [33, 56, 103, 105, 112, 113, 165, 168, 186]  # Set as of morning 2022/01/13

_COLUMNS_EXPECTED_DF_OUT = [  # There was a blank string here
        'AntigenSequence', 'Complex', 'Mutation', '(last_DDG)', 'AntigenDescription', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_3dd5889a89cf8b317451aa947b28a6e8_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ce192ed827b18b1e12f405ac4115109a_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_rosetta_listed_mutants_fl',  # Prepend _keep
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_rosetta_listed_mutants_fl',  # Prepend _keep
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_rosetta_listed_mutants_fl',  # Prepend _keep
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_178d82097769258cd03c62455b79f883_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_9ed9fecd79a129ad49117a3670df2eec_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_bc18bad93db73121d3a1bba928d46e20_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_eb92a9f931b36a40f30b7d07532fb7e4_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_65e2f8afb1df57348885469b4508142d_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_81ad2461df6bc678fdd4b2dd55e91a23_rosetta_listed_mutants_fl', 
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ead0af03c0b100ab3a1fd4f5236ca481_rosetta_listed_mutants_fl', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ce192ed827b18b1e12f405ac4115109a_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ce192ed827b18b1e12f405ac4115109a_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_178d82097769258cd03c62455b79f883_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_178d82097769258cd03c62455b79f883_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_3dd5889a89cf8b317451aa947b28a6e8_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_3dd5889a89cf8b317451aa947b28a6e8_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_9ed9fecd79a129ad49117a3670df2eec_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_9ed9fecd79a129ad49117a3670df2eec_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_bc18bad93db73121d3a1bba928d46e20_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_bc18bad93db73121d3a1bba928d46e20_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants',       # Prepend _keep
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_eb92a9f931b36a40f30b7d07532fb7e4_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_eb92a9f931b36a40f30b7d07532fb7e4_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_65e2f8afb1df57348885469b4508142d_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_65e2f8afb1df57348885469b4508142d_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_81ad2461df6bc678fdd4b2dd55e91a23_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_81ad2461df6bc678fdd4b2dd55e91a23_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants',       # Prepend _keep
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants',       # Prepend _keep
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants', 
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ead0af03c0b100ab3a1fd4f5236ca481_foldx_listed_mutants', 
        'delta_FoldXAverageDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_ead0af03c0b100ab3a1fd4f5236ca481_foldx_listed_mutants', 
        'VARIANT_Ab_FEP_ddG_stability', 
        'VARIANT_Ab_FEP_ddG_stability_uncertainty', 
        'VARIANT_Q493K_SFE_7l7e_DDG', 
        'VARIANT_Q493K_SFE_7l7e_error', 
        'VARIANT_Q493K_SFE_7l7e_confidence', 
        'VARIANT_Q493K_SFE_7l7e_Model', 
        'VARIANT_Q493K_SFE_7t9k_DDG', 
        'VARIANT_Q493K_SFE_7t9k_error', 
        'VARIANT_Q493K_SFE_7t9k_confidence', 
        'VARIANT_Q493K_SFE_7t9k_Model', 
        'VARIANT_Q493K_SFE_priority_Model', 
        'VARIANT_Q493K_SFE_priority_DDG', 
        'VARIANT_Q493K_SFE_priority_error', 
        'VARIANT_Q493K_SFE_priority_confidence', 
        'VARIANT_Q493R_SFE_7l7e_DDG', 
        'VARIANT_Q493R_SFE_7l7e_error', 
        'VARIANT_Q493R_SFE_7l7e_confidence', 
        'VARIANT_Q493R_SFE_7l7e_Model', 
        'VARIANT_Q493R_SFE_7t9k_DDG', 
        'VARIANT_Q493R_SFE_7t9k_error', 
        'VARIANT_Q493R_SFE_7t9k_confidence', 
        'VARIANT_Q493R_SFE_7t9k_Model', 
        'VARIANT_Q493R_SFE_priority_Model', 
        'VARIANT_Q493R_SFE_priority_DDG', 
        'VARIANT_Q493R_SFE_priority_error', 
        'VARIANT_Q493R_SFE_priority_confidence', 
        'VARIANT_R346K_SFE_7l7e_DDG', 
        'VARIANT_R346K_SFE_7l7e_error', 
        'VARIANT_R346K_SFE_7l7e_confidence', 
        'VARIANT_R346K_SFE_7l7e_Model', 
        'VARIANT_R346K_SFE_priority_Model', 
        'VARIANT_R346K_SFE_priority_DDG', 
        'VARIANT_R346K_SFE_priority_error', 
        'VARIANT_R346K_SFE_priority_confidence', 
        'VARIANT_R346K,Q493K_dGPullingAvg', 
        'VARIANT_R346K,Q493K_dGPullingN', 
        'VARIANT_R346K,Q493K_dGPullingRaw', 
        'VARIANT_R346K,Q493K_dGPulling1', 
        'VARIANT_R346K,Q493K_dGPulling2', 
        'VARIANT_R346K,Q493K_dGPullingMax', 
        'ML Score', 
        'Num Mutations', 
        'Position Diff', 
        '(last_DDG)_num_vals', 
        'VARIANT_Ab_FEP_ddG_stability_num_vals', 
        'VARIANT_Ab_FEP_ddG_stability_uncertainty_num_vals', 
        'VARIANT_Q493K_SFE_7l7e_DDG_num_vals', 
        'VARIANT_Q493K_SFE_7l7e_error_num_vals', 
        'VARIANT_Q493K_SFE_7l7e_confidence_num_vals', 
        'VARIANT_Q493K_SFE_7t9k_DDG_num_vals', 
        'VARIANT_Q493K_SFE_7t9k_error_num_vals', 
        'VARIANT_Q493K_SFE_7t9k_confidence_num_vals', 
        'VARIANT_Q493K_FEP_ddG_stability_num_vals', 
        'VARIANT_Q493K_FEP_ddG_stability_uncertainty_num_vals', 
        'VARIANT_Q493K_dGPullingAvg_num_vals', 
        'VARIANT_Q493K_dGPullingMax_num_vals', 
        'VARIANT_Q493K_SFE_priority_DDG_num_vals', 
        'VARIANT_Q493K_SFE_priority_error_num_vals', 
        'VARIANT_Q493K_SFE_priority_confidence_num_vals', 
        'VARIANT_Q493R_SFE_7l7e_DDG_num_vals', 
        'VARIANT_Q493R_SFE_7l7e_error_num_vals', 
        'VARIANT_Q493R_SFE_7l7e_confidence_num_vals', 
        'VARIANT_Q493R_SFE_7t9k_DDG_num_vals', 
        'VARIANT_Q493R_SFE_7t9k_error_num_vals', 
        'VARIANT_Q493R_SFE_7t9k_confidence_num_vals', 
        'VARIANT_Q493R_FEP_ddG_stability_num_vals', 
        'VARIANT_Q493R_FEP_ddG_stability_uncertainty_num_vals', 
        'VARIANT_Q493R_dGPullingAvg_num_vals', 
        'VARIANT_Q493R_dGPullingMax_num_vals', 
        'VARIANT_Q493R_SFE_priority_DDG_num_vals', 
        'VARIANT_Q493R_SFE_priority_error_num_vals', 
        'VARIANT_Q493R_SFE_priority_confidence_num_vals', 
        'VARIANT_R346K_SFE_7l7e_DDG_num_vals', 
        'VARIANT_R346K_SFE_7l7e_error_num_vals', 
        'VARIANT_R346K_SFE_7l7e_confidence_num_vals', 
        'VARIANT_R346K_SFE_7t9k_DDG_num_vals', 
        'VARIANT_R346K_SFE_7t9k_error_num_vals', 
        'VARIANT_R346K_SFE_7t9k_confidence_num_vals', 
        'VARIANT_R346K_FEP_ddG_stability_num_vals', 
        'VARIANT_R346K_FEP_ddG_stability_uncertainty_num_vals', 
        'VARIANT_R346K_dGPullingAvg_num_vals', 
        'VARIANT_R346K_dGPullingMax_num_vals', 
        'VARIANT_R346K_SFE_priority_DDG_num_vals', 
        'VARIANT_R346K_SFE_priority_error_num_vals', 
        'VARIANT_R346K_SFE_priority_confidence_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7l7e_DDG_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7l7e_error_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7l7e_confidence_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7t9k_DDG_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7t9k_error_num_vals', 
        'VARIANT_R346K,Q493K_SFE_7t9k_confidence_num_vals', 
        'VARIANT_R346K,Q493K_FEP_ddG_stability_num_vals', 
        'VARIANT_R346K,Q493K_FEP_ddG_stability_uncertainty_num_vals', 
        'VARIANT_R346K,Q493K_dGPullingAvg_num_vals', 
        'VARIANT_R346K,Q493K_dGPullingMax_num_vals', 
        'VARIANT_R346K,Q493K_SFE_priority_DDG_num_vals', 
        'VARIANT_R346K,Q493K_SFE_priority_error_num_vals', 
        'VARIANT_R346K,Q493K_SFE_priority_confidence_num_vals', 
        'MasterAntigenID', 
        'AntibodyID', 
        'SeqNumberingMutation', 
        'StudyType', 
        'Emin', 
        'FoldXFinalDG', 
        'FoldXInterfaceDG', 
        'FoldXInterfaceDeltaDG', 
        'MasterAntigenDescription', 
        'MasterAntigenSequence', 
        'RosettaFlexDDGMax', 
        'RosettaFlexDDGMin', 
        'StructureHash', 
        'StructurePath', 
        'WT_FoldXFinalDG', 
        'WT_FoldXInterfaceDG', 
        'FoldXFinalDDG', 
        'StructureHashVal', 
        'StructureHashType', 
        'MasterAntigenSequenceHashStr', 
        'AntigenID', 
        'MutLists', 
        'MutPositions', 
        'MutPositionsAllLight', 
        'MutPositionsAllHeavy', 
        33, 55, 56, 103, 104, 105, 106, 107, 108, 109, 110, 113, 116, 118, 162, 163, 164, 165, 166, 168, 185, 186, 189, 
        'Selected', 'Best10', 'Num Mutations Squared', 'downselection_score', 
        'DownSelected', 'UpSelectedAblation', 'UpSelectedLMG', 'ParetoSet', 'UpSelectedFrom200',
        'MutationString','Common Contacts'
        ]

def dom_func_scalar(row=None, column=None):
    if row is None:
        return True

    return row[column]


# By convention, SMALLER (i.e., further toward -inf) values for a scalar dominate.
# Thus, all single-criterion functions below prefer more negative DDG values

def _score_rosfl_L452R(r=None):
    return ps.simple_row_scorer(
        r, column='RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078'
                  '_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_rosetta_listed_mutants_fl')


def _score_rosfl_R346_G446S_Q493R(r=None):  # R0024
    return ps.simple_row_scorer(
        r, column='RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078'
                  '_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_rosetta_listed_mutants_fl')


def _score_rosfl_R346K_G446S_Q493R(r=None):  # R0025
    return ps.simple_row_scorer(
        r, column='RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078'
                  '_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_rosetta_listed_mutants_fl')


def _score_fxi_L452R(r=None):
    return ps.simple_row_scorer(
        r, column='FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants')


def _score_fxi_R346_G446S_Q493R(r=None):  # R0024
    return ps.simple_row_scorer(
        r, column='FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants')


def _score_fxi_R346K_G446S_Q493R(r=None):  # R0025
    return ps.simple_row_scorer(
        r, column='FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants')   

def _score_MLScore(r=None):
    return ps.simple_negative_row_scorer(r, column='ML Score')  # Positive is good (i.e., higher probability)


def _score_NumMutations(r=None):
    return ps.simple_row_scorer(r, column='Num Mutations')  # Positive is bad (i.e., more mutations)


def _score_FEP_Stability(r=None):
    return ps.simple_row_scorer(r, column=_FEP_COLUMN_NAME)  # Positive is bad (i.e., less stable)


def _score_sfe_Q493K(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_Q493K_SFE_DDG')


def _score_sfe_Q493R_7l7e(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_Q493R_SFE_7l7e_DDG')


def _score_sfe_Q493R_7t9k(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_Q493R_SFE_7t9k_DDG')


def _score_sfe_Q493R_priority7t9k(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_Q493R_SFE_priority_DDG')


def _score_sfe_R346K_7l7e(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_R346K_SFE_7l7e_DDG')


def _score_sfe_R346K_7t9k(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_R346K_SFE_7t9k_DDG')


def _score_sfe_R346K_priority7t9k(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_R346K_SFE_priority_DDG')


def _score_pulling_R346K_G446S_Q493K(r=None):
    return ps.simple_row_scorer(r, column='VARIANT_R346K,Q493K_dGPullingAvg')



def downselect_by_greatest_weighted_nansum(df, quota=100, name='DownSelected', required='Selected', columns=None, weights=None):
    """
    From required column, choose a subset as the top quota many of the weighted nansum.
    """

    df_tmp = df.copy()

    # Compute the down-selection score
    df_tmp['downselection_score'] = [0.0 for i in range(df_tmp.shape[0])]
    for coli, weighti in zip(columns, weights):
        df_tmp['downselection_score'] = [np.nansum([dsj, weighti * coli_j]) for dsj, coli_j in zip(df_tmp['downselection_score'], df_tmp[coli])]

    df_tmp['downselection_score'] = [np.NaN if not selected_i else dsi for dsi, selected_i in zip(df_tmp['downselection_score'], df_tmp[required])]

    # Select the top quota-many of these and mark them.
    selection_seq = df_tmp['downselection_score'].copy()
    best_indices = []
    for selectedidx in range(quota):
        try:
            best_indices.append(np.nanargmax(selection_seq.values))
        except ValueError:  # if all np.nan, break
            break
        selection_seq.iloc[best_indices[-1]] = np.nan


    df_tmp[name] = [True if ii in best_indices else False for ii in range(df_tmp.shape[0])]

    return df_tmp


def select_most_negative_by_column_with_quota_and_limit(df, name=None, required=None, columns=None, quotas=None, limit_column=None, limit_val=None):
    df_tmp = df.copy()

    df_tmp[name] = [False for i in range(df_tmp.shape[0])]
    for col_i, quota_i in zip(columns, quotas):
        best_indices = []
        selection_seq = df_tmp[col_i].copy()
        # Set those violating the condition to nan
        selection_seq = [np.nan if limval_j > limit_val or not reqj else ssj for limval_j, ssj, reqj in zip(df_tmp[limit_column], selection_seq, df_tmp[required])]
        for selectedidx in range(quota_i):
            try:
                best_indices.append(np.nanargmin(selection_seq))  # Note argmin vs. other argmax
            except ValueError:  # if all np.nan, break
                break
            selection_seq[best_indices[-1]] = np.nan
        # Set to selected
        indexer = [df_tmp.index[bi] for bi in best_indices]
        df_tmp.loc[indexer, name] = True
    
    return df_tmp


if __name__ == '__main__':

    path = '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/vaccines_b_v2130_20220112_merged_results_all.pkl'
    master_fasta_path = '/g/g14/tadesaut/GitRepositories/advanced_improv_system_test/abag_agent_setup/studies/nCoV/vanderbilt_2021_06_04/v2130/RBD_2130_FAB.fasta'
    heavy_chain_length = 130  # Used below in determining which mutation positions are in each chain; also to break into heavy and light
    lmg_file_path = '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/merged_results_v2130_20211229_800_renamed_SinglePointsfixed.csv'  # merged_results_v2130_20211223_single_combied.csv'
    menu_file_path = '/g/g14/tadesaut/GitRepositories/advanced_improv_system_test/abag_agent_setup/studies/nCoV/vanderbilt_2021_06_04/v2130/menu_v2130_Omicron_conversion.yaml'
    dict_of_reference_fastas = {
        'JUST_StableLines_12': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/output_20220111_JUST.fasta',
        'Vanderbilt_small20': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/output_20220112_Vanderbilt_best20.fasta',
        'pareto_20211230_203': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/output_main203_20211230.fasta',
        'sfe_20211230_30': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/sfe30.fasta',
        'AAlpha_1129': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/output_main203_20211230_set_all_noSR.fasta',
        'AdamSFE_20220113_51Proposed': '/p/lustre1/tadesaut/pareto_selection_workspace/v2130_omicron/adam51_20220113.fasta'
    }

    id_col_name = 'Unnamed: 0'
    selected_col_names = ['DownSelected', 'UpSelectedFromDownSelected', 'ParetoSet', 'Selected_UnderrepresentedSet']  # TODO: Update
    path = os.path.abspath(path)

    # Create output path
    output_path = os.path.join(os.path.split(path)[0], os.path.splitext(os.path.split(path)[1])[0] + '_paretoselected_Vanderbilt184_20220113_prelimForDan.csv') 
    output_fasta_base_path = os.path.join(os.path.split(path)[0], os.path.splitext(os.path.split(path)[1])[0] + '_set_prelim_{}.fasta')
    print('Running pareto selection for v2130 set, using input file at {}, '
          'sequence ID column {}, and writing to {}.'.format(path, id_col_name, output_path))     


    # ### Load inputs ###

    # Master sequence from .fasta:
    master_seq_as_str = recast_sequence_as_str(list_of_seqrecords_from_fasta(master_fasta_path)[0])

    # Sets of selected sequences from other .fastas
    # Appropriate columns created below
    dict_of_reference_seqrecords = {
        ki: list_of_seqrecords_from_fasta(vi) 
        for ki, vi in dict_of_reference_fastas.items() 
        if vi is not None and vi != ''
        }
    
    hash_id_to_humid_dict = {}
    for list_of_srs_i in dict_of_reference_seqrecords.values():  # Technically slightly wrong to do this depending on order of iteration for stability
        for srij in list_of_srs_i:
            if 'HumID: ' in srij.description:
                hash_id_to_humid_dict[derive_idstr_from_seq(srij)] = srij.description.split('HumID: ')[-1].split(' ')[0]  # Split out the space-separated string following 'HumID: '; breaks if two spaces after 'HumID:'
        
    # Sequences dataframe:
    try:
        df = pd.read_pickle(path)
    except Exception as e:  # narrow this
        df = pd.read_csv(path)

    # LMG sampling frequencies
    sampling_frequencies_df = pd.read_csv(lmg_file_path)

    # Menu:
    menu = get_menu(menu_file_path)
    if len(menu['studies']) > 1:
        print('Menu must have only one set of studies allowed. Reducing to just one.')
        menu['studies'] = menu['studies'][:1]
    allowed_mutations_from_menu = menu['studies'][0]['study_parameters'][AllowedMutationsKey]['AllowedMutations']
    allowed_mutation_positions_from_menu = [ai[0] for ai in allowed_mutations_from_menu]
    # SPECIAL ADDITION:
    allowed_mutations_from_menu.append([165, 'N', ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']])
    allowed_mutations_from_menu.append([112, 'G', ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']])
    allowed_mutation_positions_from_menu = [ai[0] for ai in allowed_mutations_from_menu]

    # ### Size reduction and cleanup ###
    print(df.head())
    ros_r0024_column = 'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_rosetta_listed_mutants_fl'
    ros_r0025_column = 'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_rosetta_listed_mutants_fl'
    notnan_ros_r0024 = [not logi_nan for logi_nan in df[ros_r0024_column].isna()]
    notnan_ros_r0025 = [not logi_nan for logi_nan in df[ros_r0025_column].isna()]
    
    # DROPPING FILTERING
    # df = df.loc[[notnan_ros_r0024_i or notnan_ros_r0025_i for notnan_ros_r0024_i, notnan_ros_r0025_i in zip(notnan_ros_r0024, notnan_ros_r0025)]]

    # Cleanup
    # This shouldn't be necessary, but if it is, you'll really need it:
    if conventional_id_col not in df.columns and id_col_name not in df.columns:
        print('!!! WARNING: RECREATING SEQUENCE ID VALUES!!!')
        df[id_col_name] = [derive_idstr_from_seq(si) if isinstance(si, str) else '' for si in df[conventional_seq_col]]

    df.rename(columns={id_col_name: conventional_id_col}, inplace=True)  # This is still wrong, but conventional
    df.drop(index=[idxi for idxi, vi in zip(df.index.values, df[conventional_id_col]) if not (isinstance(vi, str) and vi.startswith('md5_'))], inplace=True)

    # Calculate which mutants are heavy chain only, and which are light chain only
    df['MutLists'] = [[tuple(si.lstrip(',')[1:].split(',')) for si in mj.split(')')][:-1] if not isinstance(mj, float) else None for mj in df[conventional_mutations_col] ]
    df['MutPositions'] = [[int(ti[1]) for ti in mlistj] if mlistj else [] for mlistj in df['MutLists']]
    df['MutPositionsAllLight'] = [all([mposlisti_j > heavy_chain_length for mposlisti_j in mposlisti]) for mposlisti in df['MutPositions']]
    df['MutPositionsAllHeavy'] = [all([mposlisti_j <= heavy_chain_length for mposlisti_j in mposlisti]) for mposlisti in df['MutPositions']]
    # df['MutPositions446ContactSet'] = [sum([mutposlisti_j in _G446S_CONTACT_SET for mutposlisti_j in mposlisti]) for mposlisti in df['MutPositions']]
    # df['MutPositions498ContactSet'] = [sum([mutposlisti_j in _Q498R_CONTACT_SET for mutposlisti_j in mposlisti]) for mposlisti in df['MutPositions']]
    df['MutPositionsUnderrepresented'] = [sum([mutposlisti_j in _underrepresented_position_set for mutposlisti_j in mposlisti]) for mposlisti in df['MutPositions']]

    all_mutation_positions_in_df = set().union(*df['MutPositions'].to_list())
    # !!! Force two positions in !!!
    if 165 not in all_mutation_positions_in_df:
        all_mutation_positions_in_df.add(165)
    if 112 not in all_mutation_positions_in_df:
        all_mutation_positions_in_df.add(112)
    
    all_mutation_positions_df_and_menu = sorted(list(all_mutation_positions_in_df.union(set(allowed_mutation_positions_from_menu))))

    for mutpos_i in all_mutation_positions_df_and_menu:
        # Construct the column for this position
        df[mutpos_i] = [
            [tupijk[-1] for tupijk in muttupij if int(tupijk[1]) == mutpos_i][0] 
            if mutposlistij and mutpos_i in mutposlistij else '' 
            for mutposlistij, muttupij in zip(df['MutPositions'], df['MutLists'])
        ]

    # Size reduction
    # print('!!! REDUCING DATASET SIZE FOR TESTING !!!')
    # df = df.loc[[random.uniform(0.0, 1.0) <= 0.01 for i in range(df.shape[0])]]
    # df = df.head(400)

    # Print some information:
    print('Of the {} sequences entering the pareto selection process, {} are light-chain only and '
          '{} are heavy-chain only.'.format(df.shape[0], df['MutPositionsAllLight'].sum(),df['MutPositionsAllHeavy'].sum()))

    print('dataframe columns are:')
    for ci in df.columns:
        print(ci)

    # ### Define dataframe-specific columns of interest ###
    # columns_of_interest = [
    #     'WT_RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_'
    #     'MD5_ALL_3dd5889a89cf8b317451aa947b28a6e8_'
    #     'rosetta_listed_mutants_fl'
    # ]
    # Rosetta columns:
    # ros_columns_of_interest = [ci for ci in df.columns if 'RosettaFlexDDGAvg' in ci and '_additive' not in ci]
    ros_columns_of_interest = [
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_rosetta_listed_mutants_fl',  # VMUT15; L452R
        ros_r0024_column,  # R0024; R346_G446S_Q493R
        ros_r0025_column  # R0025; R346K_G446S_Q493R
    ] 
    # Add FoldX columns:
    fx_columns_of_interest = [
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants', # VMUT15; L452R
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants', # R0024; R346_G446S_Q493R
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants' # R0025; R346K_G446S_Q493R
    ]
    # Add pulling, SFE, ddGPMX, FEP, 
    heavyweight_columns_of_interest = [
        'VARIANT_R346K,Q493K_dGPullingAvg',  
        'VARIANT_Q493R_SFE_7l7e_DDG',
        'VARIANT_Q493R_SFE_7t9k_DDG',
        'VARIANT_Q493R_SFE_priority_DDG',
        'VARIANT_R346K_SFE_7l7e_DDG',
        'VARIANT_R346K_SFE_7t9k_DDG',
        'VARIANT_R346K_SFE_priority_DDG'
        ]  
    
    aux_columns_of_interest = [_FEP_COLUMN_NAME, 'Num Mutations', 'ML Score'] # 'ML Score' is Language model, multi-points; +native is better, because it's a log likelihood; WT score has been subtracted off, so some are positive.

    columns_of_interest = ros_columns_of_interest + fx_columns_of_interest + \
        heavyweight_columns_of_interest + aux_columns_of_interest   
    

    # ### Begin main-phase processing ###
    print('WARNING: COERCING COLUMNS OF INTEREST TO BE FLOATS')  # TODO: Evaluate removal
    for ci in columns_of_interest:
        print(ci)
        df[ci] = df[ci].astype(float)

    epsilon_dict = {'ML Score': 0.2,
                    'Num Mutations': 0.0} # unwilling to accept more mutations
    for ci in columns_of_interest:
        if ci.lower().startswith('foldx'):
            epsilon_dict[ci] = 0.0  # There are just so few foldx values that no dominance appears to be occurring among them. Eps > 0 makes for silly results.
    
    # epsilons = [epsilon_dict[ci] if ci in epsilon_dict.keys() else 0.05 for ci in columns_of_interest] # ML Score is in numpy natural log
    epsilons = [0.0 for ci in columns_of_interest]
    print('Columns of interest are:')
    print(columns_of_interest)
    print('With epsilons:')
    print(epsilons)
    print('The number of values in each is:')
    for ci in columns_of_interest:
        print('{}: {}'.format(ci, df.shape[0] - df[ci].isna().sum()))
    time.sleep(1.0)

    dom_funcs = []
    for coi_i in columns_of_interest:
        dom_funcs.append(
            lambda r=None, c=coi_i: dom_func_scalar(row=r, column=c)
        )


    ros_fl_to_domfunc_dict = {
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_rosetta_listed_mutants_fl': _score_rosfl_L452R,
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_rosetta_listed_mutants_fl': _score_rosfl_R346_G446S_Q493R, # R0024; R346_G446S_Q493R
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_rosetta_listed_mutants_fl': _score_rosfl_R346K_G446S_Q493R  # R0025; R346K_G446S_Q493R

    }        

    fx_int_to_domfunc_dict = {
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants': _score_fxi_L452R,
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants': _score_fxi_R346_G446S_Q493R, # R0024; R346_G446S_Q493R
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants': _score_fxi_R346K_G446S_Q493R # R0025; R346K_G446S_Q493R
    }

    hw_to_domfunc_dict = {
        'VARIANT_Q493R_SFE_7l7e_DDG': _score_sfe_Q493R_7l7e,
        'VARIANT_Q493R_SFE_7t9k_DDG': _score_sfe_Q493R_7t9k,
        'VARIANT_Q493R_SFE_priority_DDG': _score_sfe_Q493R_priority7t9k,
        'VARIANT_R346K_SFE_7l7e_DDG': _score_sfe_R346K_7l7e,
        'VARIANT_R346K_SFE_7t9k_DDG': _score_sfe_R346K_7t9k,
        'VARIANT_R346K_SFE_priority_DDG': _score_sfe_R346K_priority7t9k,
        'VARIANT_R346K,Q493K_dGPullingAvg': _score_pulling_R346K_G446S_Q493K
    }

    aux_to_domfunc_dict = {
        _FEP_COLUMN_NAME: _score_FEP_Stability,
        'ML Score': _score_MLScore,
        'Num Mutations': _score_NumMutations
    }

    dom_funcs_par = [ros_fl_to_domfunc_dict[ci] for ci in ros_columns_of_interest] + \
                    [fx_int_to_domfunc_dict[ci] for ci in fx_columns_of_interest] + \
                    [hw_to_domfunc_dict[ci] for ci in heavyweight_columns_of_interest] + \
                    [aux_to_domfunc_dict[ci] for ci in aux_columns_of_interest]
    
    # #######################################
    # ### Calculate (epsilon-) Pareto set ###
    # #######################################

    print('Beginning parallel ps.main_par_allblocks call at {}...'.format(datetime.datetime.now()))
    pareto_optimal_rows_par = ps.main_par_allblocks(
        df, dominance_functions=dom_funcs_par,  # dom_funcs_par is the non-lambda version
        scalar_epsilons=epsilons, blocksize=500, nprocs=120,
        verbose=True
    )
    
    print('    parallel ps.main_par_allblocks call completed at {}.'.format(datetime.datetime.now()))
    print('There are {} rows in the (epsilon-) Pareto-optimal '
          'set.'.format(len(pareto_optimal_rows_par)))

    df['Selected'] = [True if rii in pareto_optimal_rows_par else False for rii in range(df.shape[0])]

    # TODO: Add MutPositionsUnderrepresented
    print('Beginning parallel ps.main_par_allblocks call for underrepresented set ({}) at {}.'.format(_underrepresented_position_set, datetime.datetime.now()))
    pareto_optimal_rows_inner_indexing_underrepresented = ps.main_par_allblocks(
        df.loc[df['MutPositionsUnderrepresented'] > 0], dominance_functions=dom_funcs_par,  # dom_funcs_par is the non-lambda version
        scalar_epsilons=epsilons, blocksize=500, nprocs=120,
        verbose=True
    )
    indices_underrepresented = [ii for ii in df.loc[df['MutPositionsUnderrepresented'] > 0].index]  # These are the dataframe indices of the rows passed into the pareto call
    pareto_optimal_rows_outer_underrepresented = [idxi for ii, idxi in enumerate(indices_underrepresented) if ii in pareto_optimal_rows_inner_indexing_underrepresented]
    print('    parallel ps.main_par_allblocks call for Underrepresented completed at {}.'.format(datetime.datetime.now()))
    print('There are {} rows in the (epsilon-) Pareto-optimal '
          'set.'.format(len(pareto_optimal_rows_outer_underrepresented)))
    df['Selected_UnderrepresentedSet'] = [True if rii in pareto_optimal_rows_outer_underrepresented else False for rii in df.index]  


    if os.path.splitext(output_path)[-1] == '.csv':
        df.to_csv(os.path.splitext(output_path)[0] + '_interim.csv')
    else:
        df.to_pickle(os.path.splitext(output_path)[0] + '_interim.pkl')
    
    # #########################################################
    # ### Post-process to modify/reshape pareto set into an ###
    # ### appropriate set of selections                     ###
    # #########################################################

    df['Num Mutations Squared'] = [nmi ** 2 for nmi in df['Num Mutations']]


    selection_columns = [
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_rosetta_listed_mutants_fl',
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_rosetta_listed_mutants_fl',
        'RosettaFlexDDGAvg_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_rosetta_listed_mutants_fl',
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_d1da9718f8f3541f9ddd641da22b2c5e_foldx_listed_mutants',
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_6a4644eaf457d7a02dbeef30c08510bb_foldx_listed_mutants',
        'FoldXInterfaceDDG_md5_145202f53d1cb0e40c396732581c7078_MD5_ALL_614f14cc7b4d3a8f36ebf99ed751191b_foldx_listed_mutants',
        'VARIANT_R346K,Q493K_dGPullingAvg',
        # 'VARIANT_Q493R_SFE_7l7e_DDG',
        # 'VARIANT_Q493R_SFE_7t9k_DDG',
        'VARIANT_Q493R_SFE_priority_DDG',
        # 'VARIANT_R346K_SFE_7l7e_DDG',
        # 'VARIANT_R346K_SFE_7t9k_DDG',
        'VARIANT_R346K_SFE_priority_DDG',
        'VARIANT_Ab_FEP_ddG_stability',
        'Num Mutations Squared',
        'ML Score'
    ]
    selection_weights = [
        -1.0, -1.0, -1.0,  # Rosetta
        -0.5, -0.5, -0.5,  # FoldX  # TODO: Strongly consider setting to zero
        -1.5,  # Pulling
        # -1.0, -1.0,  # SFE non-"priority"
        -1.5,  # SFE priority R346
        # -1.0, -1.0,  # SFE non-"priority"
        -1.5,  # SFE priority R346K
        -0.5,  # FEP stability
        -0.5, # number of mutations  # TODO: Adjust downward
        2.0  # BERT score
    ]

    df = downselect_by_greatest_weighted_nansum(df, quota=55, name='DownSelected', required='Selected', columns=selection_columns, weights=selection_weights)
    print('Selected {} sequences in column {}.'.format(df['DownSelected'].sum(), 'DownSelected'))  

    df = downselect_by_greatest_weighted_nansum(df, quota=55, name='DownSelected_underrepresented_set', required='Selected_UnderrepresentedSet', columns=selection_columns, weights=selection_weights)
    print('Selected {} sequences in column {}.'.format(df['DownSelected_underrepresented_set'].sum(), 'DownSelected_underrepresented_set'))

    df['downselection_score'] = [0.0 for i in range(df.shape[0])]
    for coli, weighti in zip(selection_columns, selection_weights):
        df['downselection_score'] = [np.nansum([dsj, weighti * coli_j]) for dsj, coli_j in zip(df['downselection_score'], df[coli])]

    # Up-select:
    # (Supporting possible post-hoc alternatives similar to pre-selected sequences upon human review)
    # For each sequence in the down-selected set, enumerate several sequences in a "ball" around the sequence in question.
    # These should be small modifications of the original sequence, where these modifications prefer interesting single-point mutations
    # An LMG file is an appropriate way to specify the probability distribution here; something else might be reasonable as well.
    # For each enumerated sequence, check if it exists in the dataframe; if not, add a row appropriately.
    # This should borrow heavily from some of the other preselection code we've developed.
    # Reasonable choices:
    #   all single-point ablations (see preselect_seq_ablations in draft_day_preselection)
    #   LMG-like augmentation

    df['UpSelectedAblation'] = [ds is True or dsunderrepresented is True for ds, dsunderrepresented in zip(df['DownSelected'], df['DownSelected_underrepresented_set'])]
    df['UpSelectedLMG'] =      [ds is True or dsunderrepresented is True for ds, dsunderrepresented in zip(df['DownSelected'], df['DownSelected_underrepresented_set'])]
    print(df['UpSelectedLMG'].sum())
    df = ps.ablate_seqs_for_upsample(df, master_seq=master_seq_as_str, colname='UpSelectedAblation', verbose=True)
    print(df['UpSelectedLMG'].sum())
    df = ps.upsample_with_lmg(df, master_seq=master_seq_as_str, allowed_mutations=allowed_mutations_from_menu, sampling_frequencies=sampling_frequencies_df, colname='UpSelectedLMG', num_mutations_to_add=2, verbose=False)
    print(df['UpSelectedLMG'].sum())

    df['ParetoSet'] = df['Selected'].copy()
    df['UpSelectedFromDownSelected'] = [us_ablate or us_lmg for us_ablate, us_lmg in zip(df['UpSelectedAblation'], df['UpSelectedLMG'])]
    # df.drop(columns=['DownSelected','UpSelectedAblation','UpSelectedLMG'], inplace=True)

    for ki, list_of_srs_i in dict_of_reference_seqrecords.items():
        mutation_list_of_lists = [
            diff_seqs(recast_sequence_as_str(srij), master_seq_as_str)
            for srij in list_of_srs_i
        ]
        df = ps.add_rows_from_seq_list(
            df, ki, [recast_sequence_as_str(srij) for srij in list_of_srs_i], 
            mutation_list_of_lists=mutation_list_of_lists, 
            add_absent_sequences=True,
            verbose=False
            )

    for selected_col_name in selected_col_names + list(dict_of_reference_seqrecords.keys()):
        df[selected_col_name] = [True if scni is True else False for scni in df[selected_col_name]]  # effectively, coerce to bool

    
    ### Prepare for writeout by filling any important missing columns ###
    df['MutLists'] = [[tuple(si.lstrip(',')[1:].split(',')) for si in mj.split(')')][:-1] if not isinstance(mj, float) else None for mj in df[conventional_mutations_col] ]
    df['MutPositions'] = [[int(ti[1]) for ti in mlistj] if mlistj else [] for mlistj in df['MutLists']]
    df['MutPositionsAllLight'] = [all([mposlisti_j > heavy_chain_length for mposlisti_j in mposlisti]) if mposlisti else False for mposlisti in df['MutPositions']]
    df['MutPositionsAllHeavy'] = [all([mposlisti_j <= heavy_chain_length for mposlisti_j in mposlisti]) if mposlisti else False for mposlisti in df['MutPositions']]
    # df['MutPositions446ContactSet'] = [sum([mutposlisti_j in _G446S_CONTACT_SET for mutposlisti_j in mposlisti]) if mposlisti else np.NaN for mposlisti in df['MutPositions']]
    # df['MutPositions498ContactSet'] = [sum([mutposlisti_j in _Q498R_CONTACT_SET for mutposlisti_j in mposlisti]) if mposlisti else np.NaN for mposlisti in df['MutPositions']]
    df['MutPositionsUnderrepresented'] = [sum([mutposlisti_j in _underrepresented_position_set for mutposlisti_j in mposlisti]) if mposlisti else np.NaN for mposlisti in df['MutPositions']]
    
    # Add some other key pieces of information: Mutation columns, human IDs (if present), whether or not new NXST introduced, and a filtering on MD fitness
    for mutpos_i in all_mutation_positions_df_and_menu:
        # Construct the column for this position
        df[mutpos_i] = [
            [tupijk[-1] for tupijk in muttupij if int(tupijk[1]) == mutpos_i][0] 
            if mutposlistij and mutpos_i in mutposlistij else '' 
            for mutposlistij, muttupij in zip(df['MutPositions'], df['MutLists'])
        ]

    df['HumID'] = [
        hash_id_to_humid_dict[hashid_i] 
        if hashid_i in hash_id_to_humid_dict else '' 
        for hashid_i in df[conventional_id_col]
        ]

    df['NewNXST'] = check_new_nxst_vs_master(master_seq_as_str, df[conventional_seq_col])
    df['DanFilter'] = [
        mdddg_i <= -3.0 and mdnumvals_i >= 2 and cc_i <= 2 
        for mdddg_i, mdnumvals_i, cc_i 
        in zip(
            df['VARIANT_R346K,Q493K_dGPullingAvg'], 
            df['VARIANT_R346K,Q493K_dGPullingAvg_num_vals'], 
            df['Common Contacts']
            )
            ]

    
    # ################
    # ### TRIMMING ###
    # ################
    # By columns:
    # print('!!! WARNING NOT DROPPING ANY COLUMNS !!!')
    # columns_to_drop = [ci for ci in df.columns if ci not in _COLUMNS_EXPECTED_DF_OUT]  # TODO: Modify _COLUMNS_EXPECTED_DF_OUT
    # if len(columns_to_drop) > 0:
    #     print(columns_to_drop)
    # df = df[[ci for ci in _COLUMNS_EXPECTED_DF_OUT + [cj for cj in selected_col_names if cj not in _COLUMNS_EXPECTED_DF_OUT] if ci in df.columns]]
    cols_to_drop = [
    '(last_DDG)',
    'VARIANT_Ab_SFE_7l7e_DDG', 'VARIANT_Ab_SFE_7l7e_error', 'VARIANT_Ab_SFE_7l7e_confidence', 
    'VARIANT_Ab_SFE_7l7e_Model', 'VARIANT_Ab_SFE_7t9k_DDG', 'VARIANT_Ab_SFE_7t9k_error', 
    'VARIANT_Ab_SFE_7t9k_confidence', 'VARIANT_Ab_SFE_7t9k_Model',
    'VARIANT_Ab_dGPullingAvg', 'VARIANT_Ab_dGPullingN', 'VARIANT_Ab_dGPullingRaw', 
    'VARIANT_Ab_dGPulling1', 'VARIANT_Ab_dGPulling2', 'VARIANT_Ab_dGPullingMax', 
    'VARIANT_Ab_SFE_priority_Model', 'VARIANT_Ab_SFE_priority_DDG', 'VARIANT_Ab_SFE_priority_error', 
    'VARIANT_Ab_SFE_priority_confidence',
    'VARIANT_Q493K_SFE_7l7e_Model', 'VARIANT_Q493K_SFE_7t9k_DDG', 'VARIANT_Q493K_SFE_7t9k_error', 
    'VARIANT_Q493K_SFE_7t9k_confidence', 'VARIANT_Q493K_SFE_7t9k_Model', 'VARIANT_Q493K_FEP_ddG_stability', 
    'VARIANT_Q493K_FEP_ddG_stability_uncertainty', 'VARIANT_Q493K_dGPullingAvg', 'VARIANT_Q493K_dGPullingN', 
    'VARIANT_Q493K_dGPullingRaw', 'VARIANT_Q493K_dGPulling1', 'VARIANT_Q493K_dGPulling2', 
    'VARIANT_Q493K_dGPullingMax', 'VARIANT_Q493K_SFE_priority_Model',
    'VARIANT_Q493R_SFE_7t9k_Model', 'VARIANT_Q493R_FEP_ddG_stability', 
    'VARIANT_Q493R_FEP_ddG_stability_uncertainty', 'VARIANT_Q493R_dGPullingAvg', 'VARIANT_Q493R_dGPullingN', 
    'VARIANT_Q493R_dGPullingRaw', 'VARIANT_Q493R_dGPulling1', 'VARIANT_Q493R_dGPulling2', 
    'VARIANT_Q493R_dGPullingMax', 'VARIANT_Q493R_SFE_priority_Model',
    'VARIANT_R346K_SFE_7t9k_Model', 'VARIANT_R346K_FEP_ddG_stability', 
    'VARIANT_R346K_FEP_ddG_stability_uncertainty', 'VARIANT_R346K_dGPullingAvg', 'VARIANT_R346K_dGPullingN', 
    'VARIANT_R346K_dGPullingRaw', 'VARIANT_R346K_dGPulling1', 'VARIANT_R346K_dGPulling2', 
    'VARIANT_R346K_dGPullingMax', 'VARIANT_R346K_SFE_priority_Model',
    'VARIANT_R346K,Q493K_SFE_7l7e_DDG', 'VARIANT_R346K,Q493K_SFE_7l7e_error', 
    'VARIANT_R346K,Q493K_SFE_7l7e_confidence', 'VARIANT_R346K,Q493K_SFE_7l7e_Model', 
    'VARIANT_R346K,Q493K_SFE_7t9k_DDG', 'VARIANT_R346K,Q493K_SFE_7t9k_error', 
    'VARIANT_R346K,Q493K_SFE_7t9k_confidence', 'VARIANT_R346K,Q493K_SFE_7t9k_Model', 
    'VARIANT_R346K,Q493K_FEP_ddG_stability', 'VARIANT_R346K,Q493K_FEP_ddG_stability_uncertainty',
    'VARIANT_R346K,Q493K_dGPullingN', 'VARIANT_R346K,Q493K_dGPullingRaw', 'VARIANT_R346K,Q493K_dGPulling1', 
    'VARIANT_R346K,Q493K_dGPulling2',
    'VARIANT_R346K,Q493K_SFE_priority_Model', 'VARIANT_R346K,Q493K_SFE_priority_DDG', 
    'VARIANT_R346K,Q493K_SFE_priority_error', 'VARIANT_R346K,Q493K_SFE_priority_confidence',
    'MasterAntigenID', 'SeqNumberingMutation', 'StudyType', 'Emin', 'FoldXFinalDG', 'FoldXInterfaceDG', 
    'FoldXInterfaceDeltaDG', 'MasterAntigenDescription', 'MasterAntigenSequence', 'RosettaFlexDDGMax', 
    'RosettaFlexDDGMin', 'StructureHash', 'StructurePath', 'WT_FoldXFinalDG', 'WT_FoldXInterfaceDG', 
    'FoldXFinalDDG', 'StructureHashVal', 'StructureHashType', 'MasterAntigenSequenceHashStr'
    ]
    df.drop(columns=[ci for ci in cols_to_drop if ci in df.columns], inplace=True)

    
    # By rows:    
    # Best way to do this is to create a list of what we want to keep, then eliminate everything else
    # Want total number of rows to be ~20k. (supporting easy introspection)
    # Step 1: Determine the 20k best rows by downselect score
    # Step 2: Check other inclusion criteria
    #    Desired: (VARIANT_R346K,Q493K_dGPullingAvg <= -3) AND (VARIANT_R346K,Q493K_dGPullingAvg_num_vals >= 2) AND (Common Contacts <= 2)
    # Step 3: Ensure that ALL of the selection columns and input fasta columns are retained
    # Step 4: Union all of the filters, generating a list of bool, and apply df = df.loc[filters] 

    # df.to_csv('tmp.csv')
    if os.path.splitext(output_path)[-1] == '.csv':
        df.to_csv(output_path)
    else:
        df.to_pickle(output_path)

    # TODO: Check naming post-underrepresented (maybe not actually necessary here)
    for selected_col_name in selected_col_names:
        fasta_from_list_of_seqrecords(
            [
                recast_as_seqrecord(si, id_if_str_or_seq=idi) 
                for selected_i, si, idi in zip(df[selected_col_name], df[conventional_seq_col], df[conventional_id_col]) 
                if selected_i is True  # again, weirdness if Nans
                ],
                 output_fasta_base_path.format(selected_col_name), format='fasta-2line')
