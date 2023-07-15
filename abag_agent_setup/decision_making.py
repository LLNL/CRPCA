# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Module for going from menu and history to feature representations


"""

from __future__ import print_function, absolute_import, division

import pandas as pd
import os
from copy import deepcopy

from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    hash_pdb, load_pdb_structure_from_file
from vaccine_advance_core.featurization.seq_to_features import \
    make_cpx_dictionary, get_features_from_seqs_and_cpx_dict_par
from abag_agent_setup.expand_allowed_mutant_menu import derive_idstr_from_seq, \
    MasterAntigenSequenceKey


def _make_complex_quadruples_unique_by_struct(cn_cf_ch_cpc):
    """
    Ensure there is only one structure with each filename, hash combination.

    If the same filename and hash (but different fullpaths) exist, this will
    reduce to just one of the fullpaths.

    :param cn_cf_ch_cpc: list of quadruples, where each is (filename, fullpath,
        structure_name, assumed_partnered_chains).
    :return: cn_cf_ch_cpc with all but one element eliminated for each
        (filename, hash) pair.
    """

    keep_cn_cf_ch_cpc = [True for tup_i in cn_cf_ch_cpc]
    # for each name, check whether there exists only one hash; if so,
    # check if there exists greater than one file, else throw an error; if so,
    # choose one that is accessible.
    for name_i in list(set([tup_i[0] for tup_i in cn_cf_ch_cpc])):
        matches_name = [tup_i[0] == name_i for tup_i in cn_cf_ch_cpc]

        # Check that the hashes and assumed partnered chains are unique
        # That is, that we're really trying to use the same structure (by hash)
        # in the same "direction" (which chains are active/passive), just with
        # different fullpath.
        for nj, idj in [('hashes', 2), ('assumed_partnered_chains', 3)]:
            matching_nj = [tup_i[idj]
                          for mn_i, tup_i in zip(matches_name, cn_cf_ch_cpc)
                          if mn_i
                          ]
            if len(list(set(matching_nj))) != 1:
                raise ValueError(
                    'The matching {} for name {} are {}, rather than one '
                    'unique hash.'.format(nj, name_i, list(set(matching_nj)))
                )

        # There now exists just one unique hash for this name, but there may be
        # more than one file path. Keep only one, where this one is verified to
        # be accessible.
        accessible_found = False
        for i, mn_i, tup_i in zip(
                range(len(cn_cf_ch_cpc)), matches_name, cn_cf_ch_cpc):
            # Choose the first element of the list that has an accessible file
            if not mn_i:
                continue
            if accessible_found:
                # That is, we've already found an accessible file matching
                # this name, and we know from above these should all have the
                # `same hash
                keep_cn_cf_ch_cpc[i] = False
                continue
            # Try to access the file path
            if os.path.exists(tup_i[1]):
                accessible_found = True

    # And now, use the keep_... variables to rm some tuples from cn_cf_ch_cpc
    cn_cf_ch_cpc = [
        tup_i for k_i, tup_i in zip(keep_cn_cf_ch_cpc, cn_cf_ch_cpc) if k_i
    ]
    return cn_cf_ch_cpc


def _get_partnered_chains_from_structure(structure_path, active_chains):
    """
    Get chains from structure, placing all except active in context set

    :param structure_path: str, path to pdb file
    :param active_chain: str, comma-separated set of designators of
    :return: tuple of comma-separated strings, giving the partnered chains,
        (active, context).
    """

    structure = load_pdb_structure_from_file(structure_path)

    chain_ids = [ci.id for ci in structure.child_list[0].child_list]

    if not all([aci in chain_ids for aci in active_chains.split(',')]):
        raise ValueError('All of the putative active chains should appear in '
                         'the structure')

    # Compute the tuple of active and context chains
    context_chains = ','.join(
        [ci for ci in chain_ids if ci not in active_chains.split(',')]
    )

    # print('In structure {}, found the following chains:'.format(structure_path))
    # print('Active chains: {}'.format(active_chains))
    # print('Chain ids: {}'.format(chain_ids))
    # print('Remaining context chains: {}'.format(context_chains))

    return (active_chains, context_chains)


def make_cpx_dictionary_from_records(records,
                                     assumed_partnered_chains=None,
                                     pair_function=None,
                                     score_to_len_threshold=None,
                                     penalties=None):
    """
    Construct a cpx_dictionary from flattened history or expanded menu records

    TODO: Evaluate how to synchronize derivation of assumed_partnered_chains
      with _make_complex_quadruples_unique_by_struct. The first is file path-
      based and allows different active chains for the same complex, whereas
      the second is hash based and throws an error for different chain sets for
      the same complex.

    :param records: iterable with consistent order yielding dictionaries with
        'study_parameters', itself a dictionary containing 'MasterAntigenID',
        MasterAntigenSequenceKey, 'AntigenChainsInStructure', 'StructurePath',
        'StructureHash'. This iterable is, e.g., the list('history' entry in a
        flattened history) (post get_history()) or a 'studies' list from an
        expanded menu.
    :param assumed_partnered_chains: list, tuple of str, or None;
        if a tuple, the first element is the
        active chain set, subsequent elements are the context set(s), where
        each can be comma-separated. Valid choices include ('A', 'C') and
        ('A', 'H,L').  The structures should contain these chains. If a list,
        the list must be the same length as the records. If None, each element
        is automatically detected via the structure and the record's
        'AntigenChainInStructure' value.
    :param pair_function: str or None; specifies the type of interface
        detection to be used by cpx_dictionary creation. If none, defaults to
        'get_pairs_ca_ca'.
    :param score_to_len_threshold: float or None; specifies the minimum ratio
        of score to length acceptable for declaring two sequences to match in
        make_cpx_dictionary. If None, defaults to 0.4.
    :param penalties: None or list of four floats; specifies how the alignment
        should be computed.
    :return: a cpx_dictionary composed from the chosen records
    """

    if pair_function is None:
        pair_function = 'get_pairs_ca_ca'

    if score_to_len_threshold is None:
        score_to_len_threshold = 0.4

    if penalties is None:
        penalties = [-100, -10, -5, -0.1]

    # Set some constants:
    # Above: This does not seem like a good choice; maybe OK though since
    # we're going to be missing chunks of the structure in some cases, and
    # I believe that counts against the score
    if assumed_partnered_chains is None:
        # detect the chains by loading the individual structures and using the
        # values in the records to determine the active chain
        path_and_active_chain_to_pc = {}
        assumed_partnered_chains = []
        # count = 0
        for ri in records:
            t = (ri['study_parameters']['StructurePath'],
                 ri['study_parameters']['AntigenChainsInStructure'])
            if t not in path_and_active_chain_to_pc.keys():
                path_and_active_chain_to_pc[t] = \
                    _get_partnered_chains_from_structure(
                            ri['study_parameters']['StructurePath'],
                            ri['study_parameters']['AntigenChainsInStructure']
                        )
                # count = count + 1
            assumed_partnered_chains.append(path_and_active_chain_to_pc[t])

        # print(count)

    elif isinstance(assumed_partnered_chains, list):
        if len(assumed_partnered_chains) != len(records) or not all(
                [isinstance(apc_i, tuple) for apc_i in assumed_partnered_chains]
        ):
            raise ValueError('assumed_partnered_chains must be None '
                             '(detect automatically), one tuple, or a list '
                             'of tuples of the same length as records')
    elif isinstance(assumed_partnered_chains, tuple):
        assumed_partnered_chains = [assumed_partnered_chains for ri in records]
    else:
        raise ValueError('assumed_partnered_chains must be None '
                         '(detect automatically), one tuple, or a list '
                         'of tuples of the same length as records')

    # all_assumed_chains = ''.join(
    #     [''.join(apci.split(',')) for apci in assumed_partnered_chains])
    #
    # # Check that the data matches the current, fHbp assumption on chains:
    # check_chains = all([all([vij in all_assumed_chains for vij in
    #                          vi['study_parameters'][
    #                              'AntigenChainsInStructure']]) for vi in
    #                     records])
    # if not check_chains:
    #     raise ValueError(
    #         'Non-renum_6nb6_nCoV values not yet implemented for'
    #         'complex_dictionary creation!')
    # # Collate the inputs from the records

    # On the master antigen side:
    mads_mas_list = list(set([
        (vi['study_parameters']['MasterAntigenID'],
         vi['study_parameters'][MasterAntigenSequenceKey])
        for vi in records
    ]))

    # No master antigen sequence should apear in more than one tuple above;
    # Same for id values.
    master_antigen_designators = [mi[0] for mi in mads_mas_list]
    master_antigen_sequences = [mi[1] for mi in mads_mas_list]
    if len(list(set(master_antigen_designators))) != len(mads_mas_list) or \
            len(list(set(master_antigen_sequences))) != len(mads_mas_list):
        raise ValueError('The same master antigen ID appears with more '
                         'than one sequence or vice versa!')

    # And on the complex side:
    cn_cf_ch_cpc = list(set([
        (os.path.split(vi['study_parameters']['StructurePath'])[-1],
         vi['study_parameters']['StructurePath'],
         tuple(vi['study_parameters']['StructureHash']),
         apc_i)  # TODO: eliminate assumed_partnered_chains
        for vi, apc_i in zip(records, assumed_partnered_chains)
    ]))

    cn_cf_ch_cpc = _make_complex_quadruples_unique_by_struct(cn_cf_ch_cpc)

    # The old reorganization:
    complex_names = [ci[0] for ci in cn_cf_ch_cpc]
    complex_pdb_files = [ci[1] for ci in cn_cf_ch_cpc]
    complex_hashes = [ci[2] for ci in cn_cf_ch_cpc]
    complex_partnered_chains = [ci[3] for ci in cn_cf_ch_cpc]

    # (now-redundant) check for uniqueness of names/pdb_files
    if len(list(set(complex_names))) != len(cn_cf_ch_cpc) or \
            len(list(set(complex_pdb_files))) != len(cn_cf_ch_cpc):
        raise ValueError('The same complex name appears with more '
                         'than one complex file or vice versa!')

    # Make the cpx_dictionary
    # TODO: Adapt make_cpx_dictionary to new form assuming 1-to-1 pairing
    cpx_dictionary = make_cpx_dictionary(
        master_antigen_designators,
        master_antigen_sequences,
        complex_names,
        complex_pdb_files, pair_function=pair_function,
        partnered_chains=complex_partnered_chains,
        score_to_len_threshold=score_to_len_threshold,
        penalties=penalties
    )
    return cpx_dictionary


def get_records_from_h_or_m(h_or_m):
    # For each entry in the history or menu, get a feature representation
    # Construct list of sequences, list of masters IDs, and list of lists of
    # complexes
    if 'history' in h_or_m:
        # this is a history dictionary
        records = [vi for vi in h_or_m['history'].values()]
    elif 'studies' in h_or_m:
        records = h_or_m['studies']
    else:
        raise ValueError(
            'The h_or_m input to get_feature_representation_from_history_or'
            '_menu must be either a flattened history dictionary or an '
            'expanded menu dictionary.')

    return records


def get_target_records_from_records(
        records, target_study_type=None,
        target_ab_master_and_structures=None):
    """
    From a collection of records, generate target studies of a specified type

    :param records: list of study records, each of which is a dictionary;
        'study_type' and 'result' are among the keys
    :param target_study_type: str giving the study_type for the targets
    :param target_ab_master_and_structures: dictionary from structure ID (str)
        to:
        MasterAntigenSequenceKey: str, giving the sequence of the master antigen
        'MasterAntigenID': str, giving the ID of the master antigen
        'StructureID': str, giving structure ID, typically the file name
        'StructurePath': str, absolute path to the structure
        'StructureHash': list or tuple, as generated by vaccine_advance_core.
            featurization.vaccine_advance_core_io.hash_pdb, where the first
            element in the list/tuple is a hash string and the second is the
            hash type.
        'AntigenChainsInStructure': str, giving which chains in the structure
            are associated with the mutable antigen.
        'AntibodyID': str, giving the ID of the antibody.
    :return:
    """


    # ###############################################
    # ### Get the unique set of antigen sequences ###
    # ###############################################

    antigen_seqs = []
    for ri in records:
        antigen_seqs.append(ri['study_parameters']['AntigenSequence'])
    antigen_seqs = list(set(antigen_seqs))

    # #############################################
    # ### Generate complementary target records ###
    # #############################################

    tmp_records = []
    for si in antigen_seqs:
        for kk, vi in target_ab_master_and_structures.items():
            tmp_records.append(
                {'request_id': '',
                 'study_type': target_study_type,
                 'study_parameters': {
                     'AntigenSequence': si,
                     'MasterAntigenSequence': vi[MasterAntigenSequenceKey],
                     'MasterAntigenID': vi['MasterAntigenID'],
                     'StructureID': vi['StructureID'],
                     'StructureHash': vi['StructureHash'],
                     'StructurePath': vi['StructurePath'],
                     'AntigenChainsInStructure': vi['AntigenChainsInStructure']
                 }
                 }
            )

    # #####################################################################
    # ### Replace any cases where the target has actually been observed ###
    # #####################################################################

    for rj in records:
        if rj['study_type'] == target_study_type:
            replace = [False for _ in tmp_records]
            for i, tri in enumerate(tmp_records):
                check_list = [True]
                for kk in tri['study_parameters'].keys():
                    if kk == 'StructurePath':
                        continue
                    elif kk == 'StructureHash':
                        check_list.append(
                            all([vi == vj for vi, vj in
                                 zip(tri['study_parameters'][kk],
                                     rj['study_parameters'][kk])
                                 ])
                        )
                    else:
                        check_list.append(
                            tri['study_parameters'][kk] ==
                            rj['study_parameters'][kk]
                        )
                if all(check_list):
                    replace[i] = True
            if any(replace):
                tmp_records = [
                    tri for tri, replace_i in zip(tmp_records, replace)
                    if not replace_i
                ]
                tmp_records.append(rj)

    return tmp_records


def get_feature_representation_from_history_or_menu(
        h_or_m, cpx_dictionary=None, cpx_name_to_ab_id=None,
        feature_types=None):
    """
    Given a menu/history, get features for its records

    :param h_or_m:  If history, a dictionary where the 'history' entry is a
        dictionary of individual study records (dictionaries), keyed by
        request_id values. If menu, a dictionary where 'studies' contains a
        list of potential studies (dictionaries).
    :param cpx_dictionary: either None or a Vaccine_advance_core
        complex_dictionary. If None, make a vaccine_advance_core
        complex_dictionary.
    :param cpx_name_to_ab_id: either None or a dictionary keyed from complex
        name/id to the str designating the antibody present in the complex.
    :param feature_types: None or a list of str. If None, defaults to size
        class and chemical class features.
    :return: features (DataFrame) and cpx_dictionary
    """

    if feature_types is None:
        feature_types = [
            'size_class_combinations_reversible',
            'chemical_class_combinations_reversible'
        ]
    if not isinstance(feature_types, list):
        raise ValueError('The input feature_types must either be None or a list!')

    records = get_records_from_h_or_m(h_or_m)

    # Build the complex dictionary:
    if cpx_dictionary is None:
        cpx_dictionary = make_cpx_dictionary_from_records(records)

    # Prep
    study_types = [vi['study_type'] for vi in records]
    results = [vi['result'] if 'result' in vi.keys() else {} for vi in records]
    results_columns = set()
    for ri in results:
        results_columns = results_columns.union(set(ri.keys()))
    results_columns = sorted(list(results_columns))
    seqs = [vi['study_parameters']['AntigenSequence'] for vi in records]
    master_keys = [vi['study_parameters']['MasterAntigenID'] for vi in records]
    lists_of_cpx_names= [
        [os.path.split(vi['study_parameters']['StructurePath'])[-1]]  # TODO: hash?
        for vi in records
    ]

    # Get the features
    features = get_features_from_seqs_and_cpx_dict_par(
        seqs, master_keys, cpx_dictionary, feature_types,
        lists_of_cpx_names=lists_of_cpx_names)

    # Augment the features with some extra values
    if cpx_name_to_ab_id is not None:
        features['AntibodyID'] = [cpx_name_to_ab_id[complex_name_i]
                                  for complex_name_i in features['Complex']]
    features['AntigenID'] = [derive_idstr_from_seq(seq_i) for seq_i in features['AntigenSequence']]
    features['StudyType'] = study_types

    for rci in results_columns:
        if rci in features.columns:
            continue  # Avoid overwriting
        features[rci] = [ri[rci] if rci in ri.keys() else None for ri in results]

    return features, cpx_dictionary


def get_target_features_from_history_or_menu(
        h_or_m, target_ab_master_and_structures=None, target_study_type=None,
        cpx_dictionary=None, feature_types=None):
    """
    Given a menu/history, get features for its corresponding targets

    :param h_or_m: an abag_agent_setup-standard menu or history object
    :param target_ab_and_structures:
    :param target_study_type: str, giving the type of the (possibly
        inaccessible) target studies.  We may have history results of this type.
    :param target_study_type_results_columns: list of str, giving the column
        names for the results of the target type study
    :param feature_types: None or a list of feature types
    :return:
    """
    if feature_types is None:
        feature_types = [
            'size_class_combinations_reversible',
            'chemical_class_combinations_reversible'
        ]
    if not isinstance(feature_types, list):
        raise ValueError('The input feature_types must either be None or a list!')

    records = get_records_from_h_or_m(h_or_m)
    records = get_target_records_from_records(
        records,
        target_study_type=target_study_type,
        target_ab_master_and_structures=target_ab_master_and_structures
    )

    # Build the complex dictionary:
    if cpx_dictionary is None:
        cpx_dictionary = make_cpx_dictionary_from_records(records)

    # Prep
    study_types = [vi['study_type'] for vi in records]
    # TODO: Change below
    results = [vi['result'] if 'result' in vi.keys() else {} for vi in records]
    results_columns = set()
    for ri in results:
        results_columns = results_columns.union(set(ri.keys()))
    results_columns = sorted(list(results_columns))
    seqs = [vi['study_parameters']['AntigenSequence'] for vi in records]
    master_keys = [vi['study_parameters']['MasterAntigenID'] for vi in records]
    lists_of_cpx_names= [
        [os.path.split(vi['study_parameters']['StructurePath'])[-1]]  # TODO: hash?
        for vi in records
    ]

    # Get the features
    features = get_features_from_seqs_and_cpx_dict_par(
        seqs, master_keys, cpx_dictionary, feature_types,
        lists_of_cpx_names=lists_of_cpx_names)

    # Augment the features with some extra values
    cpx_name_to_ab_id = {vi['StructureID']: vi['AntibodyID'] for vi in target_ab_master_and_structures.values()}
    features['AntibodyID'] = [cpx_name_to_ab_id[complex_name_i] if complex_name_i in cpx_name_to_ab_id.keys() else None
                              for complex_name_i in features['Complex']]
    features['AntigenID'] = [derive_idstr_from_seq(seq_i) for seq_i in features['AntigenSequence']]
    features['StudyType'] = study_types

    # TODO: Evaluate
    for rci in results_columns:
        if rci in features.columns:
            continue  # Avoid overwriting
        features[rci] = [ri[rci] if rci in ri.keys() else None for ri in results]

    return features, cpx_dictionary


if __name__ == '__main__':
    pass