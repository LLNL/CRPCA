# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Module for handling menus containing master sequences, associated structures,
and allowed mutation information.

The core functions here should parallel the functions used in standard Improv
menu handling, especially expand_menu_study_params.
"""
from __future__ import division, print_function, absolute_import

from copy import copy, deepcopy
import random
import os
import yaml
import csv
import hashlib
import numpy as np
import shutil
import re
from filelock import Timeout

from Bio.PDB.Polypeptide import aa1  # Used to establish the list of AAs
import Bio.SeqIO.FastaIO as FastaIO

from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    list_of_seqrecords_from_fasta, fasta_from_list_of_seqrecords, hash_pdb, \
    hash_fasta_or_yaml
from vaccine_advance_core.featurization.bio_sequence_manipulation import \
    parse_seqrecord_description, recast_sequence_as_str, recast_as_seqrecord, \
    write_description_for_seqrecord
from vaccine_advance_core.featurization.seq_to_features import mutate_seq, \
    diff_seqs
from vaccine_advance_core.statium_automation.sequence_manipulation import \
    selected_mutations_mut_from_list_of_tuples
from improvwf.utils import yaml_safe_load_with_lock, yaml_safe_dump_with_lock, \
    write_selected_study
from abag_agent_setup.mutant_generator_sampling import generate_linear_mutants, \
    override_single_point_df
try:
    from improvwf.utils import get_history_db as get_history_db_improv
except ImportError:
    def get_history_db_improv(db_url, nrandom=None):
        raise ImportError('Unable to import get_history_db from improvwf! '
                          'Check version.')


MasterAntigenSequenceKey = 'MasterAntigenSequence'
AllowedMutationsKey = 'AllowedMutations'


INPUT_MENU_STUDY_PARAMETERS_KEY_STRUCTURE = {
    'MasterAntigenID': str,
    MasterAntigenSequenceKey: str,
    'MasterAntigenFASTAHash': (tuple, list),
    'MasterAntigenFASTAPath': str,
    'Structure': {
        'StructureID': str,
        'StructurePath': str,
        'StructureHash': (tuple, list),
        'AntigenChainsInStructure': str
    },
    AllowedMutationsKey: {
        'SVAPath': str,
        'SVAHash': (tuple, list),
        'AllowedMutations': list
    }
    }


INTERMEDIATE_MENU_STUDY_PARAMETERS_KEY_STRUCTURE = {
    'MasterAntigenID': str,  # y  Parameters marked with y check with the history flattening study_parameters block
    MasterAntigenSequenceKey: str,  # y
    'MasterAntigenFASTAHash': (tuple, list),  # y
    'MasterAntigenFASTAPath': str,  # y
    'StructureID': str,  # y
    'StructurePath': str,  # y
    'StructureHash': (tuple, list),  # y
    'AntigenChainsInStructure': str,  # y
    'AntigenSequence': str  # y
}


# Specify the structure of keys in the output menu
OUTPUT_MENU_STUDY_PARAMETERS_KEY_STRUCTURE = {
    'MasterAntigenID': str,
    MasterAntigenSequenceKey: str,
    'MasterAntigenFASTAHash': (tuple, list),
    'MasterAntigenFASTAPath': str,
    'StructureID': str,
    'StructureHash': (tuple, list),
    'StructurePath': str,
    'AntigenChainsInStructure': str,
    'AntigenFASTAPath': str,
    'AntigenFASTAHash': (tuple, list)
    }

STUDY_WRITE_OUT_KEYS = {
    'MasterAntigenID': None,
    MasterAntigenSequenceKey: None,
    'MasterAntigenFASTAHash': 'MASTER_ANTIGEN_FASTA_HASH',
    'MasterAntigenFASTAPath': 'MASTER_ANTIGEN_FASTA_PATH',
    'StructureID': None,
    'StructureHash': 'STRUCTURE_HASH',
    'StructurePath': 'STRUCTURE_PATH',
    'AntigenChainsInStructure': 'ANTIGEN_CHAINS_IN_STRUCT',
    'AntigenFASTAPath': 'ANTIGEN_FASTA_PATH',
    'AntigenFASTAHash': 'ANTIGEN_FASTA_HASH'
}

_LIST_OF_AAS = sorted([aa1i for aa1i in aa1])  # matches vaccine_advance_core


def derive_idstr_from_seq(seq):
    seq = recast_sequence_as_str(seq)
    m = hashlib.md5()
    m.update(seq.encode('utf-8'))
    return 'md5_{}'.format(m.hexdigest())


def recursive_parameters_dict_check(parameters_struct_in, key_structure=None):
    """
    Check that a menu's ['studies'][i]['study_parameters'] value conforms

    :param parameters_struct_in: the 'study_parameters' component of a menu item
        or history entry (a dict), whether expanded or unexpanded. Contains
        entries for the specified parameters.
    :return: True if conforms, False otherwise
    """
    if key_structure is None:
        key_structure = {}

    conforms = True
    try:
        if isinstance(key_structure, dict):
            for ki, vi in key_structure.items():
                try:
                    try:
                        check_sub = isinstance(parameters_struct_in[ki], vi)
                    except KeyError:
                        check_sub = True  # OK if the key is absent
                        # TODO: evaluate whether OK if key absent
                except TypeError:
                    check_sub = recursive_parameters_dict_check(parameters_struct_in[ki], vi)
                if not check_sub:
                    conforms=False
                    break
        elif isinstance(key_structure, list):
            raise ValueError('key_structure currently may not be a list')
        else:
            raise ValueError('key_structure currently must be a nested series of dicts')
    except TypeError:
        conforms = False
    return conforms


def _override_master_mutant(master_sequence, allowed_mutations,
                            override_mutant_sequence=None, **kwargs):
    """
    Override the master mutant with a new mutant sequence

    :param master_sequence: str, giving the master sequence of amino acids
    :param allowed_mutations: list of triples or lists of (int, str, list), where these
        respectively are the integer position into the chain, the current amino
        acid value at that position, and the list of str, giving allowed values
        that residue can take.
    :param override_mutant_sequence: str, giving the OVERRIDING master sequence
        of amino acids
    :param kwargs: accept but do not use any other kwargs
    :return: override_master_sequence, overridden_allowed_mutations

    >>> _override_master_mutant('ABC', [[1, 'A', ['D', 'E']]], 'EBC')
    ('EBC', [[1, 'E', ['A', 'D', 'E']]])
    >>> _override_master_mutant('ABC', [[1, 'A', ['D', 'E']]])
    ('ABC', [[1, 'A', ['D', 'E']]])
    >>> _override_master_mutant('ABC', [[1, 'A', ['D', 'E']]], irrelevant=1)
    ('ABC', [[1, 'A', ['D', 'E']]])
    """

    # Handle the null case: no override
    if override_mutant_sequence is None:
        return master_sequence, allowed_mutations

    # Check if the override_mutant_sequence is allowed under the menu
    # List of tuples, where these are (, str(integerPosition), AAFrom, AATo)
    tups_difference_override = diff_seqs(override_mutant_sequence,
                                        master_sequence)
    tups_diff_override_ver_dict = {
        int(tupdi[1]): tupdi[3] for tupdi in tups_difference_override
    }
    allowed_dict = {tupi[0]: tupi[2] for tupi in allowed_mutations}

    for tupdi in tups_difference_override:
        if not tupdi[3] in allowed_dict[int(tupdi[1])]:
            raise ValueError('Override master_sequence mutation {} not allowed '
                             'in the menu!'.format(tupdi))

    # Modify the allowed_mutations to conform to the override sequence
    overridden_allowed_mutations = deepcopy(allowed_mutations)

    for ii in range(len(overridden_allowed_mutations)):
        # First, coerce to a list if it was a tuple
        if isinstance(overridden_allowed_mutations[ii], tuple):
            overridden_allowed_mutations[ii] = \
                list(overridden_allowed_mutations[ii])

        # Ensure that the master residue is allowed in the resulting list
        if overridden_allowed_mutations[ii][0] in tups_diff_override_ver_dict:
            if overridden_allowed_mutations[ii][1] not in \
                    overridden_allowed_mutations[ii][2]:
                # Add it to the front of the list
                overridden_allowed_mutations[ii][2] = \
                    [overridden_allowed_mutations[ii][1]] + \
                    overridden_allowed_mutations[ii][2]

            # Now modify the "From" residue to match the override sequence
            overridden_allowed_mutations[ii][1] = \
                tups_diff_override_ver_dict[overridden_allowed_mutations[ii][0]]

    # Return the new sequence and the new allowed mutations
    return override_mutant_sequence, overridden_allowed_mutations


def _generate_mutant(master_sequence, allowed_mutations, generator_type=None, **kwargs):
    """
    Generate a mutant from the selected generator function

    :param master_sequence: str, giving the master sequence of amino acids
    :param allowed_mutations: list of triples or lists of (int, str, list), where these
        respectively are the integer position into the chain, the current amino
        acid value at that position, and the list of str, giving allowed values
        that residue can take.
    :param kwargs: additional keyword args
    :return: str, giving the mutant sequence of amino acids
    """

    if generator_type is None:
        generator_type = 'poisson_uniform'

    # if the kwarg override_mutant_sequence is present, use it to override
    master_seq_to_use, allowed_mutations_to_use = _override_master_mutant(
        master_sequence, allowed_mutations, **kwargs
    )

    if generator_type == 'poisson_uniform':
        mutant_seq = _generate_mutant_poisson_uniform(master_seq_to_use, allowed_mutations_to_use, **kwargs)
    elif generator_type == 'weighted_mixture_of_conditionals':
        mutant_seq = _generate_mutant_weighted_mixture_of_conditionals(
            master_seq_to_use, allowed_mutations_to_use, **kwargs
        )
    else:
        raise ValueError('Unrecognized mutant generator type!')

    return mutant_seq


def _generate_mutant_poisson_uniform(master_sequence, allowed_mutations,
                                     n_locations_expected=None, **kwargs):
    """
    Generate a mutant sequence from the master, with uniform point mutations

    :param master_sequence: str, giving the master sequence of amino acids
    :param allowed_mutations: list of triples or lists of (int, str, list), where these
        respectively are the integer position into the chain, the current amino
        acid value at that position, and the list of str, giving allowed values
        that residue can take.
    :param n_locations_expected: integer, giving the expected number of
        locations to mutate; the actual number of locations will be drawn from
        a Poisson distribution with this expectation. If None, this behavior is
        not executed and instead, all locations are (potentially) mutated.
    :param kwargs: additional keyword args; ignored
    :return: str, giving the mutant sequence of amino acids
    """

    if n_locations_expected is not None:
        # Try to mutate at at least one location, but no more than the len(allowed_mutations)
        n_locations_to_mutate = np.max([1, np.min([len(allowed_mutations), np.random.poisson(n_locations_expected)])])
    else:
        n_locations_to_mutate = len(allowed_mutations)

    # Shuffle the indices, and take n_locations_to_mutate from them
    locations_to_mutate = np.random.permutation(len(allowed_mutations))[:n_locations_to_mutate]

    # TODO: check for redundancy
    mutations_to_apply = []
    for idx, (res_loc_i, cur_res_i, allowed_res_list_i) in enumerate(allowed_mutations):
        if idx not in locations_to_mutate:
            continue
        tmp_list = deepcopy(allowed_res_list_i)
        if cur_res_i not in tmp_list:
            tmp_list = [cur_res_i] + tmp_list
        chosen_res_i = random.choice(tmp_list)
        mutations_to_apply.append(('', str(res_loc_i), cur_res_i, chosen_res_i))

    mutant_seq = mutate_seq(master_seq=master_sequence, mutations=mutations_to_apply)
    return mutant_seq


def _generate_mutant_weighted_mixture_of_conditionals(
        master_sequence, allowed_mutations, n_locations_expected=None,
        conditional_transitions=None, log_mixture_weights=None, **kwargs
):
    """
    Generate a mutant sequence by sampling from pointwise mixture distributions

    :param master_sequence: str, giving the master sequence of amino acids
    :param allowed_mutations: list of triples or lists of (int, str, list),
        where these respectively are the integer position into the chain, the
        current amino acid value at that position, and the list of str, giving
        allowed values that residue can take.
    :param n_locations_expected: integer, giving the expected number of
        locations to mutate; the actual number of locations will be drawn from
        a Poisson distribution with this expectation. If None, this behavior is
        not executed and instead, all locations are (potentially) mutated.
    :param conditional_transitions: list of n dictionaries that return a
        conditional distribution over the replacement residue, given the current
        residue.
    :param log_mixture_weights: list of n numeric values, corresponding to the
        unnormalized log mixture weights describing which of the conditional
        transition distributions should be drawn from.
    :param kwargs: accept but do not use any additional kwargs
    :return: str, giving the mutant sequence of amino acids
    """

    if conditional_transitions is None and log_mixture_weights is None:
        conditional_transitions = [{
            aai: {aaj: 0.05 for aaj in _LIST_OF_AAS}
            # Above: How we want to specify a "conditional distribution?"
            for aai in _LIST_OF_AAS
        }]
        log_mixture_weights = [0.0]
    elif log_mixture_weights is None:
        if not isinstance(conditional_transitions, list):
            raise ValueError('The input conditional_transitions must be a list '
                             'or None.')
        log_mixture_weights = [0.0 for di in conditional_transitions]
    elif conditional_transitions is None and log_mixture_weights is not None:
        raise ValueError('If the log_mixture_weights is specified, the '
                         'conditional transitions must also be specified.')

    if not (isinstance(conditional_transitions, list)
            and isinstance(log_mixture_weights, list)):
        raise ValueError('Each of conditional_transitions and '
                         'log_mixture_weights must be either list or None.')

    # Produce the normalized mixture weights from the unnormalized log mixture
    # weights, where this is now known to be a list:
    # Set to zero mean to avoid numerical problems and get unnormalized version
    mixture_weights = np.exp(log_mixture_weights - np.mean(log_mixture_weights))
    # Normalize
    mixture_weights = mixture_weights / np.sum(mixture_weights)

    # Choose how many residues should be modified via a Poisson draw
    if n_locations_expected is not None:
        # Try to mutate at at least one location, but no more than the
        # len(allowed_mutations)

        # if n_locations_expected == 1:
        #     print('Warning: if n_locations_expected is set to 1, will draw '
        #           'exactly one mutation in all cases.')
        
        drawn = 1 + np.random.poisson(np.max([0, n_locations_expected - 1]))
        # Above: This preserves the expectation, but would have unexpected
        # effects if the user sets the expectation to 1
        n_locations_to_mutate = np.min([len(allowed_mutations), drawn])
    else:
        n_locations_to_mutate = len(allowed_mutations)

    # Given the number of residues to be modified, determine which residues
    # will be modified
    locations_to_mutate = \
        np.random.permutation(len(allowed_mutations))[:n_locations_to_mutate]

    # Given the residues to be modified, draw from the mixture distribution
    # over replacement residues
    # Presently, this is done per-residue
    mutations_to_apply = []
    for idx, (res_loc_i, cur_res_i, allowed_res_list_i) in enumerate(
            allowed_mutations):
        if idx not in locations_to_mutate or len(allowed_res_list_i) == 0:
            # That is, this residue is not allowed to be mutated in this draw
            continue
        tmp_list = deepcopy(allowed_res_list_i)
        # Old behavior allowed the current residue at the selected locations
        # if cur_res_i not in tmp_list:
        #         #     tmp_list = [cur_res_i] + tmp_list

        # New behavior is to draw according to the table given, which MAY
        # include a self-mutation
        # Choose a mixture component
        comp_idx = np.random.choice(len(mixture_weights), p=mixture_weights)

        # Choose a mutant residue under that mixture component
        res_mix_weights = [
            conditional_transitions[comp_idx][cur_res_i][res_j]
            for res_j in tmp_list
        ]
        if np.sum(res_mix_weights) == 0.0:
            continue  # skip if no support under chosen mixture component
        elif np.sum(res_mix_weights) < 0.0:
            raise ValueError('res_mix_weights is negative: '
                             '{}'.format(res_mix_weights))

        # Normalize
        res_mix_weights = res_mix_weights / np.sum(res_mix_weights)

        # Draw
        chosen_res_i = np.random.choice(tmp_list, p=res_mix_weights)

        # Append
        mutations_to_apply.append(
            ('', str(res_loc_i), cur_res_i, chosen_res_i))

    # Apply mutations to the master
    mutant_seq = \
        mutate_seq(master_seq=master_sequence, mutations=mutations_to_apply)

    return mutant_seq


def expand_menu_study_params_master_antigen_structures_mutations(menu, n_mutants_per_master=1, **kwargs):
    """
    Expand a menu into the set of allowed combinations of its elements

    I/O modeled on improvwf.utils.expand_menu_study_params

    :param menu: dict, giving allowed studies in its 'studies' entry; these are
        listed, each with 'study_type' and 'study_parameters' entries.
    :param n_mutants_per_master: int, the number of mutants to enumerate per
        master antigen
    :param kwargs: any remaining keyword args, including:
        generator_type: str, giving the type of generator, e.g.,
            'poisson_uniform' and 'weighted_mixture_of_conditionals';
        n_locations_expected: int or None, giving the expected
            number of locations at which a given mutant will differ from the
            master. The actual number will be drawn from a poisson distribution
            where this value gives the expectation of the draw.  If None,
            mutate ALL allowed locations (previous default).
        override_mutant_sequence: str or None; if str, must be a str of the same
            length as the master sequence and must be allowed by the menu.
    :return: expanded_menu: a dictionary of expanded studies, where the
        "studies" field contains one entry per allowed combination of parameters
    """

    menu_out = {k: menu[k] for k in menu if k != 'studies'}  # Pass anything else in the menu dictionary
    menu_out['studies'] = []
    for si in menu['studies']:
        if 'study_parameters' in si.keys():

            # Check that the si['study_parameters'] dictionary conforms to
            # the desired pattern
            conforms = recursive_parameters_dict_check(
                si['study_parameters'],
                key_structure=INPUT_MENU_STUDY_PARAMETERS_KEY_STRUCTURE
            )
            if not conforms:
                raise ValueError('The study_parameters value must conform!')

            # Now that conformity is checked, enumerate the combinations of
            # parameters into a flattened menu
            n_mutants_enumerated = 0
            template_this_type = {k: si[k] for k in si if k !=
                                        "study_parameters"}
            template_this_type['study_parameters'] = {}

            # Generate the set of mutants
            if "generator_type" in kwargs.keys() \
                    and kwargs["generator_type"] == 'linear_mutant_generator':
                if "override_mutant_sequence" in kwargs.keys() \
                        and kwargs["override_mutant_sequence"] is not None:
                    
                    # change the original AA in dataframe to be the override mutant
                    kwargs['singlePointMutationDataWithSampleWeights'] = \
                        override_single_point_df(
                            si['study_parameters'][MasterAntigenSequenceKey],
                            si['study_parameters'][AllowedMutationsKey]['AllowedMutations'],
                            **kwargs
                        )

                    # if the kwarg override_mutant_sequence is present, use it to override 
                    master_sequence, allowed_mutations = _override_master_mutant(
                        si['study_parameters'][MasterAntigenSequenceKey],
                        si['study_parameters'][AllowedMutationsKey]['AllowedMutations'],
                        **kwargs
                    )
                    mutant_sequences = generate_linear_mutants(
                        master_sequence,
                        allowed_mutations,
                        numberMutantToGenerate=n_mutants_per_master,
                        **kwargs
                    )
                else:
                    mutant_sequences = generate_linear_mutants(
                        si['study_parameters'][MasterAntigenSequenceKey],
                        si['study_parameters'][AllowedMutationsKey]['AllowedMutations'],
                        numberMutantToGenerate=n_mutants_per_master,
                        **kwargs
                    )

            else:
                mutant_sequences = []
                fail_counter = 0
                while n_mutants_enumerated < n_mutants_per_master and fail_counter < 10:
                    mut = _generate_mutant(
                        si['study_parameters'][MasterAntigenSequenceKey],
                        si['study_parameters'][AllowedMutationsKey]['AllowedMutations'],
                        **kwargs
                    )
                    # TODO: Terminate early/check for redundancy
                    if mut not in mutant_sequences:
                        mutant_sequences.append(mut)
                        n_mutants_enumerated += 1
                    else:
                        fail_counter += 1

            # For each mutant, add it to the growing dictionary of valid parameter settings for that study
            for i, msi in enumerate(mutant_sequences):
                study_dict_to_append = deepcopy(template_this_type)
                tmp_dict = {
                    'MasterAntigenID' : si['study_parameters']['MasterAntigenID'],
                    MasterAntigenSequenceKey : si['study_parameters'][MasterAntigenSequenceKey],
                    'MasterAntigenFASTAHash': si['study_parameters']['MasterAntigenFASTAHash'],
                    'MasterAntigenFASTAPath': si['study_parameters']['MasterAntigenFASTAPath'],
                    'StructureID': si['study_parameters']['Structure']['StructureID'],
                    'StructurePath': si['study_parameters']['Structure']['StructurePath'],
                    'StructureHash': si['study_parameters']['Structure']['StructureHash'],
                    'AntigenChainsInStructure': si['study_parameters']['Structure']['AntigenChainsInStructure'],
                    'AntigenSequence': msi
                    }
                for ki, vi in tmp_dict.items():
                    study_dict_to_append['study_parameters'][ki] = vi
                menu_out['studies'].append(study_dict_to_append)
        else:
            raise ValueError("All studies for AbAg interactions are presumed to be parameterized!")

    return menu_out


def _build_template_dictionary(h_ent):
    """
    Create a pseudo-result dictionary to be filled with sub-results

    :param h_ent: dictionary, corresponding to a single history entry in an
        Improv history file; this is the history entry associated with the
        study that was actually run, which may contain experiments on several
        or many mutant sequences.
    :return: template_dictionary; dict containing the common information from
        the main study to be inherited by the pseudo-studies corresponding to
        the individual experiments. Has the following structure:
            'study_type': str,
            'status': str,
            'study_parameters': {'MasterAntigenFASTAPath': str,
                'MasterAntigenFASTAHash': (tuple, list),
                'MasterAntigenID': str,
                MasterAntigenSequenceKey: str,
                'StructureID': str,
                'StructurePath': str,
                'StructureHash': (tuple, list)
                }
        'request_id' and 'result' fields will be added later, as will
        ['study_parameters']['AntigenSequence']
    """


    template_dictionary = {
        'study_type': h_ent['study_type'],
        'status': h_ent['status'],
        'study_parameters': {
            'AntigenChainsInStructure':
                h_ent['study_parameters']['ANTIGEN_CHAINS_IN_STRUCT']['values'][0],
        }
    }

    # ###########################################
    # ### ADD THE MASTER SEQUENCE INFORMATION ###
    # ###########################################

    # Compute the MAF hash and check against the listed value:
    mafhash = hash_fasta_or_yaml(
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_PATH']['values'][0],
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_HASH']['values'][0][1]
    )
    mafhashmatch = mafhash[0] == h_ent['study_parameters']\
        ['MASTER_ANTIGEN_FASTA_HASH']['values'][0][0]
    assert mafhashmatch, 'MASTER_ANTIGEN_FASTA_HASH {} does not match ' \
                         'the computed value {} for file {}!'.format(
        mafhash,
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_HASH']['values'][0],
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_PATH']['values'][0]
    )
    # Put the path into the template:
    template_dictionary['study_parameters']['MasterAntigenFASTAPath'] = \
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_PATH']['values'][0]
    template_dictionary['study_parameters']['MasterAntigenFASTAHash'] = \
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_HASH']['values'][0]
    mafseqrecords = list_of_seqrecords_from_fasta(
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_PATH']['values'][0]
    )
    assert len(mafseqrecords) == 1, 'MASTER_ANTIGEN_FASTA_PATH {} points ' \
                                    'to a file with more than one fasta ' \
                                    'record!'.format(
        h_ent['study_parameters']['MASTER_ANTIGEN_FASTA_PATH']['values'][0]
    )
    desc = parse_seqrecord_description(mafseqrecords[0])
    template_dictionary['study_parameters']['MasterAntigenID'] = desc['id']
    template_dictionary['study_parameters'][MasterAntigenSequenceKey] = \
        recast_sequence_as_str(mafseqrecords[0])

    # ###########################################
    # ### ADD THE STRUCTURE INFORMATION ###
    # ###########################################

    # Read the listed hash value and type
    hash_val, hash_type_str = \
    h_ent['study_parameters']['STRUCTURE_HASH']['values'][0]
    # Compute the STRUCTURE hash
    structhash = hash_pdb(
        h_ent['study_parameters']['STRUCTURE_PATH']['values'][0],
        algorithm=hash_type_str.split('_')[0],
        atom_lines_only=(hash_type_str.split('_')[-1] == 'ATOM')
    )
    # Check against the existing value
    structhashmatch = structhash[0] == \
                      h_ent['study_parameters']['STRUCTURE_HASH']['values'][0][0]
    assert structhashmatch, 'STRUCTURE_HASH {} does not match ' \
                            'the computed value {} for file {}!'.format(
        structhash[0],
        h_ent['study_parameters']['STRUCTURE_HASH']['values'][0],
        h_ent['study_parameters']['STRUCTURE_PATH']['values'][0]
    )

    # Fill in the dictionary appropriately
    template_dictionary['study_parameters']['StructureID'] = \
        os.path.split(h_ent['study_parameters']['STRUCTURE_PATH']['values'][0])[-1]
    # TODO: evaluate choice of pdb file name for StructureID
    template_dictionary['study_parameters']['StructurePath'] = \
        h_ent['study_parameters']['STRUCTURE_PATH']['values'][0]
    template_dictionary['study_parameters']['StructureHash'] = \
        h_ent['study_parameters']['STRUCTURE_HASH']['values'][0]

    return template_dictionary


def _get_results_from_history(result_value):
    """
    From history file's 'result', extract dicts, keyed by antigen ID and seq

    :param result_value: dictionary from a completed study's 'result' field.
        Either has the two keys 'ResultsPath' and 'ResultsHash' (directing us to
        load a downstream file) OR is a dictionary of lists.

        If a dictionary of lists, all lists are the same length and together
        constitute a table; each position within every list contains the list's
        named value for the single antigen associated with that individual
        position. Two lists are special: AntigenID and AntigenSequence, where
        these will become the keys into a dictionary of results to be passed up.

        If we are directed to another file, it must either be a yaml upon which
        we will call this function again, or a .csv, which corresponds to a
        dictionary of lists, as here.

    :return: result_by_id (dict, keyed by antigen ID) and result_by_sequence
        (dict, keyed by antigen sequence)
    """

    result_by_id = {}
    result_by_sequence = {}

    if set(result_value.keys()) == {'ResultsPath', 'ResultsHash'}:
        # We're being pointed at a downstream file.
        hash_of_target_file = hash_fasta_or_yaml(
            result_value['ResultsPath'], result_value['ResultsHash'][1]
        )
        hash_matches = hash_of_target_file[0] == result_value['ResultsHash'][0]
        if hash_matches:
            if os.path.splitext(result_value['ResultsPath'])[-1].lower() == '.csv':
                r = csv.reader(result_value['ResultsPath'])
                for i, li in enumerate(r):
                    if i == 0 and not ('AntigenID' in li or 'AntigenSequence' in li):
                        raise ValueError(
                            'One or the other of AntigenID or AntigenSequence '
                            'must appear as a column name in li'
                        )
                    results_tmp = {
                        ki: vi for ki, vi in li.items()
                        if ki not in {'AntigenID', 'AntigenSequence'}
                    }
                    try:
                        result_by_id[li['AntigenID']] = results_tmp
                    except KeyError:
                        pass

                    try:
                        result_by_sequence[li['AntigenSequence']] = results_tmp
                    except KeyError:
                        pass

            elif os.path.splitext(result_value['ResultsPath'])[-1].lower() == '.yaml':
                # with open(result_value['ResultsPath'], 'r') as f:
                #     d = yaml.safe_load(f)
                d = yaml_safe_load_with_lock(result_value['ResultsPath'])
                result_by_id, result_by_sequence = _get_results_from_history(d)
            else:
                raise ValueError('The dictionary pointed at a results file that is neither a .csv nor a .yaml')
        else:
            raise ValueError('The dictionary pointed at a results file that does not match the hash')
    elif set(result_value.keys()).intersection({'AntigenID', 'AntigenSequence'}):
        # We're looking at a set of results, formatted as a dictionary of lists
        # All of the lists must be the same length, and at least one of the two
        #  keys 'AntigenID' or 'AntigenSequence' must appear in the dictionary.

        # Check for list length consistency:
        len_of_lists = None
        for ki, li in result_value.items():
            len_this_list = len(li)
            if not isinstance(li, list) or len_this_list is None:
                raise ValueError('Each value must be a list')
            elif len_of_lists is not None and len_of_lists != len_this_list:
                raise ValueError('Each list must have the same length')
            elif len_of_lists is None:
                len_of_lists = len_this_list

        # Pull the information out of the individual lists
        try:
            # Construct a dictionary by "unzipping" this one
            result_by_id = {agidi: {
                kj: lj[i] for kj, lj in result_value.items()
                if kj not in {'AntigenID', 'AntigenSequence'}
            }
            for i, agidi in enumerate(result_value['AntigenID'])
            }

        except KeyError:
            result_by_id = {}

        try:
            # Construct a dictionary by "unzipping" this one
            result_by_sequence = {agseqi: {
                kj: lj[i] for kj, lj in result_value.items()
                if kj not in {'AntigenID', 'AntigenSequence'}
            }
            for i, agseqi in enumerate(result_value['AntigenSequence'])
            }
        except KeyError:
            result_by_sequence = {}
    else:
        raise ValueError(
            'The result entry is improperly formatted, neither pointing to a '
            'result file nor containing a dictionary of lists, where either '
            'AntigenID or AntigenSequence is present.')

    return result_by_id, result_by_sequence


def get_history_db(db_url, nrandom=None, study_type=None, seq=None, seq_dist=None):
    """
    Get a history for the decision-maker to work off of.

    :param db_url: str, url for the db.
    :param nrandom: int, number of studies desired. If neither this nor <seq_dist> are provided,
		    all studies will be returned. Up to #nrandom studies will be chosen
                    from the "allowed set", which is all studies unless <seq_dist> is also provided.
    :param seq: str, MUST be provided if using <seq_dist>, the sequence <seq_dist> will calculate
		distance from.
    :param seq_distance: int, string distance (inclusive) allowed between <seq> and the AntigenSequence
			 of returned studies.

    :returns: A history assembled from the selected sequences.
    :raises ValueError: if <seq_dist> is provided, but not <seq>.
    """
    print('In get_history_db, acquiring history database contents...')
    if seq is not None:
        history = get_history_db_improv(db_url, 
                                        nrandom=nrandom,
                                        study_type=study_type,
                                        distance_datum="AntigenSequence", 
                                        distance_val=seq, 
                                        max_distance=seq_dist)
    else:
        history = get_history_db_improv(db_url, nrandom=nrandom, study_type=study_type)
    print('... done')

    print('Acquiring any daughter contents...')
    history_out = retrieve_nested_history_contents(history)
    print('... done')

    return history_out


def get_history(history_path, stale_history=None):
    print('In get_history, acquiring main history yaml contents...')
    try:
        history = yaml_safe_load_with_lock(history_path, acquiretime=300)
        print('Acquiring any daughter contents...')
        history_out = retrieve_nested_history_contents(history)
        print('... done')

        if stale_history is not None:
            yaml_safe_dump_with_lock(history_out, stale_history, acquiretime=1)

    except Exception as e:

        if stale_history is not None:
            print('Encountered exception in history loading:')
            print(e)
            print('Unable to retrieve primary history {}: retrieving stale '
                  'copy {} instead.'.format(history_path, stale_history))
            history_out = yaml_safe_load_with_lock(stale_history, acquiretime=1)
        else:
            raise e

    return history_out


def retrieve_nested_history_contents(history):
    """
    Retrieve sequence-by-sequence results from Improv standard history file

    To minimize the amount of information Improv is carrying around and to
    neatly specify studies, the history may contain references to fasta files
    containing multiple (or many) sequences, as well as results files
    containing the results of those many experiments. This function
    retrieves the nested contents of these studies and creates virtual
    "studies" corresponding to individual experiments, many of which may have
    actually been conducted in a given Improv-administered Maestro study.

    The needed information appears two places:
    'study_parameters': this value in the dictionary FULLY describes the
        inputs to the study, i.e., ALL of the sequences, structures, and chain
        information of interest.  In particular, along with the 'study_type'
        field, the features describing the observation "locations" are
        computable from this entry without reference to the 'result' entry
        below, since this is the only way to represent in-progress studies for
        decision-making.
    'result': The result should either include a dictionary that gives results,
        sequence-by-sequence, OR two keys: ResultsPath and ResultsHash. If the
        dictionary contains results, these results are matched with the
        study_parameters sequences identified previously.

    :param history: an Improv-standard history file's dictionary contents. At
        the highest level, it has 'description' and 'history keys. Key-value
        pairs in history['history] are keyed by request_id and have dictionary
        values. The dictionary has the following keys: 'request_id',
        'study_type', 'status', 'study_parameters', and 'result'.
        'study_parameters' must ALWAYS contain these seven keys:
            ANTIGEN_FASTA_PATH, ANTIGEN_FASTA_HASH, MASTER_ANTIGEN_FASTA_PATH,
            MASTER_ANTIGEN_FASTA_HASH, STRUCTURE_PATH, STRUCTURE_HASH,
            ANTIGEN_CHAINS_IN_STRUCT.
        where the _PATH suffixed keys yield dictionaries with 'values' giving
        a one-element list containing the str path to the target file;
        _HASH-suffixed keys yield dictionaries with 'values' giving a one-
        element list containing a list/tuple giving the hash and the hash type,
        as generated, e.g., by the vaccine_advance_core hash_pdb function.
        'result': either includes a dictionary that gives results for the
        individual mutants in the file referenced by ANTIGEN_FASTA_PATH or
        contains exactly two keys, ResultsPath and ResultsHash.
    :return: flattened_history dictionary, where the individual records in this
        history dictionary are pseudo-studies
    """

    history_out = {ki: vi for ki, vi in history.items() if ki != 'history'}
    history_out['history'] = {}
    for ksi, si in history['history'].items():
        template_dictionary = _build_template_dictionary(si)

        # ###############################################
        # ### FILL IN THE VARIANT ANTIGEN INFORMATION ###
        # ###############################################

        # Compute the AF hash and check against the listed value:
        afhash = hash_fasta_or_yaml(
            si['study_parameters']['ANTIGEN_FASTA_PATH']['values'][0],
            si['study_parameters']['ANTIGEN_FASTA_HASH']['values'][0][1]
        )
        afhashmatch = afhash[0] == si['study_parameters']\
            ['ANTIGEN_FASTA_HASH']['values'][0][0]
        assert afhashmatch, 'ANTIGEN_FASTA_HASH {} does not match ' \
                             'the computed value {} for file {}!'.format(
            afhash,
            si['study_parameters']['ANTIGEN_FASTA_HASH']['values'][0],
            si['study_parameters']['ANTIGEN_FASTA_PATH']['values'][0]
        )
        # Retrieve the variant records:
        antigen_seqrecords = list_of_seqrecords_from_fasta(
            si['study_parameters']['ANTIGEN_FASTA_PATH']['values'][0]
        )

        if 'result' in si:
            results_by_id, results_by_sequence = \
                _get_results_from_history(si['result'])

            #check if either structure is empty
            if(not results_by_id or not results_by_sequence):
                print("WARNING: Empty results in FINISHED study " + ksi)
                continue
        
        for j, asrj in enumerate(antigen_seqrecords):
            dict_under_assembly = deepcopy(template_dictionary)

            dict_under_assembly['study_parameters']['AntigenSequence'] = \
                recast_sequence_as_str(asrj)
            antigen_id = parse_seqrecord_description(asrj)['id']

            # 'result'
            if 'result' in si:
                try:
                    dict_under_assembly['result'] = results_by_id[antigen_id]
                except KeyError:
                    try:
                        dict_under_assembly['result'] = results_by_sequence[
                            dict_under_assembly['study_parameters']['AntigenSequence']
                        ]
                    except KeyError:
                        print("WARNING: Mis-matched AntigenSequence and fasta! " + ksi)
                        continue

            assert recursive_parameters_dict_check(
                dict_under_assembly['study_parameters'],
                INTERMEDIATE_MENU_STUDY_PARAMETERS_KEY_STRUCTURE), \
                'Flattened study_parameters do not comply with specification!'

            dict_under_assembly['request_id'] = '{}_{}'.format(si['request_id'], j)
            history_out['history'][dict_under_assembly['request_id']] = dict_under_assembly

    return history_out


def _get_newpath(files_directory, filename):
    # Create a candidate name
    newpath = os.path.abspath(os.path.join(files_directory, filename))

    # Check if there is already a file of the same name present
    if os.path.exists(newpath):
        split_newpath = list(os.path.split(newpath))
        match_pattern = os.path.splitext(split_newpath[-1])[0] \
                        + '_([0-9]+)\\' \
                        + os.path.splitext(split_newpath[-1])[-1]
        files = os.listdir(split_newpath[0])
        matches = [re.match(match_pattern, fi) for fi in files]
        matchints = [int(mi.groups()[0]) for mi in matches if mi is not None]
        maxint = 0
        if matchints:
            maxint = sorted(matchints)[-1]
        newint = maxint + 1
        newpath = split_newpath[:-1] + [os.path.splitext(split_newpath[-1])[0] + '_{}'.format(newint) + os.path.splitext(split_newpath[-1])[-1]]
        newpath = os.path.join(*newpath)

    if os.path.exists(newpath):
        raise RuntimeError(
            "Unable to create a unique file name for {}!".format(filename)
        )

    return newpath


def _update_hash_dict(hash_dict, hashtuple, files_directory, original_file_path):
    if hashtuple not in hash_dict:
        # Copy over the file to files_dir
        newpath = _get_newpath(
            files_directory,
            os.path.split(original_file_path)[-1]
        )

        # Copy over with a unique name
        shutil.copy2(original_file_path,newpath)
        # Add it to the dictionary
        hash_dict[hashtuple] = newpath

    return hash_dict


def _update_history_result_filepaths(result_value, hash_dict, files_directory):
    """
    Traverse a 'result' from history file 'result', moving and changing files

    Modification of the 'result' field also may require modifying other,
    secondary .yaml files or even .csv files containing results information, and
    that this file i/o is also handled by _update_history_result_filepaths.

    :param result_value: dictionary from a completed study's 'result' field.
        Either has the two keys 'ResultsPath' and 'ResultsHash' (directing us to
        load a downstream file) OR is a dictionary of lists.

        If a dictionary of lists, all lists are the same length and together
        constitute a table; each position within every list contains the list's
        named value for the single antigen associated with that individual
        position.

        If we are directed to another file, it must either be a yaml upon which
        we will call this function again, or a .csv, which corresponds to a
        dictionary of lists, as here.
    :param hash_dict: dict, keyed by file hashes, that lists all of the
        files currently in files_directory. This is augmented in the course of
        execution.
    :param files_directory: str, giving the path to the directory in which the
        moved files will reside.
    :return: result_out, hash_dict: result_out is a dictionary of the same form
        as result, with all files moved and file paths updated appropriately;
        hash_dict is incoming hash_dict, updated with the appropriate values
        for this result's files.
    """

    if set(result_value.keys()) == {'ResultsPath', 'ResultsHash'}:
        # We're being pointed at a downstream file.
        hash_of_target_file = hash_fasta_or_yaml(
            result_value['ResultsPath'], result_value['ResultsHash'][1]
        )
        hash_matches = hash_of_target_file[0] == result_value['ResultsHash'][0]
        if not hash_matches:
            raise ValueError('The dictionary pointed at a results file that '
                             'does not match the hash')

        if os.path.splitext(result_value['ResultsPath'])[-1].lower() == '.csv':
            # TODO: Implement csv handling
            raise NotImplementedError('Haven''t implemented csv handling!')
        elif os.path.splitext(result_value['ResultsPath'])[-1].lower() == '.yaml':
            # Call this function again on this guy
            # with open(result_value['ResultsPath'], 'r') as f:
            #     d = yaml.safe_load(f)
            d = yaml_safe_load_with_lock(result_value['ResultsPath'])
            d = _update_history_result_filepaths(d, hash_dict, files_directory)
            # Write this yaml back out
            newpath = _get_newpath(
                files_directory,
                os.path.split(result_value['ResultsPath'])[-1]
            )
            # with open(newpath, 'w') as f:
            #     yaml.safe_dump(d, f)
            yaml_safe_dump_with_lock(d, newpath)

            # Update result value with the new path and hash
            result_value['ResultsPath'] = newpath
            # Note: The new hash may not match the old; e.g., if paths (and
            # thus the content) in the dictionary have changed.
            result_value['ResultsHash'] = hash_fasta_or_yaml(
                result_value['ResultsPath'], result_value['ResultsHash'][1]
            )
        else:
            raise ValueError('The dictionary pointed at a results file '
                             'that is neither a .csv nor a .yaml')

    elif set(result_value.keys()).intersection({'AntigenID', 'AntigenSequence'}):
        # We're looking at a set of results, formatted as a dictionary of lists
        # All of the lists must be the same length, and at least one of the two
        #  keys 'AntigenID' or 'AntigenSequence' must appear in the dictionary.

        # Check for list length consistency:
        len_of_lists = None
        for ki, li in result_value.items():
            len_this_list = len(li)
            if not isinstance(li, list) or len_this_list is None:
                raise ValueError('Each value must be a list')
            elif len_of_lists is not None and len_of_lists != len_this_list:
                raise ValueError('Each list must have the same length')
            elif len_of_lists is None:
                len_of_lists = len_this_list

        # Now that the dictionary checks correctly, again, alter the paths.
        # Assume that anything here that is a path with be named in CamelCase
        # or camelCase, with the last part being "Path", and that if there is a
        # hash value stored, it will replace the "Path" suffix with "Hash",
        # e.g., StructurePath and StructureHash

        result_value_tmp = {}
        for ki, li in result_value.items():
            if not ki.endswith('Path'):
                # The key is not a path; we don't need to do any processing on
                # it, but we do need to determine whether to copy it now.
                if ki.endswith('Hash') and ki[:-4]+'Path' in result_value.keys():
                    # We'll update it with the path-suffixed key and we don't want any overwriting
                    continue
                # This key doesn't end in 'Path' and either doesn't end in
                # 'Hash' or the corresponding 'Path' key does not appear: copy
                # it over.
                result_value_tmp[ki] = li
                continue

            # Else, we're dealing with a Path-suffixed key:

            # Obtain the hash value if present
            hashkey_to_look_for = ki[:-4] + 'Hash'
            matching_hash_vals = [None for fp_ij in li]
            if hashkey_to_look_for in result_value:
                # Contains a list of 2-element tuples/lists, where the first
                # element is the hash (string) and the second is the type.
                matching_hash_vals = result_value[hashkey_to_look_for]

            # Walk along the list, updating
            updated_paths = []
            updated_hashes = []
            for fp_ij, hv_ij in zip(li, matching_hash_vals):
                # Choose the type of hash
                if hv_ij is None:
                    hash_type = 'MD5'
                else:
                    hash_type = hv_ij[1].split("_")[0]

                # Compute the hash
                if fp_ij.endswith('.fasta'):
                    hashtuple = tuple(hash_fasta_or_yaml(fp_ij, algorithm=hash_type))
                elif fp_ij.endswith('.pdb'):
                    hashtuple = tuple(hash_pdb(fp_ij, algorithm=hash_type))
                else:
                    raise RuntimeError('Encountered a key with Path suffix, '
                                       'pointing to something other than a .pdb '
                                       'or .fasta')

                # Check that the hash matches
                if hv_ij is not None and (hv_ij[0] != hashtuple[0]):
                    print(
                        'Hash keys for file {} do not match: computed {} vs. '
                        'nominal {}'.format(
                            fp_ij,
                            hashtuple[0],
                            hv_ij[0]
                        )
                    )
                    raise ValueError(
                        'Mismatched file hashes in parsing study results!')

                # Now that we're definitely talking about the right file, copy
                # it over and update the hash_dict.
                hash_dict = _update_hash_dict(hash_dict, hashtuple,
                                              files_directory, fp_ij)

                # Store, prior to copying to result_value_tmp
                updated_paths.append(hash_dict[hashtuple])
                updated_hashes.append(hashtuple)

            # Now that the individual list members have been checked and
            # modified, put into result_value_tmp
            result_value_tmp[ki] = updated_paths
            if hashkey_to_look_for in result_value:
                result_value_tmp[hashkey_to_look_for] = updated_hashes

        # Finally, we've copied over all of the non-Path, non-Hash keys and
        # determined new values for the Path- and any Hash-suffixed keys.
        result_value = result_value_tmp

    # Return the final value
    return result_value


def modify_nested_filepaths(history, files_directory, hash_dict, hash_type=None, write_csv_path=None):
    """
    Modify the history dict and child .yaml by consolidating fasta and pdbs

    This is the workhorse function of update_history_paths. Here, for each
    history entry (a single study's record) we modify the 'study_parameters'
    field and pass down to _update_history_result_filepaths to update the
    'result' field. When these actions have been completed for all entries in
    the history dictionary, we pass it back up. Note that the modification of
    the 'result' field also may require modifying other, secondary .yaml files
    or even .csv files containing results information, and that this file i/o is
    also handled by _update_history_result_filepaths.

    :param history: an Improv-standard history file's dictionary contents. At
        the highest level, it has 'description' and 'history keys. Key-value
        pairs in history['history] are keyed by request_id and have dictionary
        values. The dictionary has the following keys: 'request_id',
        'study_type', 'status', 'study_parameters', and 'result'.
        'study_parameters' must ALWAYS contain these seven keys:
            ANTIGEN_FASTA_PATH, ANTIGEN_FASTA_HASH, MASTER_ANTIGEN_FASTA_PATH,
            MASTER_ANTIGEN_FASTA_HASH, STRUCTURE_PATH, STRUCTURE_HASH,
            ANTIGEN_CHAINS_IN_STRUCT.
        where the _PATH suffixed keys yield dictionaries with 'values' giving
        a one-element list containing the str path to the target file;
        _HASH-suffixed keys yield dictionaries with 'values' giving a one-
        element list containing a list/tuple giving the hash and the hash type,
        as generated, e.g., by the vaccine_advance_core hash_pdb function.
        'result': either includes a dictionary that gives results for the
        individual mutants in the file referenced by ANTIGEN_FASTA_PATH or
        contains exactly two keys, ResultsPath and ResultsHash.
    :param files_directory: str, path to a directory into which all child .yaml
        files and all .pdb and .fastas will be placed.
    :param hash_dict: dictionary, keyed by file hashes, that lists all of the
        files in files_directory. This is augmented in the course of execution.
    :param write_csv_path: str, path to CSV file which will record the original
        location of files allowing request_id to be traced back to original workspace
    :return: history_out: the improv-standard history contents, to be written to
        a new file, but now with altered file paths for all of its children.
    """

    old_paths = []

    if hash_type is None:
        hash_type = 'MD5'

    history_out = {ki: vi for ki, vi in history.items() if ki != 'history'}
    history_out['history'] = {}
    for ksi, si in history['history'].items():
        # This item is a single study

        # First, handle the parameters:
        for fkj in si['study_parameters'].keys():
            # Make sure we're dealing with a path and acquire the corresponding
            # hash key
            if not fkj.endswith('_PATH'):
                continue
            hash_key = fkj[:(-1 * len('_PATH'))] + '_HASH'

            # Get the hash
            print(si['study_parameters'][fkj]['values'])
            if si['study_parameters'][fkj]['values'][0].endswith('.fasta'):
                hashtuple = tuple(hash_fasta_or_yaml(
                    si['study_parameters'][fkj]['values'][0],
                    si['study_parameters'][hash_key]['values'][0][1]
                ))
            elif si['study_parameters'][fkj]['values'][0].endswith('.pdb'):
                hashtuple = tuple(hash_pdb(
                    si['study_parameters'][fkj]['values'][0],
                    si['study_parameters'][hash_key]['values'][0][1].split("_")[0]
                ))
            else:
                raise RuntimeError('Encountered a key with _PATH suffix, '
                                   'pointing to something other than a .pdb '
                                   'or .fasta')

            # Make sure the two hashes match
            if hashtuple[0] != si['study_parameters'][hash_key]['values'][0][0]:
                print(
                    'Hash keys for file {} do not match: computed {} vs. '
                    'nominal {}'.format(
                        fkj['values'][0],
                        hashtuple[0],
                        si['study_parameters'][hash_key]['values'][0][0]
                    )
                )
                raise ValueError('Mismatched study_parameter file hashes!')

            # If a matching file does not exist, copy over and add to hash_dict
            hash_dict = _update_hash_dict(
                hash_dict, hashtuple, files_directory,
                si['study_parameters'][fkj]['values'][0]
            )

            # Write previous location to CSV file
            if fkj == 'ANTIGEN_FASTA_PATH':
                old_paths.append(str(ksi + "," + si['study_parameters'][fkj]['values'][0]))

            # Either the appropriate file was already present or we've just
            # copied it to files_directory in _update_hash_dict; update the
            # value in si with the new path
            si['study_parameters'][fkj]['values'][0] = hash_dict[hashtuple]

        # Now, handle the result, the only other field that should have paths
        # If the study is complete, 'result' is present.
        if 'result' in si:
            # Either we're pointing to a 2-entry dictionary specifying a file
            #  or a sub-dictionary
            si['result'] = _update_history_result_filepaths(
                si['result'], hash_dict, files_directory
            )

        history_out['history'][ksi] = si
    
    if write_csv_path is not None:
        with open(write_csv_path, 'w') as f:
            for line in old_paths:
                f.write(line + "\n")
    
    return history_out


def consolidate_studies_to_write(studies_to_write, study_types=None, must_match=None):
    """
    Consolidate studies of given type(s) into combined study(-ies).

    Used upstream of write_selected_study_abag.

    :param studies_to_write: list of studies to be written. Each is a dictionary
        containing two fields: 'parameters' and 'request_id'. Within
        'parameters' are two and only two fields: 'study_type' and
        'study_parameters'. The core idea of this function is to combine
        studies of like type and compatible parameters into a single study.
    :param study_types: list of str; each str gives a single type of study that
        might be combined. Entries si in studies_to_write where the
        si['parameters']['study_type'] value matches may be combined, if their
        si['parameters']['study_parameters'] values are compatible, as defined
        by must_match.
    :param must_match: study_parameters values that must match for fusion. At
        present, these must be an exact match.
    :return: studies_to_write, where some studies may have been combined by
        fusing their mutant sequences.
    """

    if study_types is None:
        study_types = []
    if (not study_types) or (must_match is None):
        return studies_to_write

    # Carry out the meat of the function
    print('studies_to_write consolidation not yet implemented; passing all studies separately.')
    for type_i in study_types:
        # Perform consolidation
        pass
    return studies_to_write


def write_selected_study_abag(selected_params, studies, request_id, output_path, fasta_path):
    """
    Merge selected parameters with the template and write to a .yaml

    :param selected_params: the dictionary describing the selected study.
    :param studies: dictionary containing the template studies, loaded from
        their individual .yaml files.
    :param request_id: string giving the name of the request
    :param output_path: directory into which the final study will be written.
    :param fasta_path: path pointing to the new fasta file to be written
    """

    # TODO: Compress multiple AntigenSequences into a single FASTA

    # Prepare the .fasta file for the mutants under consideration and write
    # them out
    print(selected_params)
    selected_params = deepcopy(selected_params) # Prevent crash if there are duplicate seqences
    idstr = derive_idstr_from_seq(selected_params['study_parameters']['AntigenSequence'])
    master_idstr = selected_params['study_parameters']['MasterAntigenID']

    # Obtain a string describing the mutations between the two sequences
    print(selected_params['study_parameters']['AntigenSequence'])
    print(selected_params['study_parameters']['MasterAntigenSequence'])
    seq_differences = selected_mutations_mut_from_list_of_tuples(diff_seqs(
        mutant_seq=recast_sequence_as_str(selected_params['study_parameters']['AntigenSequence']),
        master_seq=recast_sequence_as_str(selected_params['study_parameters']['MasterAntigenSequence'])
    ))

    description_str = write_description_for_seqrecord(
        idstr, master_idstr, seq_differences
    )

    # Recast the sequences into BioPython SeqRecords
    seqrecord = recast_as_seqrecord(selected_params['study_parameters']['AntigenSequence'],
                                    id_if_str_or_seq=idstr,
                                    description_if_str_or_seq=description_str
                                    )
    # Write the SeqRecords to file
    fasta_from_list_of_seqrecords(
        records=[seqrecord], fasta_path=fasta_path, format='fasta-2line'
    )

    # Compute the resulting FASTA file's hash
    fasta_hash = hash_fasta_or_yaml(fasta_path, 'MD5')

    selected_params['study_parameters']['AntigenFASTAPath'] = fasta_path
    selected_params['study_parameters']['AntigenFASTAHash'] = fasta_hash
    selected_params['study_parameters'].pop('AntigenSequence')

    assert recursive_parameters_dict_check(
        selected_params['study_parameters'],
        OUTPUT_MENU_STUDY_PARAMETERS_KEY_STRUCTURE
    ), 'study_parameters Structure to be written out does not match the ' \
       'specified format!'

    # Convert parameters to the yaml format (i.e., with values and label
    tmp = {}
    for ki, vi in selected_params['study_parameters'].items():
        # Do name conversion to the final output format
        try:
            kfinali = STUDY_WRITE_OUT_KEYS[ki]
        except KeyError:
            kfinali = None
        if kfinali is None:
            continue

        # Create dictionary in new names
        tmp[kfinali] = {'values': [vi],
                   'label': ['{}.{}'.format(kfinali, 1)]}  # TODO: enumeration?

    # Overwrite dictionary with the final output format
    selected_params['study_parameters'] = tmp

    # Use Improv's write out function
    write_selected_study(selected_params=selected_params, studies=studies,
                         request_id=request_id, output_path=output_path)

def study_query_data_for_expanded_menu(expanded_menu):

    query_data_for_studies = []
    for study in expanded_menu['studies']:

        antigen_fasta_hash = \
            fasta_string_from_sequence( mutant_seq=study['study_parameters']['AntigenSequence'],
                                        master_seq=study['study_parameters']['MasterAntigenSequence'],
                                        master_idstr=study['study_parameters']['MasterAntigenID'],
                                        hash_algorithm='MD5')

        study_query = {
            "ANTIGEN_FASTA_HASH": {"value" : antigen_fasta_hash},
            "MASTER_ANTIGEN_FASTA_HASH": {"value" : study['study_parameters']['MasterAntigenFASTAHash'][0]},
            "STRUCTURE_HASH": {"value" : study['study_parameters']['StructureHash'][0]},
            "ANTIGEN_CHAINS_IN_STRUCT": {"value" : [study['study_parameters']['AntigenChainsInStructure']]},
            "study_type": {"value" : study['study_type']}
        }
        query_data_for_studies.append(study_query)

    return query_data_for_studies

def fasta_string_from_sequence(mutant_seq, master_seq, master_idstr, hash_algorithm='MD5'):
    idstr = derive_idstr_from_seq(mutant_seq)
   
    # Obtain a string describing the mutations between the two sequences
    seq_differences = selected_mutations_mut_from_list_of_tuples(diff_seqs(
        mutant_seq=recast_sequence_as_str(mutant_seq),
        master_seq=recast_sequence_as_str(master_seq)
    ))

    description_str = write_description_for_seqrecord(
        idstr, master_idstr, seq_differences
    )

    # Recast the sequences into BioPython SeqRecords
    seqrecord = recast_as_seqrecord(mutant_seq,
                                    id_if_str_or_seq=idstr,
                                    description_if_str_or_seq=description_str
                                    )
    # Convert the sequence record to a fasta 2 line string
    fasta_str = FastaIO.as_fasta_2line(seqrecord)

    if hash_algorithm.upper() == 'NONE':
        return fasta_str
    elif hash_algorithm.upper() == 'SHA1':
        m = hashlib.sha1()
    elif hash_algorithm.upper() == 'SHA256':
        m = hashlib.sha256()
    elif hash_algorithm.upper() == 'MD5':
        m = hashlib.md5()
    else:
        raise ValueError('Unsupported hashing algorithm {}!'.format(hash_algorithm))

    m.update(str(fasta_str).encode('utf-8'))
    return m.hexdigest()

def update_history_paths(history_path, target_history_path, files_directory, hash_type=None, remove_running_failed=False, write_csv_path=None):
    """
    Move all referenced files to new directory and generate new history file

    :param history_path: str, path to the old history file
    :param target_history_path:  str, path to the new history file
    :param files_directory: str, path to the new home of the stored files. Note
        that because we will be moving the structures here, this should agree
        with where the menu and config want them to be.  TODO: Evaluate
    :param hash_type: str or None, designating the type of hash to be computed
        on the files in files_directory.  If None, defaults to 'MD5'.
    :param write_csv_path: str, path to CSV file which will record the original
        location of files allowing request_id to be traced back to original workspace
    """

    if hash_type is None:
        hash_type = 'MD5'

    history_path = os.path.abspath(history_path)
    target_history_path = os.path.abspath(target_history_path)
    files_directory = os.path.abspath(files_directory)

    history = yaml_safe_load_with_lock(history_path)

    # If files_directory does not exist, create it
    if not os.path.exists(files_directory):
        os.makedirs(files_directory)

    # If there are any files in files_directory, hash them and generate a
    # dictionary of pdb and fasta file paths, where this dictionary is keyed by
    # the hash tuple and allows consolidation of structures

    hash_dict = {}
    l_in_fd = os.listdir(files_directory)
    for fi in l_in_fd:
        if os.path.splitext(fi)[-1] == '.fasta':
            hash_dict[tuple(hash_fasta_or_yaml(os.path.join(files_directory, fi), algorithm=hash_type))] = os.path.join(files_directory, fi)
        elif os.path.splitext(fi)[-1] == '.pdb':
            hash_dict[tuple(hash_pdb(os.path.join(files_directory, fi), algorithm=hash_type))] = os.path.join(files_directory, fi)

    # Call a recursive function that, like retrieve_nested..., descends into
    # the results from the other studies and moves files appropriately,
    # recording the newly moved files in the hash_dict
    history = modify_nested_filepaths(history, files_directory, hash_dict, hash_type, write_csv_path)

    if remove_running_failed:
        history['history'] = {
            ki: si for ki, si in history['history'].items()
            if si['status'] == 'FINISHED'
        }

    # Dump the new history yaml to the new location
    yaml_safe_dump_with_lock(history, target_history_path)


if __name__ == '__main__':
    pass
