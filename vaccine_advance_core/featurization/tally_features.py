# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Compute tally-based features from pairs of interacting residues
"""
from __future__ import print_function
import warnings

from Bio.PDB.Polypeptide import aa1


individual_aa_feature_list = [{'name': aa1_i, 'set': [aa1_i]} for aa1_i in aa1]
individual_aa_feature_list = sorted(individual_aa_feature_list,
                                    key=lambda k: k['name'])
# Note: Because the original is alphabetical, the above line should be redundant


# Ref: https://www.compoundchem.com/2014/09/16/aminoacids/
chemical_classes_aa_feature_list = [
    {'name': 'Aliphatic', 'set': ['A', 'G', 'I', 'L', 'P', 'V']},
    {'name': 'Aromatic', 'set': ['F', 'W', 'Y']},
    {'name': 'Acidic', 'set': ['D', 'E']},
    {'name': 'Basic', 'set': ['R', 'H', 'K']},
    {'name': 'Hydroxilic', 'set': ['S', 'T']},
    {'name': 'Sulfurous', 'set': ['C', 'M']},
    {'name': 'Amidic', 'set': ['N', 'Q']}
]
chemical_classes_aa_feature_list = sorted(chemical_classes_aa_feature_list,
                                          key=lambda k: k['name'])

# Ref: http://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html
size_classes_aa_feature_list = [
    {'name': 'VeryLarge', 'set': ['F', 'W', 'Y']},
    {'name': 'Large', 'set': ['I', 'L', 'M', 'K', 'R']},
    {'name': 'Medium', 'set': ['V', 'H', 'E', 'Q']},
    {'name': 'Small', 'set': ['C', 'P', 'T', 'D', 'N']},
    {'name': 'VerySmall', 'set': ['A', 'G', 'S']}
]
size_classes_aa_feature_list = sorted(size_classes_aa_feature_list,
                                      key=lambda k: k['name'])

charge_classes_aa_feature_list = [
    {'name': 'Positive', 'set': ['R', 'H', 'K']},
    {'name': 'Negative', 'set': ['D', 'E']},
    {'name': 'Uncharged', 'set':
        ['A', 'N', 'C', 'Q', 'G', 'I', 'L', 'M',
         'F', 'P', 'S', 'T', 'W', 'Y', 'V']
     }
]
charge_classes_aa_feature_list = sorted(charge_classes_aa_feature_list,
                                        key=lambda k: k['name'])


def in_paired_sets(pair, set1, set2, reversible=False):
    """
    Determine if a pair's elements fall in a pair of sets

    :param pair: Tuple, containing pair of residues (triples)
    :param set1: Set or list of residues (single-character AA codes)
    :param set2: Set or list of residues (single-character AA codes)
    :param reversible: Logical; can pair be in set2 x set1, as opposed to
        set1 x set2?
    :return: Logical value for pair's membership in the described class of AA
        pair interactions.
    """

    pass_logical = False
    if pair[0][-1] in set1 and pair[1][-1] in set2:
        pass_logical = True
    elif reversible and (pair[1][-1] in set1 and pair[0][-1] in set2):
        pass_logical = True

    return pass_logical


def all_aa_combinations_reversible(pair, tally=None):
    """
    Given a pair, determine what combination (not permutation) of AA's it is

    :param pair: Tuple, containing pair of residues (triples)
    :param tally: List, containing one element for each (reversible) AA
        interaction name
    :return: Updated tally, feature_names corresponding to the AA pairs.
    """

    feature_names = get_feature_names_dict_based_tally_reversible(
        individual_aa_feature_list)
    if pair is not None:
        tally = generic_dict_based_tally_reversible(
            pair, individual_aa_feature_list, tally)
    return tally, feature_names


def chemical_class_combinations_reversible(pair, tally=None):
    """
    Given a pair, determine what combination of chemical classes the AA's are.

    :param pair: Tuple, containing pair of residues (triples)
    :param tally: List, containing one element for each (reversible) AA
        interaction name
    :return: Updated tally, feature_names corresponding to the AA pairs.
    """

    feature_names = get_feature_names_dict_based_tally_reversible(
        chemical_classes_aa_feature_list)
    if pair is not None:
        tally = generic_dict_based_tally_reversible(
            pair, chemical_classes_aa_feature_list, tally)
    return tally, feature_names


def size_class_combinations_reversible(pair, tally=None):
    """
    Given a pair, determine what combination of size classes the AA's are.

    :param pair: Tuple, containing pair of residues (triples)
    :param tally: List, containing one element for each (reversible) AA
        interaction name
    :return: Updated tally, feature_names corresponding to the AA pairs.
    """

    feature_names = get_feature_names_dict_based_tally_reversible(
        size_classes_aa_feature_list)
    if pair is not None:
        tally = generic_dict_based_tally_reversible(
            pair, size_classes_aa_feature_list, tally)
    return tally, feature_names


def charge_combinations_reversible(pair, tally=None):
    """
    Given a pair, assign to --, -+, -U, ++, +U, UU pairs.

    :param pair: Tuple, containing pair of residues (triples)
    :param tally: List, containing one element for each (reversible) AA
        interaction name
    :return: Updated tally, feature_names corresponding to the AA pairs.

    >>> charge_combinations_reversible((('A', '125', 'F'), ('B', '1', 'Y')))
    ([0, 0, 0, 0, 0, 1], ['NumNegativeNegative', 'NumNegativePositive', 'NumNegativeUncharged', 'NumPositivePositive', 'NumPositiveUncharged', 'NumUnchargedUncharged'])
    """

    feature_names = get_feature_names_dict_based_tally_reversible(
        charge_classes_aa_feature_list)
    if pair:
        tally = generic_dict_based_tally_reversible(
            pair, charge_classes_aa_feature_list, tally)
    return tally, feature_names


def get_feature_names_dict_based_tally_reversible(list_of_dictionaries):
    """
    Get the names of the features associated with the listed dictionaries

    :param list_of_dictionaries: list of individual category dictionaries, where
        each category dictionary has 'name' and 'set' entries.
    :return: the set of pairwise combinations of the 'name' entries in
        list_of_dictionaries child dictionaries.
    """

    # TODO: Fill in docstring
    feature_names = []
    for i, group_i in enumerate(list_of_dictionaries):
        for j, group_j in enumerate(list_of_dictionaries):
            if j < i:
                continue
            feature_names.append(
                "".join(["Num"] + sorted([group_i['name'], group_j['name']])))

    return feature_names


def generic_dict_based_tally_reversible(pair, list_of_dictionaries, tally=None):
    """
    Tally matches with the pairs of categories in the list of dictionaries

    :param pair: Tuple, containing pair of residues (triples).
    :param list_of_dictionaries: list of dict describing the individual
        categories; each dictionary corresponds to a single category and has a
        'name' and 'set'.
    :param tally: List, containing one element for each (reversible) AA
        interaction name.
    :return: Updated tally, feature_names corresponding to the AA pairs.
    """

    tally_tmp = []
    for i, group_i in enumerate(list_of_dictionaries):
        for j, group_j in enumerate(list_of_dictionaries):
            if j < i:
                continue
            if in_paired_sets(pair, group_i['set'], group_j['set'],
                              reversible=True):
                tally_tmp.append(1)
            else:
                tally_tmp.append(0)

    if tally is None:
        tally = [0 for tally_tmp_i in tally_tmp]

    tally = [tally_i + tally_tmp_i
             for tally_i, tally_tmp_i in zip(tally, tally_tmp)]
    return tally


def count_aromatic(pair, tally=None):
    """
    Increment the count of aromatic-aromatic pairs

    :param pair: tuple-ized pair of interacting amino acids
    :param tally: None, or 1-element list (# aromatic-aromatic pairs)
    :return: tally (2-element list; incremented) and feature_names (2-element
        list)
    """

    feature_names = ["NumAromaticAromatic"]

    if pair is None:
        return None, feature_names

    if tally is None:
        tally = [0]
    else:
        if isinstance(tally, list):
            if len(tally) != 1:
                raise ValueError("The tally list must have an equal number of "
                                 "entries to the number of expected outputs.")
        else:
            raise ValueError("The tally object must be either None or a "
                             "single-element list.")

    aromatics = ['F', 'W', 'Y']
    if pair[0][-1] in aromatics and pair[1][-1] in aromatics:
        tally[0] += 1

    return tally, feature_names


def tally_into_list(pairs, tally_functions):
    """
    Apply tally_function to all of the pairs, returning the total tally

    :param pairs: list of interacting AA pairs
    :param tally_functions: a function which accepts a pair (and optionally a
        list of current tallies) and returns a list of current tallies, along
        with the feature names
    :return: list, giving final tallies, and another list giving feature names

    >>> tally_into_list([(('A', '125', 'F'), ('B', '1', 'Y')), (('A', '125', 'F'), ('B', '2', 'G'))], count_aromatic)
    ([1], ['NumAromaticAromatic'])
    """

    tally_all = []
    feature_names_all = []

    if not isinstance(tally_functions, list):
        tally_functions = [tally_functions]
    tally_functions = get_tally_functions_from_function_names(tally_functions)

    for tfi in tally_functions:
        tally = None
        feature_names = None
        if pairs == []:
            # Above: If there are simply no interacting pairs to be counted
            _, feature_names = tfi(None)
            tally = [0 for fni in feature_names]
        else:
            for pair in pairs:
                tally, feature_names = tfi(pair, tally)
        tally_all = tally_all + tally
        feature_names_all = feature_names_all + feature_names

    return tally_all, feature_names_all


def get_feature_names_tally_functions(tally_functions):
    """
    Get the list of feature names associated with the set of tally functions

    :param tally_functions: str or list (where str is cast into a 1-element
        list) giving one or more tally functions
    :return: feature_names_all, a list of the names of the features counted by
        those tally functions.
    """

    if not isinstance(tally_functions, list):
        tally_functions = [tally_functions]
    tally_functions = get_tally_functions_from_function_names(tally_functions)

    feature_names_all = []
    for tfi in tally_functions:
        _, feature_names = tfi(None)
        feature_names_all += feature_names
    return feature_names_all


def get_tally_functions_from_function_names(tally_function_name_list):
    """
    Convert a list of names into the actual functions

    :param tally_function_name_list: list, nominally of str giving the names of
        the functions.
    :return: tally_functions, a list of the actual functions
    """

    # TODO: Put this dictionary of tally functions elsewhere?
    dict_for_function_lookup = {
        'count_aromatic': count_aromatic,
        'all_aa_combinations_reversible': all_aa_combinations_reversible,
        'chemical_class_combinations_reversible': chemical_class_combinations_reversible,
        'size_class_combinations_reversible': size_class_combinations_reversible,
        'charge_combinations_reversible': charge_combinations_reversible
    }

    tally_functions = []
    for tfn_i in tally_function_name_list:
        if isinstance(tfn_i, str):
            if tfn_i in dict_for_function_lookup.keys():
                tally_functions.append(dict_for_function_lookup[tfn_i])
            else:
                warnings.warn('Tally function {} not recognized!'.format(tfn_i))
                tally_functions.append(tfn_i)  # better to break things here, than silently fail by passing, e.g., None.
        else:
            tally_functions.append(tfn_i)

    return tally_functions


if __name__ == '__main__':
    import doctest
    doctest.testmod()
