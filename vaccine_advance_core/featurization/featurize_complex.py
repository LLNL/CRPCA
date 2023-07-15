# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Obtain the feature values from the list of interacting residue pairs
"""

import csv
import copy
import os
import argparse
import re
import traceback

from vaccine_advance_core.featurization.tally_features import tally_into_list,\
    all_aa_combinations_reversible, chemical_class_combinations_reversible,\
    size_class_combinations_reversible, charge_combinations_reversible, \
    get_feature_names_tally_functions
from vaccine_advance_core.featurization.interface_residues import \
    get_pairs_from_pdb_file, \
    mutate_pairs_multiple

from vaccine_advance_core.statium_automation.sequence_manipulation \
    import read_mutant_strings, get_mutant_lines


def main_from_args(args):
    """
    Wrap main for use with argparser.

    :param args: argparser arguments structure
    """

    main(name=args.name, pdb_fpath=args.pdb, mut_fpath=args.mut_file,
         mutable_chains=args.mutable_chains,
         immutable_chains=args.immutable_chains,
         pairs_function=args.interaction,
         feature_types=args.feature_types,
         target_file=args.output)


def main(name, pdb_fpath, mut_fpath, mutable_chains, immutable_chains,
         pairs_function, feature_types, target_file):
    """
    Determine the feature values associated with mutants of a given complex

    :param name: string; complex's name
    :param pdb_fpath: string; path to the pdb file
    :param mut_fpath: string; path to the mutations file (e.g., mut_file.txt)
    :param mutable_chains: Chains in which mutants will appear. Note that the
        usage of 'mutable' here can conflate this meaning with 'antigen'
    :param immutable_chains:  The 'reference' chains with which the mutable
        chains are interacting; all pairwise interactions are between two
        residues, one of which is in the mutable chains and one of which is in
        the immutable chains.
    :param pairs_function: string, giving a function to be used to determine
        which pairs of residues are interacting. These functions compare
        distances between wild-type complex alpha carbons (ca) or side chains
        (sc) in the mutable chain (listed first) and their counterparts in the
        immutable chain (listed second).  Note that the use of the single
        determination of interacting/not in the WT complex means that, e.g.,
        introducing a larger side chain in a mutant will not result in the
        recognition of a new pairwise interaction, even if this would be the
        case in reality.
    :param feature_types: string or list of strings, giving the types of
        features to be used.  Implemented choices include:
        all_aa_combinations_reversible
        chemical_class_combinations_reversible
        size_class_combinations_reversible
        charge_combinations_reversible
        All of these functions are within the tally_features module.
    :param target_file: String, giving path to the destination file to which
        the feature representation will be written.
    """

    feature_values, feature_names = get_feature_values(name, pdb_fpath,
        mut_fpath, mutable_chains, immutable_chains, pairs_function,
                                                       feature_types)

    write_feature_file(target_file, feature_names, feature_values)


def get_feature_values(name, pdb_fpath, mut_fpath, mutable_chains, immutable_chains,
         pairs_function, feature_types):
    """
    Obtain the feature values and names for a given complex's mutants

    :param name: string; complex's name
    :param pdb_fpath: string; path to the pdb file
    :param mut_fpath: string; path to the mutations file (e.g., mut_file.txt)
    :param mutable_chains: Chains in which mutants will appear. Note that the
        usage of 'mutable' here can conflate this meaning with 'antigen'
    :param immutable_chains:  The 'reference' chains with which the mutable
        chains are interacting; all pairwise interactions are between two
        residues, one of which is in the mutable chains and one of which is in
        the immutable chains.
    :param pairs_function: string, giving a function to be used to determine
        which pairs of residues are interacting. These functions compare
        distances between wild-type complex alpha carbons (ca) or side chains
        (sc) in the mutable chain (listed first) and their counterparts in the
        immutable chain (listed second).  Note that the use of the single
        determination of interacting/not in the WT complex means that, e.g.,
        introducing a larger side chain in a mutant will not result in the
        recognition of a new pairwise interaction, even if this would be the
        case in reality.
    :param feature_types: string or list of strings, giving the types of
        features to be used.  Implemented choices include:
        all_aa_combinations_reversible
        chemical_class_combinations_reversible
        size_class_combinations_reversible
        charge_combinations_reversible
        All of these functions are within the tally_features module.
    :return feature_values, feature_names: The list of feature vectors and the
        list of feature names
    """

    # Get the set of mutations from the mut_file
    mutations_list_of_strings = get_mutant_lines(mut_fpath)
    mutations = read_mutant_strings(mutations_list_of_strings)

    # Above: Statium_automation has some stuff here:
    # - write_scored_mutants reads this file (line-by-line, discarding blank
    # lines; WT must be designated by a before = after mutation).
    # - read mutant strings turns the list of lines describing the mutations
    # into lists of quadruples.
    if not isinstance(feature_types, list):
        feature_types = [feature_types]

    for i, fti in enumerate(feature_types):
        if fti == 'all_aa_combinations_reversible':
            feature_types[i] = all_aa_combinations_reversible
        elif fti == 'chemical_class_combinations_reversible':
            feature_types[i] = chemical_class_combinations_reversible
        elif fti == 'size_class_combinations_reversible':
            feature_types[i] = size_class_combinations_reversible
        elif fti == 'charge_combinations_reversible':
            feature_types[i] = charge_combinations_reversible

    # Obtain the base ('wild-type') pairs from the structure. All mutants are
    # described and derived relative to this structure.

    try:
        pairs = get_pairs_from_pdb_file(
            name, pdb_fpath, mutable_chains, immutable_chains, pairs_function)
    except TypeError as err:
        if re.search('disordered_add', traceback.format_exc()):
            print('BIO.Pdb has encountered an error with disordered atoms, '
                  'and cannot load the structure.')
            print(err)
            feature_names = get_feature_names_tally_functions(feature_types)
            pseudofeatures = [
                [None for fni in feature_names] for mi in mutations
            ]
            return pseudofeatures, feature_names
        else:
            raise

    feature_values, feature_names = get_feature_values_from_clean_pairs(pairs, mutations, feature_types)

    return feature_values, feature_names

def get_feature_values_from_clean_pairs(pairs, mutations, feature_types):
    """
    Given WT pairs, apply mutations and generate feature values

    :param pairs: list containing sorted, tuple-ized pairs, where each tuple is of
        the following form: (chainCode, residueLocation, singleLetterAACode)
    :param mutations: list of lists; each of these 1st level lists is the list
        of mutations to apply to the base pairs set.
    :param feature_types: string or list of strings, giving the types of
        features to be used.  Implemented choices include:
        all_aa_combinations_reversible
        chemical_class_combinations_reversible
        size_class_combinations_reversible
        charge_combinations_reversible
        All of these functions are within the tally_features module.
    :return feature_values, feature_names: The list of feature vectors and the
        list of feature names
    """

    # TODO: Case where the list of mutations is empty.

    # Apply the mutations and obtain the features
    mutant_pairs = []
    feature_names = []  # Will be a fixed-size list after the first iteration
    feature_values = []  # Will grow to be a list of lists
    for mi in mutations:
        pairs_tmp = copy.deepcopy(pairs)
        pairs_tmp = mutate_pairs_multiple(pairs_tmp, mi)
        mutant_pairs.append(pairs_tmp)
        feature_values_tmp, feature_names_tmp = tally_into_list(pairs_tmp, feature_types)
        if not feature_names:
            feature_names = feature_names_tmp
        assert feature_names == feature_names_tmp, \
               "Calls to tally_into_list should return the exact same set of "\
               "feature names for multiple calls."
        feature_values.append(feature_values_tmp)

    return feature_values, feature_names


def write_feature_file(target_file, feature_names, feature_values):
    """
    Write out the feature vectors to the target file.

    :param target_file: str, giving relative path to target csv file
    :param feature_names: list of column headers (str)
    :param feature_values: list of lists of feature values; comes from
        get_feature_values_from_clean_pairs and tally_into_list.
    """

    # The mutations are associated with the features implicitly
    # TODO: Explicitly associate a given mutation string with the features
    with open(os.path.abspath(target_file), "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(feature_names)
        for mutation_feature_value in feature_values:
            # print(mutation_feature_value)
            writer.writerow(mutation_feature_value)


def setup_argparser():
    """
    Set up argparser for featurize_complex main

    :return: argparse.ArgumentParser configured appropriately
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name",
                        help="The complex name (PDB code)",
                        type=str, default=None)
    parser.add_argument("-p", "--pdb", type=str,
                        help="PDB structure file")
    parser.add_argument("-m", "--mut_file", type=str,
                        help="File containing (sets of) mutations, where each "
                             "line is a separate mutant of the base molecule "
                             "to be assessed")
    parser.add_argument('-a', '--mutable_chains', type=str,
                        help="The mutable chains, as a comma-separated list.")
    parser.add_argument('-b', '--immutable_chains', type=str,
                        help="The immutable (reference) chains, as a comma-"
                             "separated list.")
    parser.add_argument("-i", '--interaction', type=str,
                        help='The types of interactions to be identified; e.g.,'
                             ' get_pairs_sc_sc')
    parser.add_argument('-f', '--feature_types', type=str,
                        help='Comma-separated list of feature sets to be used '
                             'in creating the feature vector; '
                             'see tally_features.py')
    parser.add_argument("-o", "--output",
                        help="The file which will contain the final features.",
                        type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    main_from_args(args)
