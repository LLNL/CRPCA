# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
From a sequence, generate feature values, using a constructed cpx_dict
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
import multiprocessing
from copy import deepcopy

from vaccine_advance_core.featurization.featurize_complex import \
    get_feature_values_from_clean_pairs
from vaccine_advance_core.featurization.utils import \
    extract_partnered_chains, list_from_chain_spec
from vaccine_advance_core.featurization.interface_residues import \
    get_pairs_from_pdb_file
from vaccine_advance_core.featurization.reconcile_pdb_numbering import \
    align_chains_in_structure, antigen_in_cpx
from vaccine_advance_core.statium_automation.sequence_manipulation import \
    desc_mut_from_list_of_tuples, selected_mutations_mut_from_list_of_tuples
from vaccine_advance_core.featurization.tally_features import \
    size_class_combinations_reversible
from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    fasta_from_list_of_seqrecords, mutations_to_csv, \
    load_pdb_structure_from_file, write_cpx_dict_to_yaml_file, \
    read_cpx_dict_from_yaml_file
from vaccine_advance_core.featurization.bio_sequence_manipulation import \
    recast_as_seqrecord, recast_sequence_as_str, get_seq_id, \
    parse_seqrecord_description
from vaccine_advance_core.featurization.assay_utils import wt_prefix


def get_features_from_seq_and_cpx_dict(
        seq, master_key, cpx_dictionary, feature_types, list_of_cpx_names=None):
    """
    Get features for mutant sequence, using master_seq, pairs, and renumbering

    :param seq: str, Bio.Seq.Seq, or Bio.SeqRecord.SeqRecord; contains
        information on the mutant sequence
    :param master_key: str, giving the master (antigen) key into the Ag/Ab
        dictionary
    :param cpx_dictionary: dict, keyed by antigen designator; contains
        'master_seq' (str, giving the master antigen sequence), and 'complexes';
        cpx_dictionary[key]['complexes'] contains sub-dicts keyed by complex
        name designator; and finally, the leaf dicts
        contain:
        'name' (str, same as the complex name designator);
        'pairs' (a list of pairs of interacting residues);
        'chain_designator' (a str, giving the antigenic chain of interest);
        and
        'res_num_dict' (a dictionary containing the renumbering scheme from
        the WT sequence to the WT complex numbers).
    :param feature_types: string or list of strings, giving the types of
        features to be used.  Implemented choices include:
        all_aa_combinations_reversible
        chemical_class_combinations_reversible
        size_class_combinations_reversible
        charge_combinations_reversible
    :param list_of_cpx_names: None or list of str, where each is a key into the
        cpx_dictionary at the cpx_dictionary[master_key]['complexes'] level,
        corresponding to the name of the complex.  If None, will be set to the
        list of keys, the previous default behavior.
    :return: feature values, a Pandas DataFrame
    """

    # TODO: Review all TODO in get_features_from_seq_and_cpx_dict
    # TODO: Creation of res_num_dicts
    # TODO: should output DF include WT features?

    if not isinstance(feature_types, list):
        feature_types = [feature_types]

    # Compute the set of pairwise mutations from the master to the mutant
    # sequence
    mutations = diff_seqs(seq, cpx_dictionary[master_key]['master_seq'])

    # # For each complex here, use its pairs and get these feature
    # # representations. Note that there will be ONE (possibly multi-point)
    # # mutation represented, but possibly several complexes.
    feature_values = []
    feat_names = []  # TODO: assignment that works even if 0 runs through loop
    # For complexes involving each antigen:

    # If no specific complexes requested, do all present (old default)
    if list_of_cpx_names is None:
        list_of_cpx_names = list(cpx_dictionary[master_key]['complexes'].keys())

    # For each requested complex name, produce the feature vector
    for cpx_name in list_of_cpx_names:
        cpx_contents = cpx_dictionary[master_key]['complexes'][cpx_name]
        # For each individual complex involving this antigen:
        # Renumber the mutations according to this complex's numbering scheme
        renumbered_mutations = renumber_mutations(
            mutations, cpx_contents['res_num_dict'],
            override_chain=cpx_contents['chain_designator'])

        # Get the feature values from the pre-computed pairs
        # TODO: Debug get_feature_values_from_clean_pairs
        # Errors arising here, where an empty list is problematic.
        # get_feature_values_from_clean_pairs assumes it's going to get a list
        # of lists of quadruples, each quadruple describing a point mutation,
        # each list of quadruples describing an individual mutation of the
        # master, and the overall list of lists of quadruples describing a SET
        # of mutant variants of the master.
        tmp, feat_names = \
            get_feature_values_from_clean_pairs(
                cpx_contents['pairs'], [renumbered_mutations], feature_types
            )

        tmp_master, _ = \
            get_feature_values_from_clean_pairs(
                cpx_contents['pairs'], [[]], feature_types
            )
        # Append the feature values onto the growing list of such feature
        # values.
        feature_values.append(
            [get_seq_id(seq), str(master_key), recast_sequence_as_str(seq),
             '',  # TODO: Fill in antibody name somehow
             cpx_name,
             desc_mut_from_list_of_tuples(renumbered_mutations),
             desc_mut_from_list_of_tuples(mutations)] + tmp[0] + tmp_master[0]
        )
    # Hitting errors above where list index is out of range; no features??

    feat_names = ["AntigenID", "MasterAntigenID", "AntigenSequence",
                  "AntibodyID",
                  "Complex", "Mutation", "SeqNumberingMutation"] + feat_names \
                 + [wt_prefix + fni for fni in feat_names]

    # Convert to a DataFrame:
    # TODO: Check correctness of pdDataFrame conversion.
    # TODO: Add use of wt_separate_resolve, delta_feat_from_wt
    feature_values = pd.DataFrame(data=feature_values, columns=feat_names)

    return feature_values


def get_features_from_seq_and_cpx_dict_tuple_wrapper(tup):
    """
    Wrap get_features_from_seq_and_cpx_dict for quadruple inputs

    :param tup: tuple with four or five elements: seq, master_key, cpx_dictionary,
        feature_types, and optionally, list_of_cpx_names
    :return: feature values, a Pandas DataFrame
    """

    try:
        features=get_features_from_seq_and_cpx_dict(tup[0], tup[1], tup[2], tup[3], tup[4])
    except IndexError:
        features=get_features_from_seq_and_cpx_dict(tup[0], tup[1], tup[2], tup[3])
    return features


def diff_seqs(mutant_seq, master_seq, chain_name=None,
              aa_to_arbitrary_numbering_dict=None):
    """
    Return (renumbered) tuple-ized, string-wise differences between two seqs

    :param mutant_seq: str, the sequence which has been mutated TO
    :param master_seq: str, the sequence which has been mutated FROM
    :param chain_name: str, giving the single-letter chain designator
    :param aa_to_arbitrary_numbering_dict: dict, keyed by index (str(int)) into
        the sequence, and returning the str giving the (PDB) residue 'number'
    :return: list of four-element tuples, each describing a mutation

    >>> diff_seqs('ANGRYPYTHYN', 'HAPPYPYTHYN', chain_name='A')
    [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')]
    >>> diff_seqs('ANGRYPYTHYN', 'HAPPYPYTHYN', chain_name='A', aa_to_arbitrary_numbering_dict={str(i+1): str(i+1) for i in range(11)})
    [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')]
    >>> diff_seqs('ANGRYPYTHYN', 'HAPPYPYTHYN', chain_name='A', aa_to_arbitrary_numbering_dict={str(i+1): str(i+2) for i in range(11)})
    [('A', '2', 'H', 'A'), ('A', '3', 'A', 'N'), ('A', '4', 'P', 'G'), ('A', '5', 'P', 'R')]
    """

    # TODO: Address redundancies with get_renumbering_dict_from_alignment

    mutations = _direct_string_difference(mutant_seq, master_seq, chain_name)

    if aa_to_arbitrary_numbering_dict is not None:
        mutations = renumber_mutations(mutations, aa_to_arbitrary_numbering_dict)

    return mutations


def mutate_seq(master_seq, mutations, chain_name=None, renumbering_dict=None):
    """
    Perform the opposite operation from diff_seqs: apply the diff

    :param master_seq: str, giving the sequence to be mutated from
    :param mutations: list of four-element tuples, each describing a mutation
    :param chain_name: str, giving the single-letter chain designator
    :param renumbering_dict: dict, keyed by index (str(int)) into the sequence,
        and returning the str giving the (PDB) residue 'number'. Here, used to
        decode the residue numbers in the diff.
    :return: mutant_seq: str, giving the sequence mutated to.

    >>> mutate_seq('HAPPYPYTHYN', [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')], chain_name='A')
    'ANGRYPYTHYN'
    >>> mutations = [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')]
    >>> mutate_seq('HAPPYPYTHYN', mutations, chain_name='A', renumbering_dict={str(i+1): str(i+1) for i in range(11)})
    'ANGRYPYTHYN'
    >>> mutations = [(mi[0], str(int(mi[1]) + 1), mi[2], mi[3]) for mi in mutations]
    >>> mutate_seq('HAPPYPYTHYN', mutations, chain_name='A', renumbering_dict={str(i+1): str(i+2) for i in range(11)})
    'ANGRYPYTHYN'
    """

    for mi in mutations:
        try:
            a = int(mi[1])
        except ValueError:
            print(mi)

    if chain_name is not None:
        mutations = [mi for mi in mutations if mi[0] == chain_name]
    elif chain_name is None:
        if len(list(set([mi[0] for mi in mutations]))) > 1:
            raise ValueError('Handling is not yet implemented for the case '
                             'where more than one chain appears in mutations '
                             'but the chain_name is not specified; in '
                             'principle, this should be simple.')
    # TODO: case: chain name unspecified, mutations contains multiple chains

    if renumbering_dict is not None:
        tmp_renumbering_dict = {ii: ki for ki, ii in renumbering_dict.items()}
        mutations = renumber_mutations(mutations, tmp_renumbering_dict)
        # The mutations are now numbered by the integer positions in the
        # original sequence.
    mutations = [(mi[0], int(mi[1]), mi[2], mi[3]) for mi in mutations]

    mutant_seq = _direct_string_diff_apply(master_seq, mutations, chain_name)

    return mutant_seq


def _direct_string_diff_apply(master_seq, mutations, chain_name=None):
    """
    Apply a string-wise diff to the master_seq

    :param master_seq: str, Seq, or SeqRecord, the sequence is being
        mutated FROM
    :param mutations: list of tuples, where each tuple is A chain name (may not
        be this chain), str of the AA's number in the sequence, the master
        residue, and the mutant residue.
    :param chain_name: str, giving the chain name associated with the master_seq
    :return mutant_seq: str, giving the sequence mutated TO.

    >>> _direct_string_diff_apply('HAPPYPYTHYN', [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')], chain_name='A')
    'ANGRYPYTHYN'
    """

    if chain_name is None:
        # Check if the mutations all apply to a single chain
        unique_chain_names = \
            list(set([ti[0] for ti in mutations]))
        if len(unique_chain_names) > 1:
            raise ValueError('Unable to resolve mutations to multiple chains, '
                             'because this chain is unnamed.')
        chain_name = unique_chain_names[0]

    mutant_seq = recast_sequence_as_str(master_seq)
    for mi in mutations:
        if mi[0] != chain_name:
            continue
        python_idx = int(mi[1]) - 1
        if not mutant_seq[python_idx] == mi[2]:
            raise ValueError('Expected master residue {} but master contains '
                             '{}.'.format(mutant_seq[python_idx], mi[2]))
        mutant_seq = mutant_seq[:python_idx] + mi[3] \
                     + mutant_seq[(python_idx + 1):]
    return mutant_seq


def _direct_string_difference(mutant_seq, master_seq, chain_name=None):
    """
    Compute the string-wise differences between two sequences of equal length

    :param mutant_seq: str, Seq, or SeqRecord, the sequence which has been
        mutated TO
    :param master_seq: str, Seq, or SeqRecord, the sequence which has been
        mutated FROM
    :param chain_name: str, giving the single-letter chain designator
    :return: mutations, a list of tuples, where each tuple is the chain name,
        str of the AA's number in the sequence, the master residue, and the
        mutant residue.

    >>> _direct_string_difference('ANGRYPYTHYN', 'HAPPYPYTHYN', chain_name='A')
    [('A', '1', 'H', 'A'), ('A', '2', 'A', 'N'), ('A', '3', 'P', 'G'), ('A', '4', 'P', 'R')]
    """

    if chain_name is None:
        chain_name = ''

    # If they are not strings, produce local representations of mutant_seq and
    # master_seq
    mutant_seq_tmp = recast_sequence_as_str(mutant_seq)
    master_seq_tmp = recast_sequence_as_str(master_seq)
    # Ensure that mutant_seq and master_seq are the same number of characters.
    assert len(mutant_seq_tmp) == len(master_seq_tmp),\
        'Mutant and master sequences must be the same number of characters!'

    # Compute the difference
    mutations = []
    for i, mut_res_i, master_res_i in \
            zip(range(len(mutant_seq_tmp)), mutant_seq_tmp, master_seq_tmp):
        if mut_res_i != master_res_i:
            # Record mutation using the implicit indexing of the FASTA format
            mutations.append(
                (chain_name, str(i+1),
                 master_res_i, mut_res_i)
            )

    return mutations


def renumber_mutations(mutations, aa_numbering_dict=None, check_chains=False,
                       override_chain=None):
    """
    Translate mutations from one chain name and/or numbering scheme to another

    :param mutations: differences between two strings of amino acids, given as
        list of 4-element tuples of str, where these are chain name, residue
        number, original residue, and post-mutation residue.
    :param aa_numbering_dict: a dict to be applied to the residue numbers in
        each mutation tuple, giving new residue number str. Keyed by either a
        residue 'number' (or (chain, residue_number) tuple??!?!?)
    :param check_chains: optional, defaults False. bool, allow checking for
        mutations in MULTIPLE chains.
    :param override_chain: optional, defaults None: str, a chain designator to
        be used to overwrite the chain designator in each mutation tuple. Cannot
        be combined with True check_chains.
    :return: list of mutation tuples, with numbering and chain identifiers
        changed appropriately; those which do not have a renumbering are removed.
    """

    # TODO: Put renumber_mutations on firmer footing; seems very ad hoc ATM.

    renumbered_mutations = deepcopy(mutations)

    if aa_numbering_dict is None:
        return renumbered_mutations

    if not check_chains:
        # The dictionary is keyed by the residue identifying string (element
        # one in the tuple)
        for i, mut_res_i in enumerate(renumbered_mutations):

            if mut_res_i[1] in aa_numbering_dict.keys():
                if override_chain is None:
                    tmp_quadruple = (
                        mut_res_i[0],
                        aa_numbering_dict[mut_res_i[1]],
                        mut_res_i[2],
                        mut_res_i[3]
                    )
                    # TODO: handle case where the mutant residue does not appear in the renumbering dict; absent from structure?
                else:
                    tmp_quadruple = (
                        override_chain,
                        aa_numbering_dict[mut_res_i[1]],
                        mut_res_i[2],
                        mut_res_i[3]
                    )
                renumbered_mutations[i] = tmp_quadruple
            else:
                print('Failed to renumber mutation {}!'.format(mut_res_i))
                renumbered_mutations[i] = ()
    else:
        if override_chain is not None:
            raise ValueError('non-None override_chain may not be combined with '
                             'multi-chain mutations.')
        # The dictionary is keyed first by the chain then by
        # residue identifying string
        for i, mut_res_i in enumerate(renumbered_mutations):
            try:
                renumbered_mutations[i] = (
                    mut_res_i[0],
                    aa_numbering_dict[mut_res_i[0]][mut_res_i[1]],
                    mut_res_i[2],
                    mut_res_i[3]
                )
            except KeyError:
                print('Failed to renumber mutation {}!'.format(mut_res_i))
                renumbered_mutations[i] = ()

    renumbered_mutations = [rnmi for rnmi in renumbered_mutations if rnmi != ()]
    return renumbered_mutations


def make_cpx_dictionary(master_antigen_designators, master_antigen_sequences,
                        complex_names, complex_pdb_files, partnered_chains,
                        pair_function, score_to_len_threshold=None,
                        penalties=None):
    """
    Prepare cpx_dictionary for get_features_from_seq_and_cpx_dict

    The cpx_dictionary organizes information across multiple structures
    in which the same antigenic sequence appears. The same sequence may appear
    in multiple structures because: (1) these structures are different
    hypotheses about the interaction of the same two entities (i.e., docking
    poses); and/or (2) the same sequence (antigen) is interacting with multiple
    antibodies.

    The information collated in the cpx_dictionary will be used within
    get_features_from_seq_and_cpx_dict to get the corresponding (multiple)
    feature representations.

    :param master_antigen_designators: list of n str, containing the high-level
        keys to be used for the cpx_dict
    :param master_antigen_sequences: list of n (str, Seq, of SeqRecord),
        contains the sequences of the master (i.e., 'Wild type') antigens
        that appear in the structure files.
    :param complex_names: list of m str, giving the names of the complexes, to
        be used as second-level keys into cpx_dict.
    :param complex_pdb_files: list of m str, giving the paths to the complexes'
        pdb files.  These PDB files may or may not have header information,
        but all are assumed to have the ATOM lines giving the 3D positions of
        the residues in the complex.
    :param partnered_chains: list of m tuples of strings or lists; each element
        in a tuple gives a set (e.g., a protein; comma-separated string, or
        list) of chains within the complex of interacting chains.
    :param pair_function: str, naming the function that determines which pairs
        of residues are interacting
    :param score_to_len_threshold: float in [0.0, 1.0]; if None, defaults to
        0.70. Passed to antigen_in_cpx and align_chains_in_structure to
        determine match/not on the basis of alignment score.
    :param penalties: four-element list of ints; if None, defaults to
        [-100, -10, -5, -1]. Passed to antigen_in_cpx and
        align_chains_in_structure to determine match/not on the basis of
        alignment score.
    :return cpx_dictionary: dict, keyed by antigen designator; contains
        'master_seq' (str, giving the master antigen sequence), and dict
        'complexes'; cpx_dictionary[key]['complexes'] contains sub-dicts keyed
        by complex name designator; and finally, the leaf dicts contain:
        - 'name' (str, same as the complex name designator);
        - 'pairs' (a list of pairs of interacting residues);
        - 'chain_designator' (a str, giving the antigenic chain of interest);
        and
        - 'res_num_dict' (a dictionary containing the renumbering scheme from
        the WT sequence to the WT complex numbers).
    """
    if score_to_len_threshold is None:
        score_to_len_threshold = 0.70

    if score_to_len_threshold < 0.0 or score_to_len_threshold > 1.0:
        raise ValueError(
            'The threshold sequence agreement parameter is {}: must be in '
            '[0.0, 1.0].'.format(score_to_len_threshold)
        )

    if penalties is None:
        penalties = [-100, -10, -5, -1]

    # Load all of the pdbs
    complex_structures = []
    for cpx_n_j, cpx_f_j in zip(complex_names, complex_pdb_files):
        # load structure from file
        _ = load_pdb_structure_from_file(cpx_f_j, name=cpx_n_j)

        # # Is it really necessary to have the copy here?
        # complex_structures.append(deepcopy(_))
        complex_structures.append(_)

    print('Structures loaded.')

    cpx_dictionary = {}
    # The below is an all vs. all check; should we use some stronger prior
    # information?
    for mad_i, mas_i in \
        zip(master_antigen_designators, master_antigen_sequences):
        for cpx_n_j, struct_j, partnered_chains_j in \
                zip(complex_names, complex_structures, partnered_chains):
            # TODO: Evaluate parallelizing; deepcopy?
            # Check if the antigen sequence is present in the structure
            # tmp_struct = deepcopy(struct_j)
            tmp_struct = struct_j
            union_partnered_chains = set([])
            for pc_j_k in partnered_chains_j:
                union_partnered_chains = \
                    union_partnered_chains.union(list_from_chain_spec(pc_j_k))
            print(union_partnered_chains)
            print(tmp_struct.child_list)
            deepcopied_flag = False
            for chain_id in [cjk.get_id() for cjk in
                             tmp_struct.child_list[0].child_list]:
                if chain_id not in union_partnered_chains:
                    if not deepcopied_flag:
                        tmp_struct = deepcopy(tmp_struct)
                        deepcopied_flag = True
                    tmp_struct.child_list[0].detach_child(chain_id)
            chain_code = antigen_in_cpx(
                mas_i,
                tmp_struct,
                score_to_len_threshold=score_to_len_threshold,
                penalties=penalties
            )
            if chain_code is not None:
                if mad_i not in cpx_dictionary:
                    try:
                        cpx_dictionary[mad_i] = {'master_seq': str(mas_i.seq)}
                    except AttributeError:
                        cpx_dictionary[mad_i] = {'master_seq': str(mas_i)}
                    cpx_dictionary[mad_i]['complexes'] = {}

                mutable_chains, other_chains = \
                    extract_partnered_chains(chain_code, partnered_chains_j)

                # Below: tmp_struct is a BioPDB Structure; the other inputs are
                # strings, lists/tuples of strings, etc.
                pairs = get_pairs_from_pdb_file(
                    cpx_n_j, tmp_struct, mutable_chains, other_chains,
                    pair_function
                )
                align_dict, renumbering_dict = \
                    align_chains_in_structure(
                        tmp_struct, {chain_code: mas_i},
                        score_to_len_threshold=score_to_len_threshold,
                        penalties=penalties
                    )

                # TODO: ensure pairs stored in cpx_dict match master sequence
                # The pairs are derived from the structure file, not the nominal
                # master antigen sequence.

                cpx_dictionary[mad_i]['complexes'][cpx_n_j] = {
                    'name': cpx_n_j,
                    'pairs': pairs,
                    'chain_designator': chain_code,
                    'res_num_dict': renumbering_dict[chain_code]
                }

    return cpx_dictionary


def get_features_from_seqs_and_cpx_dict(
        seqs, master_keys, cpx_dictionary, feature_types, lists_of_cpx_names=None):
    """
    Wrapper for multiple calls to get_features_from_seq_and_cpx_dict

    :param seqs: list of str, Bio.Seq.Seq, or Bio.SeqRecord.SeqRecord; contains
        information on the mutant sequence
    :param master_keys: list of str, giving the master (antigen) key into the Ag/Ab
        dictionary
    :param cpx_dictionary: Dictionary keyed by str as in master_keys
    :param feature_types: string or list of strings, giving the types of
        features to be used.
    :param lists_of_cpx_names: list or None. If None, filled out to a list of
        Nones. Gives the list of complexes for which the features should be
        returned for a given master and seq; if None, all complexes' associated
        features will be returned.

    :return pd.DataFrame containing all feature values
    """

    if lists_of_cpx_names is None:
        lists_of_cpx_names = [None for si in seqs]

    # TODO: Add in seq/master_key indexing to DataFrame.
    features = None
    for seq_i, master_key_i, lcni in zip(seqs, master_keys, lists_of_cpx_names):
        f_seq_i = get_features_from_seq_and_cpx_dict(
            seq_i, master_key_i, cpx_dictionary, feature_types, lcni)
        if features is None:
            features = deepcopy(f_seq_i)
            # features = f_seq_i  # TODO: Check that this change is safe
            # # Above: removing the deepcopy should be safe because f_seq_i
            # is ONLY used here.
        else:
            features = pd.concat([features, f_seq_i], axis=0)
    return features


def get_features_from_seqs_and_cpx_dict_par(
        seqs, master_keys, cpx_dictionary, feature_types, lists_of_cpx_names=None):
    """
    Wrapper for parallel calls to get_features_from_seq_and_cpx_dict

    :param seqs: list of str, Bio.Seq.Seq, or Bio.SeqRecord.SeqRecord; contains
        information on the mutant sequence
    :param master_keys: list of str, giving the master (antigen) key into the Ag/Ab
        dictionary
    :param cpx_dictionary: Dictionary keyed by str as in master_keys
    :param feature_types: string or list of strings, giving the types of
        features to be used.
    :param lists_of_cpx_names: None or list of lists of str, giving the complex
        names associated with the seqs and master_keys

    :return pd.DataFrame containing all feature values
    """

    if lists_of_cpx_names is None:
        lists_of_cpx_names = [None for si in seqs]

    p = multiprocessing.Pool()
    features = p.imap(
        get_features_from_seq_and_cpx_dict_tuple_wrapper,
        ((si, mki, cpx_dictionary, feature_types, lcni)
         for si, mki, lcni in zip(seqs, master_keys, lists_of_cpx_names)),
        chunksize=50
    )
    p.close()
    p.join()
    features = pd.concat(features, axis=0)
    return features


def main(
        master_antigen_designators, master_antigen_sequences,
        mutant_seqs_by_master_antigen_designators,
        complex_names, complex_pdb_files, complex_partnered_chains,
        pair_function=None,
        feature_types=None,
        score_to_len_threshold_to_make_cpx_dict=None):
    """
    For antigenic seqs, get DataFrame with features of the associated complexes

    Given several inputs describing n mutant sequences and their n associated
    master sequences, and others describing m complexes and their constituent
    chains, calculate feature representations of the n mutants, each in possibly
    several contexts, determined by the number of complexes in which the
    corresponding master appears. Return this information in a Pandas DataFrame.

    This function primarily wraps make_cpx_dictionary and
    get_features_from_seqs_and_cpx_dict.

    make_cpx_dictionary organizes the
    information about A. mutant antigens and their related, 'master' antigens,
    contained in the inputs of size n described below; and B. complexes
    (containing 'master' antigens) and their constituent chains.  The result is
    a cpx_dict that provides a means of getting feature representations of the
    mutant antigens by reference to their associated masters and the complexes
    in which the masters appear.

    get_features_from_seqs_and_cpx_dict then uses the cpx_dict to acquire the
    feature representation of every mutant sequence in the various complexes
    within which its corresponding master appears.  These feature
    representations are returned as a Pandas DataFrame.

    :param master_antigen_designators: list of n str, containing the high-level
        keys to be used for the cpx_dict
    :param master_antigen_sequences: list of n (str, Seq, of SeqRecord),
        contains the sequences of the master (i.e., 'Wild type') antigens
        that appear in the structure files.
    :param mutant_seqs_by_master_antigen_designators: dict, keyed by master
        antigen designators, containing the mutant sequences to be associated
        with the master antigens.
    :param complex_names: list of m str, containing the names of the complexes;
        a single master antigen might appear in several named complexes. This
        relationship will be established within make_cpx_dictionary via
        antigen_in_cpx.
    :param complex_pdb_files: list of m str, giving paths to the respective pdb
        files of the m complexes.
    :param complex_partnered_chains: list of m tuples of strings or lists; tuple
        j gives the groupings of the interacting chains within complex j.
    :param pair_function: str, naming the function that determines which pairs
        of residues are interacting
    :param feature_types: list of strings of feature groupings; to be passed to
        tally_into_list
    :param score_to_len_threshold_to_make_cpx_dict: value in 0.0 - 1.0 or None
        (defaults to 0.70), used in computing the alignment.
    :return: features: a DataFrame, where each row corresponds to one mutant
        antigen IN A PARTICULAR COMPLEX IN WHICH THE MASTER APPEARS and
        columns are either indexing information or tally-based features of that
        (mutated) complex's interface.
    """

    if pair_function is None:
        pair_function = 'get_pairs_ca_ca'

    if feature_types is None:
        feature_types = ['chemical_class_combinations_reversible']

    cpx_dict = make_cpx_dictionary(
        master_antigen_designators, master_antigen_sequences, complex_names,
        complex_pdb_files, pair_function=pair_function,
        partnered_chains=complex_partnered_chains,
        score_to_len_threshold=score_to_len_threshold_to_make_cpx_dict
    )

    list_of_mutant_seqs = []
    list_of_master_antigen_designators = []
    list_of_mutant_ids = []
    for ki, mut_list_i in mutant_seqs_by_master_antigen_designators.items():
        if isinstance(mut_list_i, list):
            for mut_ij in mut_list_i:
                list_of_mutant_seqs.append(mut_ij)
                list_of_master_antigen_designators.append(ki)
                # TODO: get ID
                try:
                    parsed_description = parse_seqrecord_description(mut_ij)
                    id = parsed_description['id']
                except AttributeError:
                    id = ''
                list_of_mutant_ids.append(id)
        elif isinstance(mut_list_i, dict):
            for mut_ij, id_ij in zip(mut_list_i['mutants'], mut_list_i['mutant_ids']):
                list_of_mutant_seqs.append(mut_ij)
                list_of_master_antigen_designators.append(ki)
                list_of_mutant_ids.append(id_ij)

    # features = get_features_from_seqs_and_cpx_dict(
    #     list_of_mutant_seqs, list_of_master_antigen_designators, cpx_dict,
    #     feature_types)
    features = get_features_from_seqs_and_cpx_dict_par(
        list_of_mutant_seqs, list_of_master_antigen_designators, cpx_dict,
        feature_types)  # Defaulting to None for list_of_cpx_names
    return features


if __name__ == '__main__':
    pdb_path = '/Users/desautels2/GitRepositories/vaccine_advance_core/' \
               'test_gsk/5o14_A.pdb'

    # ### Test make_cpx_dictionary ###
    master_antigen_designators = ['Var1']
    master_antigen_sequences = [
        'VAADIGAGLADALTAPLDHKDKGLQSLTLDQSVRKNEKLKLAAQGAEKTYGNGDSLNTGKLKND'
        'KVSRFDFIRQIEVDGQLITLESGEFQVYKQSHSALTAFQTEQIQDSEHSGKMVAKRQFRIGDIAG'
        'EHTSFDKLPEGGRATYRGTAFGSDDAGGKLTYTIDFAAKQGNGKIEHLKSPELNVDLAAADIKPD'
        'GKRHAVISGSVLYNQAEKGSYSLGIFGGKAQEVAGSAEVKTVNGIRHIGLAAKQ'
    ]
    master_antigen_sequences = [recast_as_seqrecord(mas_i, id_if_str_or_seq='')
                                for mas_i in master_antigen_sequences]
    complex_names = ['5o14']
    complex_pdb_files = [pdb_path]
    partnered_chains = [('A', 'H,L')]
    pair_function = 'get_pairs_ca_ca'

    cpx_dictionary = make_cpx_dictionary(
        master_antigen_designators, master_antigen_sequences,
        complex_names, complex_pdb_files, partnered_chains,
        pair_function
    )

    print(cpx_dictionary)
    write_cpx_dict_to_yaml_file(cpx_dictionary, 'test_cpx_dict.yaml')
    cpx_dict_reloaded = read_cpx_dict_from_yaml_file('test_cpx_dict.yaml')
    print(cpx_dict_reloaded)
    # Note that the res_num_dict is no longer an OrderedDict after reloading

    # ### End test make_cpx_dictionary ###

    # ### Test main ###
    # Setup
    mutants = [[
        'VAADIGAGLADALTAPLDHKDKGLQSLTLDQSVRKNEKLKLAAQGAEKTYGNGDSLNTGKLKND'\
        'KVSRFDFIRQIEVDGQLITLESGEFQVYKQSHSALTAFQTEQIQDSEHSGKMVAKRQFRIGDIAG'\
        'EHTSFDKLPEGGRATYRGTAFGSDDAGGKLTYTIDFAAKQGNGKIEHLKSPELNVDLAAADIKPD'\
        'GKRHAVISGSVLY' + 'G' + 'QAEKGSYSLGIFGGKAQEVAGSAEVKTVNGIRHIGLAAKQ'
        ]]
    mutants = [[recast_as_seqrecord(mut_ij) for mut_ij in muts_i]
               for muts_i in mutants]
    corresponding_masters = ['Var1']
    mutant_seqs_by_masters = {mad_i: muts_i for mad_i, muts_i in
                              zip(corresponding_masters, mutants)}
    # Call
    features = main(
        master_antigen_designators, master_antigen_sequences,
        mutant_seqs_by_masters,
        complex_names, complex_pdb_files, partnered_chains,
        pair_function=pair_function,
        feature_types=[size_class_combinations_reversible])

    features.set_index(["Complex", "Mutation"], inplace=True)
    print(features)
    # ### End test main ###

    # ### Produce some prototype outputs for Adam ###
    # Write out the master_seq to a .fasta file
    # Recast the master sequence to a SeqRecord
    master_seqrecord = recast_as_seqrecord(
        cpx_dictionary[corresponding_masters[0]]['master_seq'],
        id=corresponding_masters[0]
    )

    # Write to file
    fasta_from_list_of_seqrecords(
        [master_seqrecord],
        'example_20190111_master_{}.fasta'.format(corresponding_masters[0])
    )

    # Get the mutations as a list of tuples
    list_of_point_mutations = diff_seqs(mutants[0][0],
                          master_seqrecord.seq, chain_name='A')
    # Convert the tuple representation to Adam's preferred format
    mutant_in_adams_string_fasta_num = \
        selected_mutations_mut_from_list_of_tuples(list_of_point_mutations)
    pdb_numbered_mutations = renumber_mutations(
        list_of_point_mutations,
        cpx_dictionary[corresponding_masters[0]]['complexes']['5o14']['res_num_dict']
    )
    mutant_in_adams_string_pdb_num = \
        selected_mutations_mut_from_list_of_tuples(pdb_numbered_mutations)

    # Write out the mutations to a .csv file, along with the master antigen
    # designator
    mutations_to_csv(
        'example_20190111_mutations_NA215G.csv',
        [master_seqrecord],
        list_of_lists_mutation_strings_fasta=
            [[mutant_in_adams_string_fasta_num]],
        list_of_lists_mutation_strings_pdb=
            [[mutant_in_adams_string_pdb_num]]
    )
