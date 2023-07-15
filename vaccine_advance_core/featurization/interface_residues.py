# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Residue pairs from .pdb or Bio.PDB.Structure; provide utils for pair mutation
"""
from __future__ import print_function

import itertools
from copy import deepcopy

from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import three_to_one
from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    load_pdb_structure_from_file
from vaccine_advance_core.featurization.utils import list_from_chain_spec

# TODO: Clean up interface_residues.py, splitting into new modules


max_len_side_chain = 10  # Angstroms

# ##########################################################
# ### Functions for directly manipulating PDB structures ###
# ##########################################################


def get_ag_ab_from_pdb(name, fname, antigen_chain_spec, antibody_chain_spec):
    """
    Extract sub-structures from a PDB file

    :param name: Structure name to be given to parsed structure, if created.
    :param fname: either a BioPDB Structure of a str, specifying a file from
        which to parse such a structure.
    :param antigen_chain_spec: list or comma-separated string giving a set of
        chain identifiers
    :param antibody_chain_spec: list or comma-separated string giving a set of
        chain identifiers
    :return: two BioPDB substructures, corresponding to the partnered macromolecules
    """

    if isinstance(fname, Structure):
        structure = deepcopy(fname)
    else:
        structure = load_pdb_structure_from_file(fname, name=name)

    # Turn the antibody/antigen chains strings into lists.
    antigen_chains = list_from_chain_spec(antigen_chain_spec)
    antibody_chains = list_from_chain_spec(antibody_chain_spec)

    # Obtain the antigen and antibody children from the structure
    antigen = extract_children_from_structure(structure, antigen_chains)
    antibody = extract_children_from_structure(structure, antibody_chains)

    return antigen, antibody


def extract_children_from_structure(structure, list_of_child_chains):
    """
    Extract corresponding substructures via chains' (letter) identifiers

    :param structure: BioPython-formatted structure
    :param list_of_child_chains: a list (of strings) giving the chain names to
        be matched.
    :return: List of matching substructures
    """

    chain_ids = [ci.id for ci in structure.child_list[0].child_list]

    chain_list_out = []
    for ci in list_of_child_chains:
        chain_list_out.append(structure.child_list[0].child_list[chain_ids.index(ci)])
    return chain_list_out


# ### Extract interacting pairs from a structure: ###

def get_pairs_from_pdb_file(name, pdb_fpath, mutable_chains, immutable_chains, pairs_function):
    """
    Load the contents of the designated pdb_file and get the interacting pairs

    :param name: name of the structure
    :param pdb_fpath: str, path to .pdb file file.
    :param mutable_chains: list or comma-separated string giving a set of chain
        identifiers
    :param immutable_chains: list or comma-separated string giving a set of
        chain identifiers
    :param pairs_function: str giving the method of determining interacting
        pairs in the structure.
    :return: (cleaned) pairs, a list of tuples, where each element in the
        tuple is a tupleized residue, mutable chain residue given first.
    """

    if pairs_function == 'get_pairs_sc_sc':
        pairs = get_pairs_sc_sc(
            name, pdb_fpath, mutable_chains, immutable_chains, 5.0)
    elif pairs_function == 'get_pairs_ca_sc':
        pairs = get_pairs_ca_sc(
            name, pdb_fpath, mutable_chains, immutable_chains, 7.0)
    elif pairs_function == 'get_pairs_ca_ca':
        pairs = get_pairs_ca_ca(
            name, pdb_fpath, mutable_chains, immutable_chains, 10.0)
    else:
        raise ValueError("Unrecognized pairs_function; choices are "
                         "get_pairs_sc_sc, get_pairs_ca_sc, get_pairs_ca_ca")

    pairs = _clean_interacting_biopdb_residue_pairs(pairs)
    return pairs


def get_pairs_sc_sc(name, fname, antigen_chains, antibody_chains, interface_angstroms=None):
    """
    Define interaction by minimum distance between any side chain atoms

    :param name: Structure name to be used by BioPython PDB parser
    :param fname: File name of the pdb
    :param antigen_chains: The (mutable) chains from the structure
    :param antibody_chains: The (immutable) reference chains from the structure
    :param interface_angstroms: numeric; distance (# of angstroms) from a
        reference SC atoms to mutable chain SC atoms. If any pair of atoms is
        within this distance, this pair of residues is declared to be
        interacting and will be returned.
    :return: list of tuple-ized interacting pairs
    """

    # Default to <5 A inter-atomic distance ==> interaction
    if interface_angstroms is None:
        interface_angstroms = 5
    # parser = Bio.PDB.PDBParser()
    #
    # # Ingest the structure
    # structure = parser.get_structure(name, fname)
    #
    # # Turn the antibody/antigen chains strings into lists.
    # antigen_chains = list_from_chain_spec(antigen_chains)
    # antibody_chains = list_from_chain_spec(antibody_chains)
    #
    # # Obtain the antigen and antibody children from the structure
    # antigen  = extract_children_from_structure(structure, antigen_chains)
    # antibody = extract_children_from_structure(structure, antibody_chains)

    antigen, antibody = get_ag_ab_from_pdb(name, fname, antigen_chains, antibody_chains)

    close_residues_ag = set()
    close_residues_ab = set()
    interacting_pairs = []
    for ri in itertools.chain(*[ag_ci.get_residues() for ag_ci in antigen]):
        if 'CA' not in ri.child_dict:
            # Necessary for cases where, e.g.,
            # waters are in the active chain somehow
            continue
        for bi in itertools.chain(
                *[ab_ci.get_residues() for ab_ci in antibody]):
            if 'CA' not in bi.child_dict:
                continue
            inter_ca_dist = bi.child_dict['CA'] - ri.child_dict['CA']
            fails_max_dist = inter_ca_dist > \
                             interface_angstroms + 2.0 * max_len_side_chain
            if fails_max_dist:
                continue

            interact = False
            for ari in ri.get_atoms():
                # if bi in close_residues_ab:
                #     continue

                for abi in bi.get_atoms():
                    if interact:
                        break
                    if abi - ari < interface_angstroms:
                        close_residues_ag.add(ri)
                        close_residues_ab.add(bi)
                        interacting_pairs.append((ri, bi))
                        interact = True

    # return close_residues_ag, close_residues_ab, interacting_pairs
    return interacting_pairs


def get_pairs_ca_ca(
        name, fname, antigen_chains, antibody_chains, interacting_ca_dist=None):
    """
    Define interaction by dist: chain of interest CA's to ref CA's

    :param name: Structure name to be used by BioPython PDB parser
    :param fname: File name of the pdb
    :param antigen_chains: The (mutable) chains from the structure
    :param antibody_chains: The (immutable) reference chains from the structure
    :param interacting_ca_dist: numeric; distance (# of angstroms) from a
        reference CA to the CA atom of the mutable chain. If a mutable
        chain residue's CA is within this distance, this pair of residues is
        declared to be interacting and will be returned.
    :return: list of tuple-ized interacting pairs
    """

    # Default to <15 A inter-CA distance ==> interaction
    if interacting_ca_dist is None:
        interacting_ca_dist = 15
    # parser = Bio.PDB.PDBParser()
    #
    # # Ingest the structure
    # structure = parser.get_structure(name, fname)
    #
    # # Turn the antibody/antigen chains strings into lists.
    # antigen_chains = list_from_chain_spec(antigen_chains)
    # antibody_chains = list_from_chain_spec(antibody_chains)
    #
    # # Obtain the antigen and antibody children from the structure
    # antigen  = extract_children_from_structure(structure, antigen_chains)
    # antibody = extract_children_from_structure(structure, antibody_chains)

    antigen, antibody = get_ag_ab_from_pdb(name, fname, antigen_chains,
                                           antibody_chains)

    interacting_pairs = []
    for ag_ri in itertools.chain(*[ag_ci.get_residues() for ag_ci in antigen]):
        if 'CA' not in ag_ri.child_dict:
            # Necessary for cases where, e.g.,
            # waters are in the active chain somehow
            continue
        for ab_ri in itertools.chain(
                *[ab_ci.get_residues() for ab_ci in antibody]):
            if 'CA' not in ab_ri.child_dict:
                continue
            inter_CA_dist = ab_ri.child_dict['CA'] - ag_ri.child_dict['CA']
            if inter_CA_dist < interacting_ca_dist:
                interacting_pairs.append((ag_ri, ab_ri))

    return interacting_pairs


def get_pairs_ca_sc(
        name, fname, chains_of_interest, reference_chains, interacting_ca_sc_dist=None):
    """
    Define interaction by dist: chain of interest CA's to ref side chain atoms

    :param name: Structure name to be used by BioPython PDB parser
    :param fname: File name of the pdb
    :param chains_of_interest: The (mutable) chains from the structure
    :param reference_chains: The (immutable) reference chains from the structure
    :param interacting_ca_sc_dist: numeric; distance (# of angstroms) from a
        reference side chain to the CA atom of the mutable chain. If a mutable
        chain residue's CA is within this distance, this pair of residues is
        declared to be interacting and will be returned.
    :return: list of tuple-ized interacting pairs
    """

    # Default to <8 A CA-any side chain atom ==> interaction
    if interacting_ca_sc_dist is None:
        interacting_ca_sc_dist = 8
    # parser = Bio.PDB.PDBParser()
    #
    # # Ingest the structure
    # structure = parser.get_structure(name, fname)
    #
    # # Turn the antibody/antigen chains strings into lists.
    # chains_of_interest = list_from_chain_spec(chains_of_interest)
    # reference_chains = list_from_chain_spec(reference_chains)
    #
    # # Obtain the antigen and antibody children from the structure
    # interest  = extract_children_from_structure(structure, chains_of_interest)
    # reference = extract_children_from_structure(structure, reference_chains)

    interest, reference = get_ag_ab_from_pdb(name, fname, chains_of_interest,
                                             reference_chains)

    interacting_pairs = []
    for int_ri in itertools.chain(*[ag_ci.get_residues() for ag_ci in interest]):
        if 'CA' not in int_ri.child_dict:
            continue
        for ref_ri in itertools.chain(
                *[ab_ci.get_residues() for ab_ci in reference]):
            if 'CA' not in ref_ri.child_dict:
                continue
            # Check for failure if the CA's are too far apart
            inter_ca_dist = ref_ri.child_dict['CA'] - int_ri.child_dict['CA']
            fails_max_dist = inter_ca_dist > \
                             interacting_ca_sc_dist + 1.0 * max_len_side_chain
            if fails_max_dist:
                continue
            # If any atom in the reference side chain is sufficiently close,
            # an interaction is occurring.
            for ref_ai in ref_ri.get_atoms():
                if ref_ai - int_ri.child_dict['CA'] < interacting_ca_sc_dist:
                    interacting_pairs.append((int_ri, ref_ri))
                    break
    return interacting_pairs


# ###############################################################
# ### Convert BioPDB residue to tuple and compare such tuples ###
# ###############################################################


def tupleize_biopdb_residue(biopdb_residue):
    """
    tuple (chainID, residue location, single-letter code) for a Bio.PDB residue

    :param biopdb_residue: a Bio.PDB residue
    :return: The triple of containing the chainID (string), the residue
        location (string; may be a non-numeric value, e.g., 60a), and the
        single-letter AA code (string)
    """

    # Note that the biopdb_residue.full_id[3] is the residue ID designator,
    # itself a triple; the first element is a string giving the 'het code' (' '
    # if no het code), the second is an integer giving the residue number, the
    # third element is insertion code ('A', 'B', etc.; ' ' if no insertion
    # code).  The if str(ei).strip() is necessary to remove blanks if present.
    return (biopdb_residue.full_id[2],
            "".join([str(ei) for ei in biopdb_residue.full_id[3] if str(ei).strip()]),
            three_to_one(biopdb_residue.resname.upper()))


def match_residue_triples(trip, query_trip, aa_from_location=2):
    """
    Logical comparing trip and query_trip

    :param trip: triple, (chain, residue_location, current_residue_value)
    :param query_trip: triple, (chain_to_change, residue_location_to_change,
        residue_to_change_from), where the last value may be None.
    :param aa_from_location: the location at which the aa symbol is given.
    :return: If the aa_from_location element in query_trip is not None, True if
        all entries equal; if last is None, True if first two elements equal

    >>> match_residue_triples(('A', 125, 'Q'), ('A', 125, 'Q'))
    True
    >>> match_residue_triples(('A', 125, 'Q'), ('B', 125, 'Q'))
    False
    >>> match_residue_triples(('B', 1, 'Q'), ('B', 1, None))
    True
    """

    truth = all(
        [t_i == q_i if (i != aa_from_location or q_i is not None) else True
         for i, t_i, q_i in zip(range(len(trip)), trip, query_trip)]
    )
    return truth


############################################
# Functions for manipulating pairs (lists) #
############################################

# ### After tuple-izing the residues: ###


def mutate_pairs_once(pairs, mutation):
    """
    Modify the sequence in pairs by applying the single mutation

    :param pairs: A list of pairs (tuples), where each element in the list is a
        pair of interacting residues, and each residue is represented as a
        triple, in which the elements are (chain, locus, residue_aa).
    :param mutation: A change, represented as a quadruple: the elements are
        (chain, locus, change_from_aa, change_to_aa). change_from_aa may be
        None.
    :return: pairs, with the modification applied

    >>> mutate_pairs_once([(('A', '125', 'E'), ('B', '1', 'Q'))], ('B', '1', 'Q', 'A'))
    [(('A', '125', 'E'), ('B', '1', 'A'))]
    >>> mutate_pairs_once([(('A', '125', 'E'), ('B', '1', 'Q')), (('A', '125', 'E'), ('B', '2', 'Y'))], ('A', '125', 'E', 'A'))
    [(('A', '125', 'A'), ('B', '1', 'Q')), (('A', '125', 'A'), ('B', '2', 'Y'))]
    """

    for i, pair_i in enumerate(pairs):
        # Find matches
        match = [match_residue_triples(pair_i_j, mutation[:3])
                 for pair_i_j in pair_i]

        # Check that there sum(match) in [0, 1]
        if sum(match) > 1:
            print("WARNING: More than one match; is a residue interacting with "
                  "itself?")

        # Substitute for the AA if a match, else, keep the original
        pairs[i] = tuple([pair_i_j[:-1] + (mutation[-1],) if m_ij else pair_i_j
                          for m_ij, pair_i_j in zip(match, pair_i)])

    return pairs


def mutate_pairs_multiple(pairs, mutations):
    """
    Apply a list of mutations to pairs

    :param pairs: a list of pairs of interacting residues
    :param mutations: a list of quadruples (or a single quadruple) giving
        changes in individual residues.
    :return: pairs: mutated version of the list of pairs.

    >>> mutate_pairs_multiple([(('A', '125', 'E'), ('B', '1', 'Q'))], [('B', '1', 'Q', 'A'), ('A', '125', 'E', 'F')])
    [(('A', '125', 'F'), ('B', '1', 'A'))]
    """

    if not isinstance(mutations, list):
        mutations = list(mutations)

    # Check to make sure that no two mutations apply to the same residue; this
    # would introduce an order dependency
    # TODO: Implement double mutation check
    for mi in mutations:
        pairs = mutate_pairs_once(pairs, mi)

    return pairs

# ### Before tuple-izing the residues: ###


def _clean_interacting_biopdb_residue_pairs(interacting_pairs):
    """
    Reduce the pairs list contents from BioPython-formatted residues to tuples

    :param interacting_pairs: list of tuples, where each element in a
        tuple is a BioPython-formatted residue.
    :return: list containing sorted, tuple-ized pairs, where each tuple is of
        the following form: (chainCode, residueLocation, singleLetterAACode)
    """

    interacting_pairs = sorted(
        [
            (tupleize_biopdb_residue(resi), tupleize_biopdb_residue(abi))
            for resi, abi in interacting_pairs
        ]
    )
    return interacting_pairs


def _residues_from_pairs_of_biopdb_residues(pairs):
    """
    Extract the list of unique, 1st partner residues from pairs

    :param pairs: list containing the interacting pairs of BioPDB.Residues
        (ResidueFrom1stPartner, ResidueFrom2ndPartner) from the interface
    :return: list containing the sorted, unique, tuple-ized residues from the
        1st partner that take part in at least one pair.
    """

    interest_residues = [
        tupleize_biopdb_residue(resi)
        for resi, _ in pairs
    ]
    interest_residues = sorted(list(set(interest_residues)))
    return interest_residues


# ####################
# Test functionality #
# ####################


def main(name, fname, antigen_chains, antibody_chains):
    """
    Use several methods in parallel to identify interacting residues

    :param name: str, giving structure name
    :param fname: str, giving path to .pdb file
    :param antigen_chains: The (mutable) chains from the structure
    :param antibody_chains: The (immutable) reference chains from the structure
    """

    inter_atomic_distance = 5.0  # Angstroms
    ca_sc_distance = 7.0  # Angstroms
    inter_ca_distance = 12.0  # Angstroms

    # ####################################
    # ### Inter-atomic distance method ###
    # ####################################
    # Uses both side chains
    pairs_sc_sc = \
        get_pairs_sc_sc(
            name, fname, antigen_chains, antibody_chains, inter_atomic_distance
        )
    ag_residues_sc_sc = _residues_from_pairs_of_biopdb_residues(pairs_sc_sc)
    pairs_sc_sc = _clean_interacting_biopdb_residue_pairs(pairs_sc_sc)

    print('The following AG residues (n={}) are found to interact by a {} '
          'Angstrom inter-atomic distance:'.format(len(ag_residues_sc_sc),
                                                   inter_atomic_distance))
    print(ag_residues_sc_sc)

    print('The interacting pairs (n={}) are:'.format(
        len(pairs_sc_sc)))
    print(pairs_sc_sc)

    # ######################################
    # ### ref-SC to chain-of-interest CA ###
    # ######################################
    # Uses reference chain side chains to determine interaction
    pairs_ca_sc = get_pairs_ca_sc(
        name, fname, antigen_chains, antibody_chains, ca_sc_distance)
    ag_residues_ca_sc = _residues_from_pairs_of_biopdb_residues(pairs_ca_sc)
    pairs_ca_sc = _clean_interacting_biopdb_residue_pairs(pairs_ca_sc)

    print('The following AG residues (n={}) are found to interact by a {} '
          'Angstrom ref side-chain-to-CA distance:'.format(len(ag_residues_ca_sc),
                                                           ca_sc_distance))
    print(ag_residues_ca_sc)

    print('The interacting pairs (n={}) are:'.format(
        len(pairs_ca_sc)))
    print(pairs_ca_sc)

    # ################################
    # ### Inter CA distance method ###
    # ################################
    # Uses neither side chain
    pairs_ca_ca = get_pairs_ca_ca(
        name, fname, antigen_chains, antibody_chains, inter_ca_distance)

    ag_residues_ca_ca = _residues_from_pairs_of_biopdb_residues(pairs_ca_ca)
    pairs_ca_ca = _clean_interacting_biopdb_residue_pairs(pairs_ca_ca)

    print('The following AG residues (n={}) are found to interact by a {} '
          'Angstrom inter-CA distance:'.format(len(ag_residues_ca_ca),
                                               inter_ca_distance))
    print(ag_residues_ca_ca)

    print('The interacting pairs (n={}) are:'.format(
        len(pairs_ca_ca)))
    print(pairs_ca_ca)


if __name__ == '__main__':
    # Doctesting:
    import doctest
    doctest.testmod()

    # args = sys.argv
    # # Some defaults for testing:
    # if len(args) == 1:
    #     args.append('4b3-fHBP')
    # if len(args) == 2:
    #     args.append('/Users/desautels2/Documents/vaccineDesign/'
    #                 'Data/Structures/'
    #                 'Fab4B3_fHbp.v1_toLLNL.PDB')
    # if len(args) == 3:
    #     args.append('D')
    # if len(args) == 4:
    #     args.append('H,L')
    #
    # # Invoke main
    # main(*args[1:])