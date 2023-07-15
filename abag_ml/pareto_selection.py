# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

"""
Module for the selection of pareto-optimal subsets of sequences

The following uses an epsilon-dominance definition like so, as given in
Hernández Castellanos et al., "Non-Epsilon Dominated Evolutionary Algorithm
for the Set of Approximate Solutions", 2020, Definition 2. With some changes in
notation, this definition is:

Definition 2:
Let e= (e1,. . .,ek) in R^k_+ and x,y in Q. x is said to e-dominate y
(written dom_e(x, y)) with respect to the scalar multi-objective optimization
problem MOP, for which minimization in each F_i is desirable, if F(x)−e <= F(y)
and F(x)−e != F(y).

That is, dom_e(x, y) iff F_i(x) - e_i <= F_i(y) for all i in {1, ..., k} AND
there exists i in {1, ..., k} s.t. F_i(x) - e_i < F_i(y). That is, x must
be at least ei better than y under the ith criterion, FOR ALL i, and there must
be at least 1 i such that x is more than ei better that y under the ith
criterion.

This can be translated to the context of this module by allowing non-zero
epsilon values for scalar functions (as above) and implementing special versions
of pairwise comparator functions for non-scalar scores.

If we compute:
not any([dom_i(b, a, -e_i) for dom_i, e_i in doms, eps]) AND
any([dom_i(a, b, e_i) for dom_i, e_i in doms, eps]);

the first term is
f_i(b) + e_i <= f_i(a) for all i  <--> f_i(a) - e_i >= f_i(b) for all i
while the second term is
there exists i such that f_i(a) - e_i > f_i(b).

These are the two requirements.  For e_i = 0 for all i, these requirements can
be checked all at once, but if e_i != 0, then two separate versions of the
computation must be executed.

Note that the two terms we want are again the transpose of the matrix we
normally compute (with -eps; allows checking if a_i + e_i >= b_i for all i) and
the negation (with positive eps; allows checking if a_i + e_i > b_i for some i).
"""

import scipy
# import dill
import pickle  # consider dill for broader pickle-ability
import pandas as pd
import numpy as np
import warnings
import multiprocessing as mp
from itertools import product, chain, repeat, combinations
import copy

from abag_agent_setup.expand_allowed_mutant_menu import AllowedMutationsKey, \
    derive_idstr_from_seq, _override_master_mutant  # Yes, it's supposed to be private, but we need it here.
from abag_agent_setup.mutant_generator_sampling import override_single_point_df, \
    generate_linear_mutants
from vaccine_advance_core.featurization.seq_to_features import diff_seqs, mutate_seq
from vaccine_advance_core.featurization.bio_sequence_manipulation import \
    recast_sequence_as_str

conventional_id_col = 'AntigenID'
conventional_seq_col = 'AntigenSequence'
conventional_mutations_col = 'Mutation'  # contains the text form of the mutation as tuples

# ################################
# ### Pareto-related functions ###
# ################################
def simple_row_scorer(row=None, column=None):
    if row is None:
        return True

    # else, compute the score
    return row[column]

def simple_negative_row_scorer(row=None, column=None):
    if row is None:
        return True

    # else, compute the score as -1 * the value.
    return -1 * row[column]


def chunker(df_tmp, bs, dom_func_tmp, scalar_epsilons_tmp):
    for rowstart_i in range(0, df_tmp.shape[0], bs):
        slice_end = min([rowstart_i + bs, df_tmp.shape[0]])
        yield df_tmp.iloc[rowstart_i:slice_end, :].copy(), \
              rowstart_i, \
              dom_func_tmp, \
              scalar_epsilons_tmp


def inner_pareto_rows_tuple(argstuple=None):
    """

    :param argstuple: four elements:
        0: the dataframe
        1: the integer-valued offset to the rows; the original row index is the
            row index within this dataframe chunk, plus the offset
        2: the dominance functions
        3: the epsilons
    :return:
    """
    try:
        _ = argstuple[2]
    except IndexError as e:
        print('Malformed argstuple detected!')
        print(argstuple)
        raise e

    return [ri + argstuple[1]
            for ri in get_pareto_rows(
                argstuple[0],
                dominance_functions=argstuple[2],
                scalar_epsilons=argstuple[3],
                returnints=True
            )]


def get_pareto_rows(df, dominance_functions=None, scalar_epsilons=None,
                    returnints=False):
    """
    For a dataframe, return indices of the pareto-optimal set of rows

    :param df: pandas DataFrame containing distinct entities (sequences) on
        each row, along with information about them that will be used to
        compare them.
    :param dominance_functions: list of functions that apply to rows (or
        pairs of rows) of the dataframe. Each function yields True/False if not
        given any arguments (does this function return a scalar description of
        the row); if a scalar description of a single row, when applied to the
        row, yields a score; if a pairwise comparison of rows a and b, yields a
        bool indicating if a dominates b.
    :param scalar_epsilons: None or list of float or None, giving a margin by
        which one scalar score must dominate another to be declared dominant;
        e.g., if scalar_epsilon=0.1, -0.15 dominates 0.0, but -0.05 does not.
    :param returnints: bool; if True, return the integer-valued row indices;
        else, return the dataframe indices.
    :return: the dataframe indices
    """

    if scalar_epsilons is None and isinstance(dominance_functions, list):
        # Initialize scalar_epsilons appropriately
        scalar_epsilons = \
            [0.0 if dfi() else None for dfi in dominance_functions]

    not_dominated_matrix_pos_eps = None
    for tfi, sei in zip(dominance_functions, scalar_epsilons):
        not_dominated_matrix_pos_eps = apply_dominance_function_naive(
            df,
            dominance_function=tfi,
            scalar_epsilon=sei,
            no_dominance_so_far_array=not_dominated_matrix_pos_eps
        )

    # TODO: implement switching off here for memory consumption
    if not all([si is None or si == 0.0 for si in scalar_epsilons]):
        not_dominated_matrix_neg_eps = None
        for tfi, sei in zip(dominance_functions, scalar_epsilons):
            not_dominated_matrix_neg_eps = apply_dominance_function_naive(
                df,
                dominance_function=tfi,
                scalar_epsilon=-sei,  # Note sign here
                no_dominance_so_far_array=not_dominated_matrix_neg_eps
            )
    else:
        not_dominated_matrix_neg_eps = not_dominated_matrix_pos_eps  # TODO: Copy?

    # print('The set of _not_ dominated relationships is:')
    # print(not_dominated_matrix)
    # print('The resulting dominance relationships are:')
    dominance_relationships = np.logical_and(
        np.logical_not(not_dominated_matrix_pos_eps), not_dominated_matrix_neg_eps.transpose())
    # print(dominance_relationships)
    # print('')
    # print('The set of non-dominated entities is:')
    non_dominated_by_row = np.logical_not(
        np.any(dominance_relationships, axis=0))  # array of bool
    # print(non_dominated_by_row)
    # print('')
    # print('And finally, the Pareto set of dataframe indices is:')
    if not returnints:
        indices_pareto_row_set = df.loc[non_dominated_by_row].index.tolist()
    else:
        indices_pareto_row_set = list(np.nonzero(non_dominated_by_row)[0])

    return indices_pareto_row_set


def apply_dominance_function_naive(df, dominance_function,
                                   no_dominance_so_far_array=None,
                                   scalar_epsilon=0.0, verbose=False):
    """
    Produce a matrix describing whether rows in df are non-dominated by each
    other row.

    Memory consumption is expected to be o(n^2), decreasing substantially with
    more criteria; run-time depends on sorting algorithm (scalar case) or is
    < O(n^2) for pairwise case, greatly decreased if sparsity can be achieved
    early.

    :param df: pd.DataFrame, where each row describes an entity.
    :param dominance_function: a function that applies to df, row-wise or
        pairwise over rows.  If row-wise, it must return a scalar, where
        smaller scalar values dominate; if pairwise, it must return a bool
        dominates(A,B) for two rows A, B.  When invoked without argument, gives
        either True (scalar) or False (pairwise).
    :param no_dominance_so_far_array:  np.array of 0 or 1; element i,j
        indicates \notexist k s.t. dominates_k(i,j) (that is, there is no
        criterion under which i dominates j). Note that the computation
        necessary to determine if i dominates j OVER SEVERAL CRITERIA is
        (not any([dom_k(j, i) for k]) AND (any([dom_k(i, j) for k]));
        the j, ith entry in this matrix is the first term and the negation of
        the i, jth entry is the second.
    :param scalar_epsilon: None or float, giving a margin by which one scalar
        score must dominate another to be declared dominant; e.g., if
        scalar_epsilon=0.1, -0.15 dominates 0.0, but -0.05 does not. Note that
        the comparison is > (not >=), such that -0.1 does NOT dominate 0.0 if
        scalar_epsilon is 0.1.
    :param verbose: bool, indicating if should be verbose in printout to stdout.
    :return: no_dominance_so_far_array, updated using dominance_function
    """

    is_scalar = dominance_function()
    if is_scalar and scalar_epsilon is None:
        scalar_epsilon = 0.0

    if is_scalar:
        # If dominance_function is a scalar, determine scores, determine the sorted
        # order (KEEPING ties), and use the sorted order to fill the dominance matrix

        scores = np.array([dominance_function(row_i) for idx_i, row_i in df.iterrows()])  # TODO: avoid iterrows
        order = np.argsort(scores, kind='quicksort')  # by default, ascending

        # Walk along the order, filling the array appropriately
        # not_dominance_this_criterion = \
        #     scipy.sparse.lil_matrix([df.shape[0], df.shape[0]], dtype=np.bool)
        # # Use the sorted values above to construct the list of lists of
        # # no_dominance_this_criterion elements, in the indexing of the original
        # # df.
        # not_dominance_this_criterion.rows = [[] for i in range(df.shape[0])]

        not_dominance_this_criterion = np.zeros([df.shape[0], df.shape[0]], dtype=bool)
        # warnings.warn('Implemented scalar comparison does not respect strict equality!')
        for i, vi in enumerate(order):
            # Determine which of the rows is the first dominated by
            # this element  (note that the possibility of epsilon being
            # negative means that we need to include all objects prior in the
            # ranking as well).
            pos_maybe_dominated = -1
            dominated = False
            # TODO: Version that's faster using some logical check & find; e.g.,
            #  positivity after subtracting current value and epsilon
            while not dominated:
                pos_maybe_dominated = pos_maybe_dominated + 1
                if pos_maybe_dominated == len(scores) or \
                        scores[order[pos_maybe_dominated]] > \
                        scores[order[i]] + scalar_epsilon:
                    dominated = True

            # Mark that this object fails to dominate everything PRIOR to the
            # first dominated element.
            not_dominance_this_criterion[vi, order[:pos_maybe_dominated]] = True

            # For verbosity:
            if verbose:
                if pos_maybe_dominated < len(order):
                    print(
                        'For position {}, row {} in the dataframe, with value '
                        '{}, determined that first dominated row in the '
                        'dataframe is at position {} in the order, row {}, and '
                        'has value {}'.format(
                            i, vi, scores[vi],
                            pos_maybe_dominated,
                            order[pos_maybe_dominated],
                            scores[order[pos_maybe_dominated]]
                        )
                    )
                else:
                    print(
                        'For position {}, row {} in the dataframe, with value '
                        '{}, no other rows are dominated!'.format(
                            i, vi, scores[vi]
                        )
                    )
                print(scores[order])
                print(not_dominance_this_criterion[vi, order])

    else:
        # print('Non-Scalar dominance function!')

        # If the dominance_function is only pairwise, compute pairwise values for
        # only those dominance relationships that are undetermined.

        not_dominance_this_criterion = np.zeros([df.shape[0], df.shape[0]],
                                                dtype=bool)
        # Above: Will become True if no dominance observed, False otherwise
        for i, j in product(range(df.shape[0]), range(df.shape[0])):
            if verbose:
                print(i, j)
                print(df.iloc[i])
                print(df.iloc[j])
                print(no_dominance_so_far_array is None \
                        or no_dominance_so_far_array[i, j])
                print(dominance_function(df.iloc[i, :], df.iloc[j, :]))
            if no_dominance_so_far_array is None \
                    or no_dominance_so_far_array[i, j]:
                if not dominance_function(df.iloc[i, :], df.iloc[j, :]):
                    # That is, if i does not dominate j...
                    not_dominance_this_criterion[i, j] = True
                    # mark that i does not dominate j
                # else: remains False

                # Note: we are ONLY FILLING THOSE CELLS WHERE THERE (A) there is
                # no dominance observed so far and (B) THERE IS NO DOMINANCE
                # OBSERVED UNDER THIS CRITERION. This is because we're going to
                # take an "AND" below.  A more memory efficient algorithm would
                # be to element-wise update the entries in
                # no_dominance_so_far_array, thus avoiding instantiating another
                # nxn array.

    # Now that we've computed the relationship of dominances, update
    # 1 if so far = 1 and (not dominance_this) = 1; 0 otherwise
    if no_dominance_so_far_array is None:
        # This is the first iteration
        no_dominance_so_far_array = not_dominance_this_criterion
    else:
        no_dominance_so_far_array = np.logical_and(
            no_dominance_so_far_array,
            not_dominance_this_criterion
        )

    return no_dominance_so_far_array


# ####################################################################
# ### Functions for use after Pareto selection, allowing reshaping ###
# ### of the selected set                                          ###
# ####################################################################

# ### Downsampling ###



# ### Up-sampling  ###
def add_rows_from_seq_list(df, colname, seq_list, mutation_list_of_lists=None, add_absent_sequences=True, verbose=False):
    """
    Find (or optionally, add) sequences from seq_list in the dataframe and return marked version.
    """

    df_tmp = df.copy()
    for i, seq_i in enumerate(seq_list):

        # Compute the ID
        seq_id_i = derive_idstr_from_seq(seq_i)

        # If present, select the ID
        hits = df_tmp[conventional_id_col] == seq_id_i
        if any(hits):
            df_tmp.loc[hits, colname] = True
        elif add_absent_sequences:
            seq_dict = {}
            # Add the sequence and ID to appropriate columns
            seq_dict[conventional_id_col] = [seq_id_i]
            seq_dict[conventional_seq_col] = [seq_i]
            if mutation_list_of_lists:
                # seq_dict[conventional_mutations_col] = ",".join([str(mj) for mj in mutation_list_of_lists[i]])  # TODO: revise chain handling?
                seq_dict[conventional_mutations_col] = ",".join(['(' + ','.join([str(mij) for mij in mj]) + ')' for mj in mutation_list_of_lists[i]])  # TODO: revise chain handling?

            # Mark preselection
            seq_dict[colname] = [True]
            # Add row to the dataframe with this sequence
            # The df_tmp index should not be ignored. However, the indices are all ints and we need
            # to know what indices are in the dataframe. Take the max of these integer values and add one.
            # TODO: Evaluate whether this allows for a possible collision later if we are looking at a df
            #       that is a chunk of something else.            
            df_tmp = df_tmp.append(pd.DataFrame(copy.deepcopy(seq_dict), index=[df_tmp.index.max() + 1])) 

        else:
            if verbose and mutation_list_of_lists:
                print('Unable to find sequence with pattern {}!'.format(mutation_list_of_lists[i]))
            elif verbose:
                print('Unable to find sequence with id {}!'.format(seq_id_i))
    return df_tmp



def ablate_seqs_for_upsample(df, master_seq=None, colname=None, add_absent_ablations=True, verbose=False):
    """
    Ablate and select ablated versions of sequences of interest.

    Closely modeled on preselect_seq_ablations from draft_day_preselection.

    :param df: incoming DataFrame, with columns AntigenSequence and Mutation
    :param master_seq: str, giving the amino acid sequence of the master protein.
    :param colname: str, giving column name to use to determine which sequences to 
        ablate and also wherein True values will be given for proposed ablations or 
        identified ablations in the data set. 
    :param add_absent_ablations: bool, default True. Whether or not to add ablations not 
        found in the df.
    :param verbose: bool, default False. Whether or not to print information on number of 
        ablations added per starting sequence.
    :return df_tmp: original Df with (potentially) additional rows
    """
    df_tmp = df.copy()
    nstartingrows = df_tmp.shape[0]

    if master_seq is None:
        raise ValueError('Master seq must be supplied as a protein sequence in string form')

    if colname is None and 'Selected' in df_tmp.columns:
        colname = 'Selected'
    elif colname is None:
        raise ValueError('colname must be specified if the default value, "Selected" is '
        'not in the column set of df.')

    # for each row, if selected, construct ablations, check if present, if yes, mark 
    # selected, if not, create and mark selected
    frozen_index = list(df_tmp.index)
    frozen_selections = list(df_tmp[colname])
    for ii, fsi in zip(frozen_index, frozen_selections):
        if not fsi:
            continue
        
        start_rows = df_tmp.shape[0]
        mut_id = derive_idstr_from_seq(df_tmp.loc[ii][conventional_seq_col])

        # Get the sequence and diff it from the master
        mutation_list = diff_seqs(df_tmp.loc[ii][conventional_seq_col], master_seq)

        if len(mutation_list) == 1:  # There's only one mutation; the ablation is the WT
            if verbose:
                print('Added 0 rows for ablations of {}; singlepoint.'.format(mut_id))
            continue

        # Create the list of leave-one-out sequences
        mutation_list_of_lists = [
            mutation_list[:j] + mutation_list[j+1:] for j in range(len(mutation_list))
            ]
        seq_list = [
            mutate_seq(
                recast_sequence_as_str(master_seq),
                mutation_list_j
            ) 
            for mutation_list_j in mutation_list_of_lists
        ]

        df_tmp = add_rows_from_seq_list(
            df_tmp, colname, seq_list, 
            mutation_list_of_lists=mutation_list_of_lists, 
            add_absent_sequences=add_absent_ablations, 
            verbose=verbose
        )

        if verbose:
            print('Added {} rows for ablations of {}.'.format(df_tmp.shape[0] - start_rows, mut_id))
    print('Added {} rows in total to the dataframe via ablations.'.format(df_tmp.shape[0] - nstartingrows))
    return df_tmp


# def ablate_seqs_for_upsample_par(df, master_seq=None, colname=None, add_absent_ablations=True, verbose=False, nprocs=None):
#     """
#     Ablate and select ablated versions of sequences of interest.

#     Closely modeled on preselect_seq_ablations from draft_day_preselection.

#     :param df: incoming DataFrame, with columns AntigenSequence and Mutation
#     :param master_seq: str, giving the amino acid sequence of the master protein.
#     :param colname: str, giving column name to use to determine which sequences to 
#         ablate and also wherein True values will be given for proposed ablations or 
#         identified ablations in the data set. 
#     :param add_absent_ablations: bool, default True. Whether or not to add ablations not 
#         found in the df.
#     :param verbose: bool, default False. Whether or not to print information on number of 
#         ablations added per starting sequence.
#     :param nprocs: int or None, number of cpu to use in pool; if None, defaults
#         to max(1, system CPUs - 1)
#     :return df_tmp: original Df with (potentially) additional rows
#     """

#     if nprocs is None:
#         nprocs = mp.cpu_count() - 1
#     nprocs = max(1, min(nprocs, mp.cpu_count() - 1))

#     df_tmp = df.copy()
#     nstartingrows = df_tmp.shape[0]

#     if master_seq is None:
#         raise ValueError('Master seq must be supplied as a protein sequence in string form')

#     if colname is None and 'Selected' in df_tmp.columns:
#         colname = 'Selected'
#     elif colname is None:
#         raise ValueError('colname must be specified if the default value, "Selected" is '
#         'not in the column set of df.')

#     # for each row, if selected, construct ablations, check if present, if yes, mark 
#     # selected, if not, create and mark selected
#     frozen_index = list(df_tmp.index)
#     frozen_selections = list(df_tmp[colname])

#     # Main loop in parallel:
#     with mp.Pool(processes=nprocs) as pool:
#         # df_tmp, master_seq, frozen_index, frozen_selections must be passed; data parallel in frozen_index, frozen_selections
#         seq_list_of_lists, mutation_lolol = pool.map(

#         )
#         # Operations:
#         # mutation_list = diff_seqs(df_tmp.loc[ii][conventional_seq_col], master_seq)
#         # if len(mutation_list) == 0: continue
#         # 
#         # mutation_list_of_lists = [
#         #     mutation_list[:j] + mutation_list[j+1:] for j in range(len(mutation_list))  # Use itertools.combinations?
#         #     ]
#         # seq_list = [
#         #     mutate_seq(
#         #         recast_sequence_as_str(master_seq),
#         #         mutation_list_j
#         #     ) 
#         #     for mutation_list_j in mutation_list_of_lists
#         # ]


#     seq_list = list(chain.from_iterable(seq_list_of_lists))  # flatten
#     mutation_list_of_lists = list(chain.from_iterable(mutation_lolol)) 
#     df_tmp = add_rows_from_seq_list(
#         df_tmp, colname, seq_list, 
#         mutation_list_of_lists=mutation_list_of_lists, 
#         add_absent_sequences=add_absent_ablations, 
#         verbose=verbose
#     )
#     print('Added {} rows in total to the dataframe via ablations.'.format(df_tmp.shape[0] - nstartingrows))
#     return df_tmp


def upsample_with_lmg(df, master_seq=None, allowed_mutations=None, sampling_frequencies=None, colname=None, num_mutations_to_add=3, verbose=False):
    """
    Create mutated versions of sequences of interest, where mutations come from an LMG-formatted file.

    :param df: incoming DataFrame, with columns AntigenSequence and Mutation
    :param colname: str, giving column name to use to determine which sequences to 
        mutate and also wherein True values will be given for proposed ablations or 
        identified ablations in the data set. 
    :return df: original Df with (potentially) additional rows
    """

    if master_seq is None:
        raise ValueError('Master seq must be supplied as a protein sequence in string form')

    if allowed_mutations is None:
        raise ValueError('allowed_mutations must be supplied as FILL IN HOW')  # TODO: Return and fill this.

    if sampling_frequencies is None:
        raise ValueError('sampling_frequencies must be supplied as a dataframe')


    df_tmp = df.copy()
    nstartingrows = df_tmp.shape[0]

    if colname is None and 'Selected' in df_tmp.columns:
        colname = 'Selected'
    elif colname is None:
        raise ValueError('colname must be specified if the default value, "Selected" is '
        'not in the column set of df.')

    # for each row, if selected, construct up-sampled mutations, check if present, if yes, mark 
    # selected, if not, create and mark selected
    frozen_index = list(df_tmp.index)
    frozen_selections = [True if fsi is True else False for fsi in list(df_tmp[colname])]  # If filled with NaNs from prior expansion of rows
    print('Upsampling from {} selected sequences.'.format(df_tmp[colname].sum()))
    for ii, fsi in zip(frozen_index, frozen_selections):
        if not fsi:
            continue
        
        start_rows = df_tmp.shape[0]
        mut_id = derive_idstr_from_seq(df_tmp.loc[ii][conventional_seq_col])

        # pass down the information to the generator 
        # # In expand_allowed_mutant_menu.expand_menu_study_params_master_antigen_structures_mutations,
        # # Sampling is by the following series of three steps, where arguments have been localized:
        # # Need: master_seq, allowed_mutations (derived from menu)

        altered_singlepoint_df = \
            override_single_point_df(
                copy.deepcopy(master_seq),
                copy.deepcopy(allowed_mutations),
                singlePointMutationDataWithSampleWeights=sampling_frequencies,
                override_mutant_sequence=df_tmp.loc[ii][conventional_seq_col]
            )
        # if the kwarg override_mutant_sequence is present, use it to override 
        overriden_master_sequence, overridden_allowed_mutations = _override_master_mutant(
            copy.deepcopy(master_seq),
            copy.deepcopy(allowed_mutations),
            override_mutant_sequence=df_tmp.loc[ii][conventional_seq_col]
        )
        mutant_sequences = generate_linear_mutants(
            overriden_master_sequence,
            overridden_allowed_mutations,
            numberMutantToGenerate=num_mutations_to_add,
            singlePointMutationDataWithSampleWeights=altered_singlepoint_df,
            minNumLocationsToMutate=1,
            maxNumLocationsToMutate=1
        )
        print('Generated {} mutations from sequence {}.'.format(len(mutant_sequences), mut_id))
        # TODO: Debug # of mutations added; disagrees with later check on total 
        #       number of mutations made. 
        mutation_list_of_lists = [
            diff_seqs(mutant_seq_i, master_seq)
            for mutant_seq_i in mutant_sequences
        ]

        # For each, check if exists; if yes, mark, if no, add to dataframe
        # (if verboase,) Print information
        # Re-use code from above
        # print(ii)
        # print(isinstance(df_tmp.index, pd.RangeIndex))
        pre_addition_size = df_tmp.shape[0]
        df_tmp = add_rows_from_seq_list(
            df_tmp, colname, mutant_sequences, 
            mutation_list_of_lists=mutation_list_of_lists, 
            add_absent_sequences=True, 
            verbose=verbose
        )
        # print(isinstance(df_tmp.index, pd.RangeIndex))
        
        if verbose:
            print('Added {} of {} proposed mutations.'.format(df_tmp.shape[0] - pre_addition_size, len(mutant_sequences)))
            print('Added {} rows for LMG modifications of {}.'.format(df_tmp.shape[0] - start_rows, mut_id))
    print('Added {} rows in total to the dataframe via LMG modifications of selected sequeces.'.format(df_tmp.shape[0] - nstartingrows))

    # Return Dataframe
    return df_tmp


def main(df, dominance_functions=None, scalar_epsilons=None, blocksize=1000):
    """
    Compute Pareto optimal set of rows of a dataframe.

    :param df: pandas DataFrame, from which the Pareto-optimal set of rows is
        to be computed.
    :param dominance_functions: list of row-acting functions to be applied to
        df
    :param blocksize: int, maximum block size to be used in first sweep over
        dataframe.  Note that memory consumption scales with blocksize ** 2.
    :return: list of int giving the Pareto optimal rows of df.
    """

    # Note that, because the union of Pareto-optimal row sets of disjoint blocks
    # of rows of the original dataframe is a superset of the Pareto-optimal row
    # set of the whole dataframe, it is possible to chunk the dataframe into
    # smaller components, compute their respective Pareto-optimal sets, and
    # then compute the Pareto-optimal subset of the union of these sets. Doing
    # so will allow greater memory efficiency (and presumably reasonable speed)
    # while still returning the correct result.

    assert isinstance(blocksize, int), 'Blocksize must be an int!'

    if scalar_epsilons is None and isinstance(dominance_functions, list):
        # Initialize scalar_epsilons appropriately
        scalar_epsilons = [0.0 if dfi() else None for dfi in dominance_functions]

    blocksize = max(2, min([blocksize, df.shape[0]]))
    # Above: Make sure the blocksize is <= # rows, at least 2 (which is still
    # silly small, but not 100% useless.
    nblocks = df.shape[0] // blocksize

    # If there's just one block, return directly
    if nblocks == 1:
        return get_pareto_rows(
            df,
            dominance_functions=dominance_functions,
            scalar_epsilons=scalar_epsilons,
            returnints=True
        )

    # If more than one block, compute individually and unify.
    # Compute rows_prelim for each block
    # Prefer to use contiguous blocks, on the intutition that the dataframe
    # might be (locally) sorted.
    rows_prelim = []
    for rowstart_i in range(0, df.shape[0], blocksize):
        slice_end = min([rowstart_i + blocksize, df.shape[0]]) # SLICING end
        # Chunk
        print('Computing pareto optimal set for rows '
              '{} to {} ...'.format(rowstart_i, slice_end - 1)
              )

        # Add the set of pareto-optimal rows in this chunk of dataframe, in the
        # integer row indices of the original dataframe
        # Probably not necessary to do this in integer indexing (covers non-
        # unique index case) but may as well.

        rows_prelim.append(
            [ri + rowstart_i for ri in
             get_pareto_rows(
                 df.iloc[rowstart_i:slice_end, :],  # Chunk
                 dominance_functions=dominance_functions,
                 scalar_epsilons=scalar_epsilons,
                 returnints=True
             )
             ]
        )
        print(
            '... found {} of {} locally Pareto optimal rows from range '
            '{} to {}.'.format(len(rows_prelim[-1]), slice_end - rowstart_i,
                               rowstart_i, slice_end - 1)
        )

    rows_prelim = list(chain.from_iterable(rows_prelim))  # flatten

    # Create conversion dictionary:
    conversion = {i: rpi for i, rpi in enumerate(rows_prelim)}

    print('Computing pareto optimal set from first-pass row set of '
          '{} rows of {} total...'.format(len(rows_prelim), df.shape[0])
          )
    best_of_best_rows = get_pareto_rows(
        df.iloc[rows_prelim, :],
        dominance_functions=dominance_functions,
        scalar_epsilons=scalar_epsilons,
        returnints=True
    )
    print(
        '... done; found {} globally Pareto optimal rows.'.format(
        len(best_of_best_rows)
        )
    )

    return [conversion[bbri] for bbri in best_of_best_rows]


def main_par(df, dominance_functions=None, scalar_epsilons=None,
             blocksize=1000, nprocs=None, depth=0, max_depth=2, verbose=False):
    """
    Compute Pareto optimal set of rows of a dataframe.

    :param df: pandas DataFrame, from which the Pareto-optimal set of rows is
        to be computed.
    :param dominance_functions: list of row-acting functions to be applied to
        df. Critically, must be pickle-able.
    :param blocksize: int, maximum block size to be used in first sweep over
        dataframe.  Note that memory consumption scales with blocksize ** 2.
    :param nprocs: int or None, number of cpu to use in pool; if None, defaults
        to max(1, system CPUs - 1)
    :param depth: int, recursion depth into this process for the consolidation.
    :param max_depth: int, MAXIMUM recursion depth into this process for 
        consolidation. if depth == max_depth, no further splitting will be 
        used.  
    :param verbose: bool, print or not.
    :return: list of int giving the Pareto optimal rows of df.
    """

    assert isinstance(blocksize, int), 'Blocksize must be an int!'

    for i, dfi in enumerate(dominance_functions):
        try:
            _ = pickle.dumps(dfi)
        except pickle.PicklingError as e:
            print('Unable to pickle dominance_function with index {}, value '
                  '{}!'.format(i, dfi))

    if scalar_epsilons is None and isinstance(dominance_functions, list):
        # Initialize scalar_epsilons appropriately
        scalar_epsilons = [0.0 if dfi() else None for dfi in dominance_functions]

    blocksize = max(2, min([blocksize, df.shape[0]]))
    # Above: Make sure the blocksize is <= # rows, at least 2 (which is still
    # silly small, but not 100% useless.
    nblocks = df.shape[0] // blocksize

    # If there's just one block, return directly
    if nblocks == 1:
        return get_pareto_rows(
            df,
            dominance_functions=dominance_functions,
            scalar_epsilons=scalar_epsilons,
            returnints=True
        )

    if nprocs is None:
        nprocs = mp.cpu_count() - 1
    nprocs = max(1, min(nprocs, mp.cpu_count() - 1))

    # If more than one block, compute individually and unify.
    # Compute rows_prelim for each block
    # Prefer to use contiguous blocks, on the intutition that the dataframe
    # might be (locally) sorted.

    print('Executing first pass optimization on {} processes...'.format(nprocs))
    with mp.Pool(processes=nprocs) as pool:
        rows_prelim = pool.map(
            inner_pareto_rows_tuple,
            chunker(df, blocksize, dominance_functions, scalar_epsilons)
        )
    print('... done with first pass Pareto optimization using {} '
          'processes.'.format(nprocs))

    rows_prelim = list(chain.from_iterable(rows_prelim))  # flatten

    # TODO: on the face of it, this should be fine... Check.
    rows_prelim = list(np.random.permutation(rows_prelim))
    
    # Create conversion dictionary:
    conversion = {i: rpi for i, rpi in enumerate(rows_prelim)}

    print('Computing pareto optimal set from first-pass row set of '
          '{} rows of {} total...'.format(len(rows_prelim), df.shape[0])
          )
    
    if depth == max_depth or blocksize >= len(rows_prelim):
        # Don't go deeper than this; you're most likely wasting time
        best_of_best_rows = get_pareto_rows(
            df.iloc[rows_prelim, :],
            dominance_functions=dominance_functions,
            scalar_epsilons=scalar_epsilons,
            returnints=True
        )
    else:
        # some recursion is appropriate
        print('Executing recursive call to pareto_selection.main_par...')
        best_of_best_rows = main_par(
            df.iloc[rows_prelim, :],
            dominance_functions=dominance_functions,
            scalar_epsilons=scalar_epsilons,
            blocksize=blocksize,
            nprocs=nprocs,
            depth=(depth + 1),
            max_depth=max_depth  # ,
            # verbose=False
        )
    print(
        '... done; found {} globally Pareto optimal rows.'.format(
            len(best_of_best_rows)
        )
    )

    return [conversion[bbri] for bbri in best_of_best_rows]


def main_par_allblocks(df, dominance_functions=None, scalar_epsilons=None,
             blocksize=1000, nprocs=None, verbose=False):
    """
    Compute Pareto optimal set of rows of a dataframe.

    :param df: pandas DataFrame, from which the Pareto-optimal set of rows is
        to be computed.
    :param dominance_functions: list of row-acting functions to be applied to
        df. Critically, must be pickle-able.
    :param blocksize: int, maximum block size to be used in first sweep over
        dataframe.  Note that memory consumption scales with blocksize ** 2.
    :param nprocs: int or None, number of cpu to use in pool; if None, defaults
        to max(1, system CPUs - 1)
    :param verbose: bool, print or not.
    :return: list of int giving the Pareto optimal rows of df.
    """

    assert isinstance(blocksize, int), 'Blocksize must be an int!'

    for i, dfi in enumerate(dominance_functions):
        try:
            _ = pickle.dumps(dfi)
        except pickle.PicklingError as e:
            print('Unable to pickle dominance_function with index {}, value '
                  '{}!'.format(i, dfi))

    if scalar_epsilons is None and isinstance(dominance_functions, list):
        # Initialize scalar_epsilons appropriately
        scalar_epsilons = [0.0 if dfi() else None for dfi in dominance_functions]

    blocksize = max(2, min([blocksize, df.shape[0]]))
    # Above: Make sure the blocksize is <= # rows, at least 2 (which is still
    # silly small, but not 100% useless.
    nblocks = df.shape[0] // blocksize

    # If there's just one block, return directly
    if nblocks == 1:
        return get_pareto_rows(
            df,
            dominance_functions=dominance_functions,
            scalar_epsilons=scalar_epsilons,
            returnints=True
        )

    if nprocs is None:
        nprocs = mp.cpu_count() - 1
    nprocs = max(1, min(nprocs, mp.cpu_count() - 1))

    # For nblocks iterations, run comparisons between two blocks, returning the
    # _two_ lists of pareto optimal rows.
    with mp.Pool(processes=nprocs) as pool:
        print('Executing blockwise Pareto optimization on {} processes...'.format(
            nprocs))

        # As chunker does, create a set of rows, here as a list
        rows = [
            list(range(rowstart_i, min([rowstart_i + blocksize, df.shape[0]])))
            for rowstart_i in range(0, df.shape[0], blocksize)
        ]
        # For the 0th iteration, call on individual blocks
        print('Starting 0th iteration (self-comparisons) of blocks; starting '
              'with {} rows.'.format(df.shape[0]))
        rows = pool.map(
            inner_pareto_rows_tuple,
            [(df.iloc[rows_i, :], rows_i[0], dominance_functions, scalar_epsilons) for rows_i in rows]
        )

        # Instead of flattening, however, we will now call a pairwise operation
        # on each pair of blocks
        # Ultimately, this is n (above) + nChoose2, but we're eliminating rows as we go
        # and hopefully making it cheaper

        # older: sort of an upper triangular approach
        # for plus_interval in range(1, len(rows)):

        # Newer: Do upper-triangular matrix via wrapping around
        for plus_interval in range(1, 1 + len(rows) // 2 ):
            # Above: go from next neighbor to full length of the set.
            print(
                'Starting iteration {} of pairwise block comparisons; starting '
                'with {} rows.'.format(plus_interval, len(list(chain.from_iterable(rows)))))
            # Old: only positive differences
            # blockpairs = [(i, i+plus_interval) for i in range(0, len(rows) - plus_interval)]

            # Newer: allow wrap-around; reduces # iterations
            # Remove any redundant blockpairs
            blockpairs = list(set([
                tuple(sorted((i, (i+plus_interval) % len(rows))))
                for i in range(0, len(rows))
            ]))


            # Generate renumbering dictionaries for each chunk as required
            list_of_renumbering_dictionaries = [
                {k: r_bp_jk
                 for k, r_bp_jk in enumerate(rows[bp_j[0]] + rows[bp_j[1]])
                 }
                for bp_j in blockpairs
            ]

            # Slight inefficiency if one of the blocks becomes empty
            raw_survivors = pool.map(
                inner_pareto_rows_tuple,
                [(df.iloc[rows[b0] + rows[b1], :], 0,
                  dominance_functions, scalar_epsilons)
                 for b0, b1 in blockpairs]
            )
            # Note that each individual block may appear in TWO comparisons above
            # FOR DEBUGGING ONLY
            # print(raw_survivors)

            # Renumber
            renumbered_survivors = [
                [lord_i[raw_ij] for raw_ij in raw_i]
                for raw_i, lord_i
                in zip(raw_survivors, list_of_renumbering_dictionaries)
            ]

            # Finally, update rows
            for i in range(len(rows)):
                # if i appears in no blockpairs this iteration (which is
                # possible), continue
                if not any([i in bp_j for bp_j in blockpairs]):
                    continue

                # Pick out those renumbered survivors that come from ith block
                relevant_survivors = [
                    [rjk for rjk in renumbered_survivors_j if rjk in rows[i]]
                    for renumbered_survivors_j, bp_j
                    in zip(renumbered_survivors, blockpairs)
                    if i in bp_j
                ]

                # Now, compute the appropriate set intersection:
                # all such subsets
                # try:
                    # DEBUGGING ONLY
                    # print(relevant_survivors)
                rows[i] = list(set.intersection(
                    *[set(rsj) for rsj in relevant_survivors]
                ))
                # except TypeError:
                #     rows[i] = []

    # Flatten
    rows = list(chain.from_iterable(rows))
    print('Finished with blockwise comparisons; Pareto set is {} of '
          '{} rows.'.format(len(rows), df.shape[0]))

    return rows


if __name__ == '__main__':
    pass
