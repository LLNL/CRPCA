# Copyright (c) 2018-2023, Lawrence Livermore National Security, LLC 
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: MIT

from __future__ import absolute_import, division, print_function
from vaccine_advance_core.featurization.seq_dist import \
    compute_distance_matrix_from_subs_mat, compute_distance_to_ref_from_subs_mat

from vaccine_advance_core.featurization.vaccine_advance_core_io import \
    fasta_from_list_of_seqrecords
from vaccine_advance_core.featurization.seq_to_features import mutate_seq, diff_seqs
from vaccine_advance_core.featurization.bio_sequence_manipulation import \
    write_description_for_seqrecord, recast_as_seqrecord, recast_sequence_as_str


import pandas as pd
from Bio import SeqIO
import numpy as np
import sys
import pickle
import hashlib
import random


def override_single_point_df(master_sequence, allowed_mutations, **kwargs):
    """
    make a deep copy of the df to return if 'singlePointMutationDataWithSampleWeights' in kwargs.keys():
    
    :param master_sequence:
    :param allowed_mutations:
    :param kwargs: remaining kwargs
    :return: new dataframe with original sequence updated.
    """

    verbose = False
    if verbose in kwargs.keys():
        verbose = kwargs['verbose']

    single_point_mutation_df = kwargs['singlePointMutationDataWithSampleWeights'].copy()
    override_mutant_sequence = kwargs["override_mutant_sequence"]
    diffseq = diff_seqs(override_mutant_sequence, master_sequence)
    
    if verbose: 
        print(single_point_mutation_df)

    for subsitution_tuple in diffseq:
        new_aa = subsitution_tuple[3]
        location = int(subsitution_tuple[1])

        #TODO: find a list comprehension way to do this such as:
        # single_point_mutation_df.iloc[single_point_mutation_df['location'] == location, 'originalAA'] = new_aa
        
        for i, row in single_point_mutation_df.iterrows():
            if row['location'] == location:
                single_point_mutation_df.at[i,'originalAA'] = new_aa
                single_point_mutation_df.at[i,'mutation4Tuple'] = \
                    "('{0}')".format("', '".join([row['chain'], str(location), new_aa, row['AA']]))  # string from CSV file in form "('A', '250', 'G', 'V')"

    return single_point_mutation_df


##  nnumber of mutations possible must be >= maxNumLocationsToMutate
# to use for a different antibody change "from someM396data import . . ."  to above to import from other file
## can't get "from abag_agent_setup.expand_allowed_mutant_menu import derive_idstr_from_seq" to work right now


def generate_linear_mutants(MasterAntigenSequence, AllowedMutations, singlePointMutationDataWithSampleWeights=None, numberMutantToGenerate=None, minNumLocationsToMutate=None, maxNumLocationsToMutate=None, **kwargs):
    #input looks like this:
    # 'MasterAntigenSequence': 'QVQLQQSGAEVKKPGSSVKVSCKASGGTFSSYTISWVRQAPGQGLEWMGGITPILGIANYAQKFQGRVTITTDESTSTAYMELSSLRSEDTAVYYCARDTVMGGMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTSALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTSPLFVHHHHHHGDYKDDDDKGSYELTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDSSSDYVFGTGTKVTVLGQPKANPTVTLFPPSSEEFQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS'
    #"AllowedMutations": [[31, 'S', ['D', 'F', 'M', 'S']], [47, 'W', ['D', 'L', 'Q', 'T', 'V', 'W']]]
    #samplingWeights = dataFrame {('A', '126', 'D', 'A') : 0.5 , ('A', '126', 'D', 'F') : 0.01 }       (chain, residueNumner, from AA, toAA) : samplingWeight   passed via  **kwargs
    #output is a list of full length sequences (strings)

    verbose = False
    if verbose in kwargs.keys():
        verbose = kwargs['verbose']

    maxNumberSamplesToTryPerMutant = 200  # to prevent infinite while loop

    singlePointMutationDataWithSampleWeights["mutation"] = singlePointMutationDataWithSampleWeights.apply(lambda x :   (x['chain'], str(int(x["location"])), x["originalAA"], x["AA"]  ), axis=1)    #make mutation 4 tuple
    singlePointMutationDataWithSampleWeights["location"] = singlePointMutationDataWithSampleWeights.apply(lambda x :   str(int(x["location"])), axis=1)    #make mutation 4 tuple
    singlePointMutationDataWithSampleWeights['mutationHumanReadable'] = singlePointMutationDataWithSampleWeights.apply(lambda x :  x["originalAA"] +  str(int(x["location"])) +  x["AA"]  , axis=1)   #singlePointMutationDataWithSampleWeights["mutation"].apply(lambda x : x[2] + x[1] + x[3])
    singlePointMutationDataWithSampleWeights = singlePointMutationDataWithSampleWeights[["mutation", "samplingWeight", "location", "mutationHumanReadable"]]

    data = singlePointMutationDataWithSampleWeights

    allowedMutationsHumanReadable = set([])
    for location in AllowedMutations:
        for mutation in location[2]:
            allowedMutationsHumanReadable.add(location[1] + str(location[0])+mutation)

    #onlny allow mutations spewcific in AllowedMutations argument
    if verbose:
        print("allowedMutationsHumanReadable")
        print(sorted(allowedMutationsHumanReadable))
        print("data['mutationHumanReadable']")
        [print(x) for x in data['mutationHumanReadable']]

    data = data.loc[data['mutationHumanReadable'].isin(allowedMutationsHumanReadable)]
    if verbose: 
        print("intersection")
        [print(x) for x in data['mutationHumanReadable']]
    #Residue numbers On an antibody being considered for a  mutation
    locations = set(data['location'])

    assert len(locations) >= maxNumLocationsToMutate, 'maxNumLocationsToMutate ({}) is greater than number locations allowed/available mutate ({}).'.format(maxNumLocationsToMutate, len(locations))

    combinations = []
    #print("using weightedSamplingMethod")
    #for each 4Tuple, there is a weight
    for i in range(numberMutantToGenerate):
        #sample number of locations to mutate
        numLocationsToMutate = np.random.randint(minNumLocationsToMutate,maxNumLocationsToMutate +1 ,1)  # TODO: use Conway–Maxwell–Poisson
        mutations = set()
        loop_count = 0
        while len(mutations) < numLocationsToMutate:
            loop_count = loop_count + 1
            mutation = random.choices(list(data["mutation"]), weights=list(data["samplingWeight"]), k=1)
            if (not mutation[0][1] in [x[1] for x in mutations]) and (mutation[0][2] != mutation[0][3]):
                mutations.add(mutation[0])
            if loop_count > maxNumberSamplesToTryPerMutant:
                # TODO: remove ValueError here somehow
                raise ValueError('entered infinite while loop in sampling to create a mutant.')
                break
        combinations.append(list(mutations))

    linearMutants = pd.DataFrame(
        {'mutations': combinations,
         #'sumOfPointwiseDDGs': mutantDDGs,
         })
    # pd.set_option('display.max_colwidth', None)
    # print(linearMutants)
    # print(MasterAntigenSequence[223])
    linearMutants['FullSequence'] = linearMutants.apply(
        lambda row: mutate_seq(
            MasterAntigenSequence, row["mutations"],
            renumbering_dict=None
        ), axis=1
    )
    # remove duplicates
    mutants = list(set(linearMutants['FullSequence'].tolist()))
    # print(mutants)

    return mutants


if __name__ == '__main__':
    pass
