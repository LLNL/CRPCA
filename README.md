SPDX-License-Identifier: MIT

## Summary

All code and information contained here are selected and provided for the 
exclusive purpose of supporting the manuscript, "Computationally 
restoring the potency of a clinical antibody against Omicron."

This code is for reference purposes only and does not represent a functional 
version of the GUIDE platform, nor will it be maintained.

## Code Structure and Capabilities

The files in this archive are as follows (!!! denotes a file of particular interest):
```python
abag_ml:
    scripts:
        pareto_select_v2130_omicron_vanderbilt_larger.py  
            # !!! At end of design process, select sequences balancing !!!
            # !!! desirable attributes                                 !!!
        gpytorch_decision_script.py  # !!! Online, select among candidate studies to run. !!!
        pareto_selection.py  # At the end of the design process, find the Pareto set of sequences
        models_gpytorch.py # defines GP model for **production, online decision making**
        MultitaskVanillaGPModelDKL.py # defines GP model fit / predict for **training regime**
        GPBase.py  # base model class for **training** GP models, class definition for core features
        db_pull.py  # experiment to assemble dataset (pull from DB) and train GP model 
        experimenter.py  # training pipeline to train GP model
            - prepares dataset 
            - runs model.fit and model.predict
            - compute performance metrics 
            - save model and data 
        DBDataset.py  # dataset class. Pulls data from DB or read off file if it exists
        DatasetTemplate.py # dataset base class / classdef. used for defining other datasets
        utils.py  # Provide utility functions
        performance_metrics.py # metics for evaluatiting ML prediction

abag_agent_setup:
    decision_making.py  # Featurize historical and candidate studies.
    expand_allowed_mutant_menu.py  # Manipulate history, menu, and sample novel sequences
    mutant_generator_sampling.py  # !!! Implementation of mutant generation !!!

vaccine_advance_core:
    featurization:
        featurize_complex.py  
            # For the purposes of this work, provide the pre-computed, 
            # per-Ab/Ag co-complex analysis of what residues are interacting,
            # supporting the feature representation described in the manuscript
        interface_residues.py  # Functions used above to analyze PDB co-complexes
        seq_to_features.py  
            # Functions used to produce the feature vector for a 
            # given mutant sequence, using the above, pre-computed information.
        tally_features.py
            # Given pairs of (mutated) interacting residues, count pairs falling 
            # into various classes.
```

## Background

For historical perspective, please note that most of this code was written first 
for the design of vaccine antigens and only with the beginning of the SARS-COV-2 
pandemic was repurposed to the design of antibodies.  The oldest components of 
this archive are the contents of the vaccine_advance_core package.  The newest 
component is the pareto_select....py script in abag_ml/scripts, which was written 
at the end of the design process (early January 2022) to support selection of 
what the manuscript describes as "Set 2," ultimately 204 IgG sequences sent to 
Vanderbilt University.


## License

This source code is distributed under the MIT license. Contributions must be made under this license. 

More information can be found in the `LICENSE` and `NOTICE` files.

LLNL-CODE-853517
