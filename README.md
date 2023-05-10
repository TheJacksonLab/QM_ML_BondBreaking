# QM_ML_BondBreaking
Machine learning based adaptable bond topology (MLABT) for simulating bond scission in molecular dynamics of thermosets under large deformation

Please refer to the preprint paper https://chemrxiv.org/engage/chemrxiv/article-details/640cdd12b5d5dbe9e81b77f9 for more details, the formal paper will be available soon. 

Authors: Zheng Yu and Nick Jackson, University of Illinois at Urbana Champaign 

# Data
Includes the datasets needed for training the ML model (Model3 in the paper). 

- The 1st column is whether a bond breaks (0/1) based on the QM geometry optimization.

- The 2nd column is bond type (either 2 CT-CT or 6 CT-CA)

- The 3-6th columns are index numbers of atoms associated with the two neighboring bonds (no need to worry about)

- The >=7th columns are SOAP vectors. The details can be found in the paper. 

# Prep
Inlcudes the codes for 

- Extract local structures that potentially contain broken bonds from MD trajectories.

- Generate data based on the QM geometry optimization (ORCA)

# MLAPT
Includes the notebook for training the classification model

## Run MLAPT with Lammps 
Provides an example of how to apply MLAPT, i.e., predict bond breaking and make modification to topology, on-the-fly in MD simulations. 
 
