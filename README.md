# QM_ML_BondBreaking
Codes for developing and performing the Machine-Learning-based Adaptable Bond Topology (MLABT) method for simulating bond scission in molecular dynamics of thermosets under large deformation. 

Please refer to the published papers for more details.

- [Yu, Z., & Jackson, N. E. (2023). Machine learning quantum-chemical bond scission in thermosets under extreme deformation. Applied Physics Letters, 122(21).](https://doi.org/10.1063/5.0150085)

- [Yu, Z., & Jackson, N. E. (2024). Exploring Thermoset Fracture with a Quantum Chemically Accurate Model of Bond Scission. Macromolecules.](https://doi.org/10.1021/acs.macromol.3c02549)

<!-- Authors: Zheng Yu and Nick Jackson, University of Illinois at Urbana Champaign  -->

## Dependencies

```bash
# Create a Conda environment
conda create --name your_env_name python=3.8
conda activate your_env_name

# Install required Python packages
pip install -r requirements.txt
```

## Prep

### Crosslinking of the thermoset network

Please refer to the github repository [bond_react_MD](https://github.com/zyumse/bond_react_MD) for detailed guidelines

### Extract local structures from MD trajectories

- Select the local environements that potentially contain broken bonds

- Compute the SOAP vectors as the inputs

### Get QM output of bond breaking

- ORCA geometry optimization, via either DFT or xTB

- Determine whether the bond is broken based on the final distance between the two atoms initially involved in the bond

## MLAPT

### Models

The directory includes the trained SVM model with active learning at the level of DFT. The models can be easily loaded with pickle and be used as a sklearn model object.

### Training

### Active learning

### Run MLAPT with Lammps

Provides an example of how to apply MLAPT, i.e., predict bond breaking and make modification to topology, on-the-fly in MD simulations.

## Data

Includes the datasets needed for training the ML model (Model3 in the paper).

- The 1st column is whether a bond breaks (0/1) based on the QM geometry optimization.

- The 2nd column is bond type (either 2 CT-CT or 6 CT-CA)

- The 3-6th columns are index numbers of atoms associated with the two neighboring bonds (no need to worry about)

- The >=7th columns are SOAP vectors. The details can be found in the paper.