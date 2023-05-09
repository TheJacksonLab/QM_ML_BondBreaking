import numpy as np
import pandas as pd
import my_common as mc
import extract_local_str as els
import scipy.constants as scc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# import joblib
import os
import pickle 

from dscribe.descriptors import SOAP 
from ase import Atoms

f_check = 1000
t_rate = 0.000001

soap = SOAP(
    species=['C','H','O','N'],
    periodic=True,
    r_cut=4.0,
    n_max=1,
    l_max=8,
)

with open('model3soap.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler3soap.pkl', 'rb') as f:
    scaler = pickle.load(f)

os.system('mpirun lmp -in in.deform0')
is_break = []
n_break = 0
itime = int(0.5/t_rate/f_check-1)   # starting from true strain = 0.5 
log = (mc.read_log_lammps('log.lammps'))
BB_tmp = np.zeros(len(log))
BB_tmp[0] = np.sum(is_break)
log.insert(1,'BB',BB_tmp)

while (itime < 10000) & (n_break<70):
    # read structure file 
    lmp_tmp = els.read_lammps_full('tmp.dat')
    bc = lmp_tmp.bond_coeff
    bond_info = lmp_tmp.bond_info
    natoms,box,index,atom_type,coors = mc.read_lammps('tmp.dat','full')
    coors = coors[np.argsort(index)]
    atom_type = atom_type[np.argsort(index)]
    index = index[np.argsort(index)]
    
    atom_type_new = np.empty(atom_type.shape,dtype=str)
    atom_type_new[atom_type==1] = 'C'
    atom_type_new[atom_type==2] = 'C'
    atom_type_new[atom_type==3] = 'C'
    atom_type_new[atom_type==4] = 'H'
    atom_type_new[atom_type==5] = 'H'
    atom_type_new[atom_type==6] = 'H'
    atom_type_new[atom_type==7] = 'H'
    atom_type_new[atom_type==8] = 'N'
    atom_type_new[atom_type==9] = 'O'
    atom_type_new[atom_type==10] = 'O'
    atom_type_new[atom_type==11] = 'O'
    case1 = Atoms(atom_type_new,positions=coors,cell=box,pbc=[1,1,1])
    input_deform = soap.create(case1)
    
    bond_idx = bond_info[:,2:].astype(int)-1
    idx_26a = bond_idx[(bond_info[:,1]==2) | (bond_info[:,1]==6)]
    idx_26b = np.squeeze(np.argwhere((bond_info[:,1]==2) | (bond_info[:,1]==6)))
    bond_info_26 = bond_info[idx_26b]  
    
    # find the neighboring bond
    idx_26bn = np.zeros(len(idx_26b))
    for i26a in range(len(idx_26a)):
        idx_neighbor = np.argwhere(((bond_idx[:,0] == idx_26a[i26a,0]) | (bond_idx[:,1] == idx_26a[i26a,0])
        | (bond_idx[:,0] == idx_26a[i26a,1]) | (bond_idx[:,1] == idx_26a[i26a,1]))
        & (bond_info[:,1] == bond_info[idx_26b[i26a],1]))
        idx_neighbor = np.setdiff1d(idx_neighbor,idx_26b[i26a])
        if len(idx_neighbor)!=1:
            print(len(idx_neighbor),n_break,flush=True)
            idx_26bn[i26a] = -1
            # exit('neighbor searching error')
        else:
            # print(i26a,idx_neighbor)
            idx_26bn[i26a] = idx_neighbor[0]
    idx_26bn = idx_26bn.astype(int)

    idx_26b_real = idx_26b[idx_26bn>0] ### the one with only 1 neighboring bond
    idx_26bn_real = idx_26bn[idx_26bn>0]
    idx_26a_real = idx_26a[idx_26bn>0,:]

    idx_atom_26b_real = bond_idx[idx_26b_real]
    idx_atom_26bn_real = bond_idx[idx_26bn_real]
   
    bond_length = els.BL(box, coors, bond_idx)
    ebond_length = bc[bond_info[:,1].astype(int)-1,2]
    ebond_k = bc[bond_info[:,1].astype(int)-1,1]
    rstrain_bond = (bond_length-ebond_length)/ebond_length
    energy_bond = ebond_k*(bond_length-ebond_length)**2
    
    # predict bond breaking 
    X = np.hstack((input_deform[idx_atom_26b_real][:,0,:],input_deform[idx_atom_26b_real][:,1,:])) + \
    np.hstack((input_deform[idx_atom_26bn_real][:,0,:],input_deform[idx_atom_26bn_real][:,1,:]))

    is_break = model.predict(scaler.transform(X))

    for i26 in range(len(idx_26b_real)):
        if is_break[i26] == 1:
            if rstrain_bond[idx_26b_real[i26]] > rstrain_bond[idx_26bn_real[i26]]:
                is_break[i26] = 1
            else:
                is_break[i26] = 0

    # change topology
    if np.sum(is_break)>0:
        print(bond_info[np.squeeze(idx_26b_real[np.argwhere(is_break)]),:])
        bond_info_new = np.delete(bond_info,np.squeeze(idx_26b_real[np.argwhere(is_break)]),axis=0)
        lmp_tmp.bond_info=bond_info_new
        lmp_tmp.nbonds = len(bond_info_new)
        
        idx_del_angle = []
        idx_del_dihedral = []
        idx_del_improper = []
        for ibb in range(np.sum(is_break==1)):
            idx_a1 = idx_26a_real[np.squeeze(np.argwhere(is_break))].reshape(-1,2)[ibb,0]
            idx_a2 = idx_26a_real[np.squeeze(np.argwhere(is_break))].reshape(-1,2)[ibb,1]
            for iangle in range(len(lmp_tmp.angle_info)):
                if (idx_a1 in lmp_tmp.angle_info[iangle,2:]-1) and (idx_a2 in lmp_tmp.angle_info[iangle,2:]-1): 
                    idx_del_angle.append(iangle)
            for idihedral in range(len(lmp_tmp.dihedral_info)):
                if (idx_a1 in lmp_tmp.dihedral_info[idihedral,2:]-1) and (idx_a2 in lmp_tmp.dihedral_info[idihedral,2:]-1):
                    idx_del_dihedral.append(idihedral)
            for iimproper in range(len(lmp_tmp.improper_info)):
                if (idx_a1 in lmp_tmp.improper_info[iimproper,2:]-1) and (idx_a2 in lmp_tmp.improper_info[iimproper,2:]-1):
                    idx_del_improper.append(iimproper)        
        angle_info_new = np.delete(lmp_tmp.angle_info,idx_del_angle,axis=0)
        lmp_tmp.angle_info = angle_info_new
        lmp_tmp.nangles = len(angle_info_new)
        dihedral_info_new = np.delete(lmp_tmp.dihedral_info,idx_del_dihedral,axis=0)
        lmp_tmp.dihedral_info = dihedral_info_new
        lmp_tmp.ndihedrals = len(dihedral_info_new)
        improper_info_new = np.delete(lmp_tmp.improper_info,idx_del_improper,axis=0)
        lmp_tmp.improper_info = improper_info_new
        lmp_tmp.nimpropers = len(improper_info_new)
        
        els.write_lammps_full('tmp.dat',lmp_tmp)
    
    n_break = n_break+ np.sum(is_break)
    print('Num of broken bonds = ', np.sum(is_break), n_break, flush = True)
    # print(mc.read_log_lammps('log.lammps').iloc[-1,0], np.sum(is_break), n_break)
    itime = itime+1
    os.system('sed ''s/xxx/{}/g'' in.deform > in.deform1'.format(itime*f_check))
    # run MD simulation
    os.system('mpirun lmp -in in.deform1')
    log_tmp = mc.read_log_lammps('log.lammps')
    BB_tmp = np.zeros(len(log_tmp))
    BB_tmp[1] = np.sum(is_break)
    log_tmp.insert(1,'BB',BB_tmp)
    print(log_tmp)
    log = pd.concat((log,log_tmp),ignore_index=True)
    log.to_csv('log.csv',header=True,index=False)

