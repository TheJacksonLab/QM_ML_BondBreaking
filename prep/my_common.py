#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import numpy.linalg as LA
import re
import scipy.constants as scc
import pandas as pd

# In[67]:


def read_gro(file_name):
    """return box,natom,type_atom,coors"""
    f = open(file_name,'r')
    lf = list(f)
    f.close()
    len1 = float(lf[-1].split()[0])
    len2 = float(lf[-1].split()[1])
    len3 = float(lf[-1].split()[2])
    box = np.diag([len1,len2,len3])
    natom = int(lf[1])
    coors = np.zeros([natom,3])
    type_atom = []
    l=0
    for ia in lf[2:2+natom]:
        if l<9999:
            coors[l,:] = np.array(ia.split()[3:6:1]).astype('float')
            type_atom.append(ia.split()[1])
        else:
            coors[l,:] = np.array(ia.split()[2:5:1]).astype('float')
            tmp=ia.split()[1]
            type_atom.append(re.findall(r'(\w+?)(\d+)',tmp)[0][0])
        l+=1
        
    type_atom = np.array(type_atom)
    return box,natom,type_atom,coors


def write_gro(filename,box,natom,type_atom,coors):
    """Input:filename,box,natom,type_atom,coors"""
    f = open('{}.gro'.format(filename),'w')
    f.write('SiO2\n')
    f.write('{0:5d} \n'.format(natom))
    for i in range(natom):
        if i<9999:
            f.write('{0:5d}SIO     {1:s}{2:5d}{3:8.3f}{4:8.3f}{5:8.3f}{6:8.4f}{7:8.4f}{8:8.4f}\n'.format(i+1,type_atom[i],i+1,coors[i,0],coors[i,1],coors[i,2],0,0,0))
        else:
            f.write('{0:5d}SIO     {1:s}{2:6d}{3:8.3f}{4:8.3f}{5:8.3f}{6:8.4f}{7:8.4f}{8:8.4f}\n'.format(i+1,type_atom[i],i+1,coors[i,0],coors[i,1],coors[i,2],0,0,0))
    f.write('{0:10.5f}{1:10.5f}{2:10.5f}'.format(box[0,0],box[1,1],box[2,2])) # only work for othorgonal boxes
    f.close()

def read_gro_multi(gro_file):
    """
    read multiple frames in one gro file,
    return a list, in which each element contains box, natom, type_atom, coors
    """
    
    f = open(gro_file)
    lft = list(f)
    f.close()
    lt=[]
    t=[]
    for il in range(len(lft)):
        if 't=' in lft[il]:
            lt.append(il)
            t.append(lft[il].split()[2])
    
    def read_lf(lf):
        len1 = float(lf[-1].split()[0])
        len2 = float(lf[-1].split()[1])
        len3 = float(lf[-1].split()[2])
        box = np.diag([len1,len2,len3])
        natom = int(lf[1])
        coors = np.zeros([natom,3])
        type_atom = []
        l=0
        for ia in lf[2:2+natom]:
            if l<9999:
                coors[l,:] = np.array(ia.split()[3:6:1]).astype('float')
                type_atom.append(ia.split()[1])
            else:
                coors[l,:] = np.array(ia.split()[2:5:1]).astype('float')
                tmp=ia.split()[1]
                type_atom.append(re.findall(r'(\w+?)(\d+)',tmp)[0][0])
            l+=1

        type_atom = np.array(type_atom)
        return box,natom,type_atom,coors
    
    # lf=[]
    result = []
    for it in range(len(lt)):
        if it==len(lt)-1:
    #         lf.append(lft[lt[it]:])
            result.append(read_lf(lft[lt[it]:]))
        else:
    #         lf.append(lft[lt[it]:lt[it+1]-1])
            result.append(read_lf(lft[lt[it]:lt[it+1]]))
    
    return result,t
    
    




def CN(box,coors,cutoff):
    """ 
    return CN, CN_idx, CN_dist, diff 
    """
    
    rcoors = np.dot(coors,np.linalg.inv(box))

    r1 = rcoors[:,np.newaxis,:]
    r2 = rcoors[np.newaxis,:,:]

    rdis = r1-r2

    while np.sum((rdis<-0.5) | (rdis>0.5))>0 :
        rdis[rdis<-0.5] = rdis[rdis<-0.5]+1
        rdis[rdis>0.5] = rdis[rdis>0.5]-1

    diff = np.dot(rdis,box)

    dis = np.sqrt(np.sum(np.square(diff),axis=2))

    CN_idx = [];
    CN_dist = [];
    CN = np.zeros(coors.shape[0])
    for i in range(coors.shape[0]):
        tmp = np.argwhere((dis[i,:]<cutoff) & (dis[i,:]>0) )
        CN[i] = tmp.shape[0]
        CN_idx.append(tmp)
        CN_dist.append(dis[i,(dis[i,:]<cutoff) & (dis[i,:]>0)])
    return CN, CN_idx, CN_dist, diff


def bond_angle_SiOSi(type_atoms, CN_idx, vectors):
    '''
    CN_idx stores ...
    vectors stores n*n*3 matrix, diff
    ''' 
    bond_angle = []
    for i in np.argwhere(type_atoms=='O')[:,0]:
        for j1 in range(CN_idx[i].shape[0]):
            for j2 in np.arange(j1+1,CN_idx[i].shape[0]):
                a1 = vectors[np.int(CN_idx[i][j1]),i,:]
                a2 = vectors[np.int(CN_idx[i][j2]),i,:]
                cos_tmp = np.dot(a1,a2)/LA.norm(a1)/LA.norm(a2)
#                 if cos_tmp>1:
#                     cos_tmp=1
#                 elif cos_tmp<-1
#                     cos_tmp=-1
                bond_angle.append(np.arccos(cos_tmp)/np.pi*180)
    bond_angle = np.array(bond_angle)
    bond_angle = bond_angle[np.logical_not(np.isnan(bond_angle))]
    
    return bond_angle

def bond_angle_OSiO(type_atoms, CN_idx, vectors):
    """
    CN_idx stores ...
    vectors stores n*n*3 matrix, diff
    """
    bond_angle = []
    for i in np.argwhere(type_atoms=='Si')[:,0]:
        for j1 in range(CN_idx[i].shape[0]):
            for j2 in range(CN_idx[i].shape[0]):
                a1 = vectors[np.int(CN_idx[i][j1]),i,:]
                a2 = vectors[np.int(CN_idx[i][j2]),i,:]
                cos_tmp = np.dot(a1,a2)/LA.norm(a1)/LA.norm(a2)
                
                bond_angle.append(np.arccos(cos_tmp)/np.pi*180)
    bond_angle = np.array(bond_angle)
    bond_angle = bond_angle[np.logical_not(np.isnan(bond_angle))]

    return bond_angle

def bond_length_SiO(type_atoms, CN_idx, vectors):
    """
    """
    bond_length = [] 
    for i in np.argwhere(type_atoms=='Si')[:,0]:
        for j1 in range(CN_idx[i].shape[0]):
            a1 = vectors[np.int(CN_idx[i][j1]),i,:]
            bond_length.append(np.sum(a1**2)**0.5)

    return bond_length

def density_SiO2(box,type_atom):
    density = (np.sum(type_atom=='Si')*28.084 + np.sum(type_atom=='O')*15.999)/scc.Avogadro/np.dot(np.cross(box[:,0],box[:,1]),box[:,2])*1e21
    return density 


def gro2pos(posfile,grofile):
    """Convert .gro to POSCAR """
    box,natom,type_atom,coors0 = read_gro('{}'.format(grofile))

    elements = list(set(type_atom))
    n_elements = len(elements)

    n_atom=[]
    id_atom=[]
    for i_e in range(n_elements):
        id_atom.append(np.array(np.where(type_atom==elements[i_e]))[0])
        n_atom.append(id_atom[i_e].shape[0])

    f1=open('{}'.format(posfile),'w')
    f1.write('generated by gro2pos\n')
    f1.write(' 1.0\n')
    for ib in range(3):
        f1.write(' {0:20.12f} {1:20.12f} {2:20.12f}\n'.format(10*box[ib,0],10*box[ib,1],10*box[ib,2]))

    for i_e in range(n_elements):
        f1.write(' {}'.format(elements[i_e]))
    f1.write('\n')

    for i_e in range(n_elements):
        f1.write(' {0:8d}'.format(n_atom[i_e]))
    f1.write('\n')
    f1.write('C\n')
    for i_e in range(n_elements):
        for ic in range(n_atom[i_e]):
            f1.write(' {0:20.12f} {1:20.12f} {2:20.12f}\n'.format(10*coors0[id_atom[i_e][ic],0],10*coors0[id_atom[i_e][ic],1],10*coors0[id_atom[i_e][ic],2]))

    f1.close()




def supercell(natoms,box0,nx,ny,nz,index0,atom_type0,coors0):
    """
    at this moment, only for orthogonal cell
    """
    
    box_new = box0@np.array([[nx,0,0],[0,ny,0],[0,0,nz]])
    natoms_new = natoms*nx*ny*nz
    
    # coors_new = np.empty([1,3])
    # index_new = np.empty

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ix+iy+iz==0:
                    coors_new=coors0
                    atom_type_new = atom_type0
                    index_new = index0
                    index_tmp = index0
                else:
                    coors_tmp = coors0 + ix*box0[0,:] + iy*box0[1,:] + iz*box0[2,:]
                    coors_new = np.vstack((coors_new,coors_tmp))

                    atom_type_new = np.concatenate((atom_type_new,atom_type0))
                    index_tmp = index_tmp+natoms
                    index_new = np.concatenate((index_new,index_tmp))
                    
    return natoms_new,box_new,index_new,atom_type_new,coors_new


def write_lammps(file_name,box_new,ntypes,natoms_new,mass,charge,index_new,atom_type_new,coors_new):
    """
    A = (xhi-xlo,0,0); B = (xy,yhi-ylo,0); C = (xz,yz,zhi-zlo)
    
    shift the origin to 0 0 0 
    """
    f=open('{}'.format(file_name),'w')
    f.write('Generated by python script\n\n')

    f.write('{0:5d} atoms\n'.format(natoms_new))
    f.write('{0:d} atom types\n'.format(ntypes))
    f.write('\n')

    f.write('{0:.16f} {1:.16f} xlo xhi\n'.format(0,box_new[0,0]))
    f.write('{0:.16f} {1:.16f} ylo yhi\n'.format(0,box_new[1,1]))
    f.write('{0:.16f} {1:.16f} zlo zhi\n'.format(0,box_new[2,2]))
    
    if box_new[0,1]!=0 or box_new[0,1]!=0 or box_new[0,1]!=0: 
        f.write('{0:.16f} {1:.16f} {2:.16f} xy xz yz\n'.format(box_new[0,1],box_new[0,2],box_new[1,2]))
    
    f.write('\n')
    f.write('Masses\n\n')

    for i in range(len(mass)):
        f.write('{0:d} {1:.6f}\n'.format(i+1,mass[i]))
    f.write('\n')
    f.write('Atoms # charge\n\n')

    for ia in range(natoms_new):
        f.write('{0:d} {1:d} {2:.4f} {3:.16f} {4:.16f} {5:.16f}\n'.format(int(index_new[ia]),int(atom_type_new[ia]),charge[int(atom_type_new[ia])-1],coors_new[ia,0],coors_new[ia,1],coors_new[ia,2]))

    f.close()

def read_lammps(file, lmp_mode='charge'):
    """
    shift the original point to 0 0 0
    """

    f=open(file,'r')
    L=f.readlines()
    f.close()

    isxyxzyz = 0
    for iline in range(len(L)):
        if 'atoms' in L[iline]:
            natoms = int(L[iline].split()[0])
        if 'xlo' in L[iline]:
            xlo=float(L[iline].split()[0])
            xhi=float(L[iline].split()[1])
        if 'ylo' in L[iline]:
            ylo=float(L[iline].split()[0])
            yhi=float(L[iline].split()[1])
        if 'zlo' in L[iline]:
            zlo=float(L[iline].split()[0])
            zhi=float(L[iline].split()[1])

        if 'xy' in L[iline]:
            isxyxzyz=1
            xy=float(L[iline].split()[0])
            xz=float(L[iline].split()[1])
            yz=float(L[iline].split()[2])

        if 'Atoms #' in L[iline]:
            latom = iline+2

    if isxyxzyz==0:
        xy=0; xz=0; yz=0

    box = np.array([[xhi-xlo,0,0],[xy,yhi-ylo,0],[xz,yz,zhi-zlo]])

    index = np.empty(natoms)
    atom_type = np.empty(natoms)
    coors = np.empty([natoms,3])

    for i in range(natoms):
        index[i] = int(L[latom+i].split()[0])
        if lmp_mode =='charge':
            atom_type[i] = int(L[latom+i].split()[1])
            coors[i,:] = np.array([float(L[latom+i].split()[3])-xlo,float(L[latom+i].split()[4])-ylo,float(L[latom+i].split()[5])-zlo])
        elif lmp_mode =='full':
            atom_type[i] = int(L[latom+i].split()[2])
            coors[i,:] = np.array([float(L[latom+i].split()[4])-xlo,float(L[latom+i].split()[5])-ylo,float(L[latom+i].split()[6])-zlo])

    if atom_type[-1]>0:
        return natoms,box,index,atom_type,coors
    else:
        print(error)


def read_log_lammps(logfile):
    f=open(logfile,'r')
    L=f.readlines()
    f.close()
    for i in range(len(L)):
        if 'Step' in L[i]:
            l1=i
        if 'Loop time' in L[i]:
            l2=i
    data = np.array(L[l1+1].split())
    for i in range(l1+1,l2):
        data = np.vstack((data,L[i].split()))
    data = pd.DataFrame(data,dtype='float64',columns=L[l1].split())
    return data

def read_mutiple_xyz(file):
    """
    read multiple frames in output xyz of lammps, 
    input: xyz_file, 
    output: result (type_atoms, coors) and t 
    """
    f = open(file)
    lft = list(f)
    f.close()
    lt=[]
    t=[]
    natom = int(lft[0].split()[0])
    for il in range(len(lft)):
        if 'Timestep:' in lft[il]:
            lt.append(il)
            t.append(lft[il].split()[2])

    def read_lf(lf):
        coors = np.zeros([natom,3])
        type_atom = []
        l=0
        for ia in lf[1:1+natom]:
            coors[l,:] = np.array(ia.split()[1:4:1]).astype('float')
            type_atom.append(ia.split()[0])
            l+=1

        type_atom = np.array(type_atom)
        return type_atom,coors

    # lf=[]
    result = []
    for it in range(len(lt)):
        if it==len(lt)-1:
    #         lf.append(lft[lt[it]:])
            result.append(read_lf(lft[lt[it]:]))
        else:
    #         lf.append(lft[lt[it]:lt[it+1]-1])
            result.append(read_lf(lft[lt[it]:lt[it+1]]))

    return result,t

def read_pos(file_name):
    """
    read POSCAR format structure file for VASP calculations
    at this moment, only 'C' is applied
    """
    f = open(file_name,'r')
    lf = list(f)
    f.close()
    box=np.zeros((3,3))
    ratio = float(lf[1].split()[0])
    box[0,:] = np.array(lf[2].split()).astype(float)*ratio
    box[1,:] = np.array(lf[3].split()).astype(float)*ratio
    box[2,:] = np.array(lf[4].split()).astype(float)*ratio
    a_type = np.array(lf[5].split())
    num_type =  np.array(lf[6].split()).astype(int)

    natom = np.sum(num_type)
    coors = np.zeros((natom,3))

    if lf[7].split()[0]=='C' or lf[7].split()[0]=='c':
        l=0
        for ia in lf[8:8+natom]:
            coors[l,:]= np.array(ia.split()[0:3:1]).astype('float')
            l+=1

    if lf[7].split()[0][0]=='D' or lf[7].split()[0][0]=='d':
        l=0
        rcoors = np.zeros((natom,3))
        for ia in lf[8:8+natom]:
            rcoors[l,:]= np.array(ia.split()[0:3:1]).astype('float')
            l+=1
        coors = rcoors @ box

    return box,a_type,num_type,coors

def write_pos(file_name,box,a_type,num_type,coors):
    """
    write POSCAR format structure file for VASP calculations
    input: file_name,box,a_type,num_type,coors
    """
    f = open(file_name,'w')
    f.write('written by python script\n')
    f.write('1.0\n')

    for i in range(3):
        f.write('{0:20.12f}{1:20.12f}{2:20.12f}\n'.format(box[i,0],box[i,1],box[i,2]))

    for i in range(len(a_type)):
        f.write(' {}'.format(a_type[i]))
    f.write('\n')
    for i in range(len(a_type)):
        f.write(' {}'.format(num_type[i]))
    f.write('\n')

    natom = np.sum(num_type)
    f.write('C\n')
    for i in range(natom):
        f.write('{0:20.12f}{1:20.12f}{2:20.12f}\n'.format(coors[i,0],coors[i,1],coors[i,2]))

    f.close()

def pdf_sq_1type(box,natom,type_atom,coors,r_cutoff=10,delta_r = 0.01):
    """
    only one type of particles
    inputs: box,natom,type_atom,coors,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    type_atom = np.array(type_atom)
    n1 = natom
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([natom, natom, 3])
    for i in range(natom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1 / V
    c = np.array([rho1 * rho1]) * V
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) / 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R,g1,Q,S1

def write_xyz(file_name,natoms,type_atoms,coors):
    f=open('{}'.format(file_name),'w')
    f.write('{0:5d}\n'.format(natoms))
    f.write('Generated by python script\n')
    for ia in range(natoms):
        f.write('{0:}     {1:.16f} {2:.16f} {3:.16f}\n'.format(type_atoms[ia],coors[ia,0],coors[ia,1],coors[ia,2]))
    f.close()
