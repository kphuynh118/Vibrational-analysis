#!/usr/bin/env python
import os, glob, sys, re, math
import numpy as np
import xyzgeom
import pandas as pd
from optparse import OptionParser
import argparse
import shutil
#after the user run force job on qchem, all output files should be in get_forces_qchem_input_files
#user need to move all output files into one directory 

AtomicMass_dict =   {
    'H': 1.00783, 'He': 4.00260, 'Li': 7.01600, 'Be': 9.01218, 'B': 11.00931,
    'C': 12.00000, 'N': 14.00307, 'O': 15.99491, 'F': 18.99840, 'Ne': 19.99244,
    'Na': 22.9898, 'Mg': 23.98504, 'Al': 26.98153, 'Si': 27.97693, 'P': 30.97376,
    'S': 31.97207, 'Cl': 34.96885, 'Ar': 39.948, 'K': 38.96371, 'Ca': 39.96259,
    'Sc': 44.95592, 'Ti': 47.90, 'V': 50.9440, 'Cr': 51.9405, 'Mn': 54.9380,
    'Fe': 55.9349, 'Co': 58.9332, 'Ni': 57.9353, 'Cu': 62.9296, 'Zn': 63.9291,
    'Ga': 68.9257, 'Ge': 73.9219, 'As': 74.9216, 'Se': 79.9165, 'Br': 78.9183,
    'Kr': 83.80, 'Rb': 84.9117, 'Sr': 87.9056, 'Y': 88.9059, 'Zr': 89.9043,
    'Nb': 92.9060, 'Mo': 97.9055, 'Tc': 98.9062, 'Ru': 101.9037, 'Rh': 102.9048,
    'Pd': 105.9032, 'Ag': 106.90509, 'Cd': 113.9036, 'In': 114.9041, 'Sn': 118.69,
    'Sb': 120.9038, 'Te': 129.9067, 'I': 126.9044, 'Xe': 131.9042, 'Cs': 132.9051,
    'Ba': 137.9050, 'La': 138.9061, 'Ce': 139.9053, 'Pr': 140.9074, 'Nd': 141.9075,
    'Pm': 144.913, 'Sm': 151.9195, 'Eu': 152.9209, 'Gd': 157.9241, 'Tb': 159.9250,
    'Dy': 163.9288, 'Ho': 164.9303, 'Er': 165.9304, 'Tm': 168.9344, 'Yb': 173.9390,
    'Lu': 174.9409, 'Hf': 179.9468, 'Ta': 180.9480, 'W': 183.9510, 'Re': 186.9560,
    'Os': 192, 'Ir': 192.9633, 'Pt': 194.9648, 'Au': 196.9666, 'Hg': 201.9706,
    'Tl': 204.9745, 'Pb': 207.9766, 'Bi': 208.9804, 'Po': 208.9825, 'At': 209.987,
    'Rn': 222.0175, 'Fr': 223.0198, 'Ra': 226.0254, 'Ac': 227.0278, 'Th': 232.0382,
    'Pa': 231.0359, 'U': 238.0508, 'Np': 237.0480, 'Pu': 244.064, 'Am': 243.0614,
    'Cm': 247.070, 'Bk': 247.0702, 'Cf': 251.080, 'Es': 254.0881, 'Fm': 257.095,
    'Md': 258.099, 'No': 259.101, 'Lr': 266.1198 
}
def find_atomic_mass(symbol, atomic_mass_dict):
    if symbol in atomic_mass_dict:
        return atomic_mass_dict[symbol]
    else:
        return "Element not found"

def find_j(nameroot, AtomList, directory):
    file_pattern = r'.*_(.+)_(.+)\.grad$'
    method, basis = find_method_and_basis(directory, file_pattern)
    for i in range(len(AtomList)):
        for ii in range(3):
            old_filename_plus = os.path.join(directory, f"{nameroot}_shift_atom_{i+1}_coord_{ii+1}_+_{method}_{basis}.grad")
            j = i * 3 + ii + 1
            new_filename_plus = os.path.join(directory, f"j{j}+.grad")

            if os.path.exists(old_filename_plus):
                try:
                    os.rename(old_filename_plus, new_filename_plus)
                
                except OSError as error:
                    print(f"Error renaming file {old_filename_plus}: {error}")
            else:
                print(f"File not found: {old_filename_plus}")
    for i in range(len(AtomList)):
        for ii in range(3):
            old_filename_minus = os.path.join(directory, f"{nameroot}_shift_atom_{i+1}_coord_{ii+1}_-_{method}_{basis}.grad")
            j = i * 3 + ii + 1
            new_filename_minus = os.path.join(directory, f"j{j}-.grad")

            if os.path.exists(old_filename_minus):
                try:
                    os.rename(old_filename_minus, new_filename_minus)
                
                except OSError as error:
                    print(f"Error renaming file {old_filename_minus}: {error}")
            else:
                print(f"File not found: {old_filename_minus}")

def Hessian(AtomList, directory):
    coords_plus = {}
    matrix_plus = {}
    coords_minus = {}
    matrix_minus = {}
    Hessian_dict = {}
    Hessian_matrix = np.zeros((len(AtomList) * 3, len(AtomList) * 3))
    sym_Hessian_matrix = np.zeros((len(AtomList) * 3, len(AtomList) * 3))
    for i in range(1,(len(AtomList)*3+1)):
        #filename_plus = os.path.join(directory,f"j{i}+.grad")
        #filename_minus = os.path.join(directory,f"j{i}-.grad")
    
        atoms_list_plus, coords_list_plus = xyzgeom.parse_xyz_file(os.path.join(directory,f"j{i}+.grad"))
        coords_plus[i] = coords_list_plus
        matrix_plus[i] = coords_plus[i].flatten()
        atoms_list_minus,coords_list_minus = xyzgeom.parse_xyz_file(os.path.join(directory,f"j{i}-.grad"))
        coords_minus[i] = coords_list_minus
        matrix_minus[i] = coords_minus[i].flatten()   
    for i in range(1,(len(AtomList)*3+1)):
        Hessian_dict[i] = (0.529117/0.002)*(matrix_plus[i] - matrix_minus[i]) 
    for i in range(len(AtomList)*3):
        Hessian_matrix[i,:]=Hessian_dict[i+1]   

    sym_Hessian_matrix = 0.5 * (Hessian_matrix + np.transpose(Hessian_matrix))
    return sym_Hessian_matrix

def mass_weighted_Hessian(H, AtomList, AtomicMass_dict):
    
    masses = [find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList]
    
    # Expand the mass list for each coordinate
    new_mass_list = np.repeat(masses, 3)

    weighted_H = np.zeros((len(new_mass_list), len(new_mass_list)))

    for i in range(len(new_mass_list)):
        for j in range(len(new_mass_list)):
            weighted_H[i,j] = H[i,j] / np.sqrt(new_mass_list[i] * new_mass_list[j])
        
    return weighted_H 

def find_method_and_basis(directory, file_pattern):
    pattern = re.compile(file_pattern)
    for filename in glob.glob(os.path.join(directory, '*.grad')):
        match = pattern.search(os.path.basename(filename))
        if match:
            return match.groups()  # Returns the method and basis
    return None, None

def main():
    UseMsg = '''
    python [script] [output dir] [xyzfile]
    '''
    parser = argparse.ArgumentParser(description="Find the frequency using finite difference.",usage=UseMsg)
    parser.add_argument("output_dir", help="The directory to store the shifted xyz files after running force job on Qchem")
    parser.add_argument("xyzfile", help="name of the original xyz file")

    args = parser.parse_args()

    command = f"get_forces_in_xyzformat {args.output_dir} {args.xyzfile}"
    os.system(command)

    file_pattern = r'.*_(.+)_(.+)\.grad$'
    method, basis = find_method_and_basis(args.output_dir, file_pattern)
  
    
    AtomList, Coords = xyzgeom.parse_xyz_file(args.xyzfile)
    nameroot = os.path.splitext(args.xyzfile)[0]
    find_j(nameroot, AtomList, args.output_dir)
    Hessian_matrix = Hessian(AtomList, args.output_dir)
    weighted_Hessian = mass_weighted_Hessian(Hessian_matrix, AtomList, AtomicMass_dict)
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_Hessian)
    #Convert eigevalues in amu to au 
    eigenvalues_au_list = [eigen * (5.4857990945*1e-4) for eigen in eigenvalues]
    eigenvalues_au_list.sort()
    lambda_list = eigenvalues_au_list[6:]
    freq_list = [(np.sqrt(lamda)/(2*math.pi*(2.41884*1e-17)))/(2.998*1e10) for lamda in lambda_list]
    print(freq_list)
    
if __name__ == "__main__":
    main()
