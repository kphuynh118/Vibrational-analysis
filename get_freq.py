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

h = 6.62607015*1e-34 #Planck's constant (Js)
c = 299792458        #Speed of light (m/s)
NA = 6.02214076e23  #Avogadro's number (mol^-1)
k = 1.380649*1e-23 #Boltzmann constant (J/K)
conversion_factor = 1000 * 4.184  #Conversion from Joules to kcal/mol
T = 298.15 #standard temperature
R = 0.0019872 #R constant in kcal/(mol K)

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
def moment_of_inertia(AtomList, Coords, AtomicMass_dict):
    I = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for atom, coord in zip(AtomList, Coords):
        mass = AtomicMass_dict[atom]
        x, y, z = coord
        I[0][0] += mass * (y**2 + z**2)
        I[1][1] += mass * (x**2 + z**2)
        I[2][2] += mass * (x**2 + y**2)
        I[0][1] -= mass * x * y
        I[1][0] -= mass * x * y
        I[0][2] -= mass * x * z
        I[2][0] -= mass * x * z
        I[1][2] -= mass * y * z
        I[2][1] -= mass * y * z

    return I

def COM(AtomList, Coords, AtomicMass_dict):
    masses = [find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList]
  
    total_mass = 0
    for i in range(0,len(masses)):
        total_mass += masses[i] 
        
    mass_times_coords = [tuple(masses[i] * Coords[i][j] for j in range(3)) for i in range(len(AtomList))]

    summed_coords = [sum(coord[j] for coord in mass_times_coords) for j in range(3)]
    
    center_of_mass = tuple(coord / total_mass for coord in summed_coords)
    return center_of_mass

def translational_V(AtomList, AtomicMass_dict):
    masses = [find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList]
    ami = np.sqrt(masses)
   
    trans_V = np.zeros((3,3*len(AtomList)))
    
    for i in range(3):
        trans_V[i, i::3] = ami

    return trans_V

def rotational_V(AtomList,AtomicMass_dict,Coords,inertia):
    masses = [find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList]
    ami = np.sqrt(masses)
    center_of_mass = COM(AtomList, Coords, AtomicMass_dict)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    rot_V = np.zeros((3,3*len(AtomList)))
    for i in range(3):
        for j,atom in enumerate(AtomList):
            # Cross product of eigenvector and (r - center_of_mass) weighted by sqrt of atomic mass
            w = np.cross(eigenvectors[:, i], (Coords[j] - np.array(center_of_mass)))
            rot_V[i, 3*j:3*j+3] = w * ami[j]

    return rot_V

def normalize(V):
    for i in range(V.shape[0]):
        norm = np.linalg.norm(V[i])
        if norm > 0:
            V[i] /= norm
    return V

def gram_schmidt(V):
    U = np.zeros_like(V)
    for i in range(V.shape[0]):
        U[i] = V[i]
        for j in range(i):
            proj = np.dot(U[j], V[i]) / np.dot(U[j], U[j])
            U[i] -= proj * U[j]

        norm = np.linalg.norm(U[i])
        if norm > 0:
            U[i] /= norm
    return U

def projected_Hessian(translational_vectors, rotational_vectors, weighted_Hessian_matrix):
    V = np.concatenate((translational_vectors,rotational_vectors),axis=0)
    V = normalize(V)
    V = gram_schmidt(V)
    V = V.reshape(6,-1).T #9x6
    
    WVT = np.dot(V,V.T)
    Idenity = np.eye(weighted_Hessian_matrix.shape[0])
  
    D = Idenity - WVT #projection matrix
     
    projected_H = np.dot(D.T,np.dot(weighted_Hessian_matrix,D))
    return projected_H  

def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix, fmt='%.6f')
    print(f"Matrix saved to '{filename}'")

def enthalpy(frequencies):
    gas_constant = R*T
    E_trans = E_rot = (3/2) * gas_constant
    
    frequencies = np.array(frequencies)
    zpve_joules = 0.5 * h * c * frequencies * 1e2  # Convert from cm^-1 to m^-1
    vib_tempt = (h * c * frequencies * 1e2 * np.exp(-h*c*frequencies*1e2/(k*T)))/(1-np.exp((-h*c*frequencies*1e2)/(k*T)))
    zpve = np.sum(zpve_joules * NA / conversion_factor)
    E_vib = np.sum(zpve_joules + vib_tempt) * NA / conversion_factor
    
    #print("Zero point vibrational energy: %12.5f" % zpve, "kcal/mol")
    #print("Translational Enthalpy: %12.5f" % E_trans,  "kcal/mol")
    #print("Rotational Enthalpy:    %12.5f" % E_rot,    "kcal/mol")
    #print("Vibrational Enthalpy:   %12.5f" % E_vib,    "kcal/mol")
    return zpve, E_trans, E_rot, E_vib

def get_moment_inertia(Coords,AtomList,AtomicMass_dict):
    natoms = Coords.shape[1]
    atomic_mass = np.array([find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList])
    total_mass = np.sum(atomic_mass)  
    
    com =  np.dot(Coords, atomic_mass) / total_mass

    shifted_coords = Coords - com[:,np.newaxis] #shift coordinates to the COM, np.newaxis make 2D column vector 
   
    inertia_tensor = np.zeros((3,3))
    for i in range(natoms):
        r_squared = np.sum(shifted_coords[:,i]**2)
        inertia_tensor += (r_squared*np.eye(3)-np.outer(shifted_coords[:,i],shifted_coords[:,i]))*atomic_mass[i] #np.outer calculates the outer product of 2 vectors
    moment_evals, principal_axes = np.linalg.eigh(inertia_tensor)

    return principal_axes, moment_evals
    
def entropy(AtomList,AtomicMass_dict,frequencies,moment_evals):
    M = 0
    S_vib_tempt = 0
    symmetry_number = 1
    R_cal = 1.987216
    total_mass = np.sum(np.array([find_atomic_mass(atom, AtomicMass_dict) for atom in AtomList])) 
    S_trans = 0.993608*(5*np.log(T) + 3*np.log(total_mass)) - 2.31482

    frequencies = np.array(frequencies)    
    expui = np.exp(frequencies*c*1e2*h/(k*T))
    vib = (frequencies*c*1e2*h / (k*T) /  (expui-1.0) - np.log(expui-1.0) + frequencies*c*1e2*h/(k*T))
    S_vib = np.sum(vib)*8.314462618*0.239005736
        
    amu_A2_to_kg_m2 = 1.66053906660e-46
    amu_to_kg = 1.660539066*1e-27
    bohr_to_angstrom = 5.291772108e-1
    V_const = 8.0*math.pi*math.pi*(k*T/h)*(amu_to_kg/h)*bohr_to_angstrom*(1e-10)*bohr_to_angstrom*(1e-10)
    
    prodI = np.prod(moment_evals*V_const)
    V_av = np.sqrt(math.pi*prodI + 1e-20)
    S_rot = R_cal * (1.5 + np.log(V_av/symmetry_number)) 

    #print("Translational Entropy:  %12.5f" % S_trans,  "cal/mol.K")
    #print("Rotational Entropy:     %12.5f" % S_rot,  "cal/mol.K")
    #print("Vibrational Entropy:    %12.5f" % S_vib,    "cal/mol.K")
    return S_trans, S_rot, S_vib

def main():
    UseMsg = '''
    python [script] [option] [output dir] [xyzfile]
    '''
    parser = argparse.ArgumentParser(description="Find the frequency using finite difference.",usage=UseMsg)
    parser.add_argument("output_dir", help="The directory to store the shifted xyz files after running force job on Qchem")
    parser.add_argument("xyzfile", help="name of the original xyz file")
    parser.add_argument('--sol',dest='sol',action='store_true',default=False,help="Turn this on when implicit solvent is used; currently supporting PCM and SMD")
    
    args = parser.parse_args()
    if args.sol:
        command = f"get_forces_in_xyzformat --sol {args.output_dir} {args.xyzfile}" 
    else:
        command = f"get_forces_in_xyzformat {args.output_dir} {args.xyzfile}"


    os.system(command)

    file_pattern = r'.*_(.+)_(.+)\.grad$'
    method, basis = find_method_and_basis(args.output_dir, file_pattern)
    print(method)
    print(basis)
    
    AtomList, Coords = xyzgeom.parse_xyz_file(args.xyzfile)
    nameroot = os.path.splitext(args.xyzfile)[0]
    output_directory = f"{nameroot}_freq_compare"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    find_j(nameroot, AtomList, args.output_dir)
    Hessian_matrix = Hessian(AtomList, args.output_dir)
    mass_weighted_Hessian_matrix = mass_weighted_Hessian(Hessian_matrix, AtomList, AtomicMass_dict)
    inertia = moment_of_inertia(AtomList, Coords, AtomicMass_dict)
    print(inertia)
    trans_V = translational_V(AtomList, AtomicMass_dict)
    rot_V = rotational_V(AtomList, AtomicMass_dict, Coords, inertia)
    projected_mass_weighted_Hessian_matrix = projected_Hessian(trans_V,rot_V,mass_weighted_Hessian_matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(projected_mass_weighted_Hessian_matrix)
    eigenvalues_au_list = [eigen * (5.4857990945*1e-4) for eigen in eigenvalues] #Convert eigevalues in amu to au 
    eigenvalues_au_list.sort()
    #lambda_list = eigenvalues_au_list[6:]
    lambda_list = [eigen for eigen in eigenvalues_au_list if np.abs(eigen) >= 1e-12] 
    
    #freq_list = [(np.sqrt(lamda)/(2*math.pi*(2.41884*1e-17)))/(2.998*1e10) for lamda in lambda_list]
    freq_list = [-(np.sqrt(np.abs(lamda)) / (2 * math.pi * (2.41884 * 1e-17))) / (2.998 * 1e10) if lamda < 0 else
    (np.sqrt(lamda) / (2 * math.pi * (2.41884 * 1e-17))) / (2.998 * 1e10) for lamda in lambda_list]
    
    data = [nameroot] + freq_list
    df = pd.DataFrame([data])
    output_csv_file_name = f"{output_directory}/{nameroot}_frequency.csv"
    df.to_csv(output_csv_file_name, index=False, header=False)
    print(f"Frequencies saved to '{output_csv_file_name}'") 

    
    borh_Coords = (Coords*angstrom_to_bohr).T
    axes, moment_evals = get_moment_inertia(borh_Coords,AtomList,AtomicMass_dict)
    enthalpy_list = enthalpy(freq_list)
    entropy_list = entropy(AtomList,AtomicMass_dict,freq_list,moment_evals)
    enthalpy_entropy_list = np.array(enthalpy_list + entropy_list).tolist()
    enthalpy_entropy_data = [nameroot] + enthalpy_entropy_list
    headers = ['rootname','zpve','trans_enthalpy','rot_enthalpy','vib_enthalpy','trans_entropy','rot_entropy','vib_entropy']
    dff = pd.DataFrame([enthalpy_entropy_data],columns=headers)
    enthalpy_entropy_csv_file_name = f"{output_directory}/{nameroot}_enthalpy_entropy.csv"
    dff.to_csv(enthalpy_entropy_csv_file_name, index=False, header=True)
    print(f"Enthalpy and entropy saved to '{enthalpy_entropy_csv_file_name}'") 

if __name__ == "__main__":
    main()
