import pandas as pd
import numpy as np
import argparse
import os
import glob
#dir where two csv files you want to compare are located
#to get the freq out of qchem output file, use get_geom_freq --do_freq -t [dir] where [dir] contains the output files. Ideally [dir] should contains two files that have freq from qchem and freq from my code

def root_mean_square_error(data1, data2):
    """Calculate the Root Mean Square Error between two data sets."""
    return np.sqrt((data1 - data2) ** 2)
    
def read_frequency_data(file_path):
    """Read the data from a frequency CSV file."""
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        #Handle comma 
        if line.endswith(','):
            line = line[:-1]
        data = line.split(',')[1:]  #skip the name and split the rest
        data = [float(value) for value in data]  
    return data

def find_frequency_files(directory):
    """Find the specific files in the given directory."""
    frequency_file = os.path.join(directory, 'frequency.csv')
    matched_files = glob.glob(os.path.join(directory, '*_frequency.csv'))
    rootname_frequency_files = [file for file in matched_files if file != frequency_file]

    if not os.path.exists(frequency_file) or not rootname_frequency_files:
        raise FileNotFoundError("Required files not found in the directory.")

    return frequency_file, rootname_frequency_files[0]

def read_enthalpy_entropy_data(file_path):
    """Read the data from a enthalpy entropy CSV file."""
    with open(file_path, 'r') as file:
        next(file) #skip the header
        line = file.readline().strip()
        #handle comma 
        if line.endswith(','):
            line = line[:-1]
        data = line.split(',')[1:]  #skip the name and split the rest
        data = [float(value) for value in data]  
    return data

def find_enthalpy_entropy_files(directory):
    """Find the specific files in the given directory."""
    enthalpy_entropy_file = os.path.join(directory, 'QChem_enthalpy_entropy.csv')
    matched_files = glob.glob(os.path.join(directory, '*_enthalpy_entropy.csv'))
    rootname_enthalpy_entropy_files = [file for file in matched_files if file != enthalpy_entropy_file]

    if not os.path.exists(enthalpy_entropy_file) or not rootname_enthalpy_entropy_files:
        raise FileNotFoundError("Required files not found in the directory.")

    return enthalpy_entropy_file, rootname_enthalpy_entropy_files[0]

def compare_frequency(directory):
    frequency_file, rootname_frequency_file = find_frequency_files(directory)

    data1 = read_frequency_data(frequency_file)  # Qchem 
    data2 = read_frequency_data(rootname_frequency_file)  # Mycode 

    rmse_values = root_mean_square_error(np.array(data1), np.array(data2))
    output_csv_file_name = os.path.join(directory, 'RMSE_frequency.csv')
    with open(output_csv_file_name, 'w') as f:
        f.write('Qchem,' + ','.join(map(str, data1)) + '\n')
        f.write('Mycode,' + ','.join(map(str, data2)) + '\n')
        f.write('RMSE,' + str(rmse_values))

    print(f"RMSE for Frequencies saved to '{output_csv_file_name}'")

def compare_enthalpy_entropy(directory):
    enthalpy_entropy_file, rootname_enthalpy_entropy_file = find_enthalpy_entropy_files(directory)

    data1 = read_enthalpy_entropy_data(enthalpy_entropy_file,)  # Qchem 
    data2 = read_enthalpy_entropy_data(rootname_enthalpy_entropy_file)  # Mycode 

    rmse_values = root_mean_square_error(np.array(data1), np.array(data2))
    output_csv_file_name = os.path.join(directory, 'RMSE_enthalpy_entropy.csv')
    with open(output_csv_file_name, 'w') as f:
        f.write('Qchem,' + ','.join(map(str, data1)) + '\n')
        f.write('Mycode,' + ','.join(map(str, data2)) + '\n')
        f.write('RMSE,' + str(rmse_values))

    print(f"RMSE for Enthalpy and Entropy saved to '{output_csv_file_name}'")

def main():
    UseMsg = '''
    python [script] [target dir contains csv files] 
    '''
    parser = argparse.ArgumentParser(description="Find RMSE from csv file data.",usage=UseMsg)
    parser.add_argument('-t',"--target", required=True, help="directory containing the CSV files you want to compare")
   
    args = parser.parse_args()
    compare_frequency(args.target)
    compare_enthalpy_entropy(args.target)

if __name__ == "__main__":
    main()

