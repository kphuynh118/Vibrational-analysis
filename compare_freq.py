import pandas as pd
import numpy as np
import argparse
import os
import glob
#dir where two csv files you want to compare are located
#to get the freq out of qchem output file, use get_geom_freq --do_freq -t [dir] where [dir] contains the output files. Idealy [dir] should contains two files that have freq from qchem and freq from my code
def root_mean_square_error(data1, data2):
    """Calculate the Root Mean Square Error between two data sets."""
    return np.sqrt((data1 - data2) ** 2)

def read_data(file_path):
    """Read the data from a CSV file assuming it has only one line."""
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        # Handle trailing comma if present
        if line.endswith(','):
            line = line[:-1]
        data = line.split(',')[1:]  # Skip the first value (name) and split the rest
        data = [float(value) for value in data]  # Convert to float
    return data

def find_files(directory):
    """Find the specific files in the given directory."""
    frequency_file = os.path.join(directory, 'frequency.csv')
    matched_files = glob.glob(os.path.join(directory, '*_frequency.csv'))
    rootname_frequency_files = [file for file in matched_files if file != frequency_file]

    if not os.path.exists(frequency_file) or not rootname_frequency_files:
        raise FileNotFoundError("Required files not found in the directory.")

    return frequency_file, rootname_frequency_files[0]

def main():
    UseMsg = '''
    python [script] [dir contains 2 csv files] 
    '''
    parser = argparse.ArgumentParser(description="Find RMSE from csv file data.",usage=UseMsg)
    parser.add_argument('-t',"--target", required=True, help="directory containing the CSV files you want to compare")

    args = parser.parse_args()

    directory = args.target
    frequency_file, rootname_frequency_file = find_files(directory)

    data1 = read_data(frequency_file)  # Qchem 
    data2 = read_data(rootname_frequency_file)  # Mycode 

    # Calculate RMSE for each pair of data points
    rmse_values = root_mean_square_error(np.array(data1), np.array(data2))

    # Prepare and write the data to a new CSV file
    output_csv_file_name = os.path.join(directory, 'rmse.csv')
    with open(output_csv_file_name, 'w') as f:
        f.write('Qchem,' + ','.join(map(str, data1)) + '\n')
        f.write('Mycode,' + ','.join(map(str, data2)) + '\n')
        f.write('RMSE,' + ','.join(map(str, rmse_values)))

    print(f"Data and RMSE saved to '{output_csv_file_name}'")


if __name__ == "__main__":
    main()
