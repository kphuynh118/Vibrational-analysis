**make_input_sp_geom, get_forces_in_xyzformat, get_geom_freq** can be found at [python_qinput/master](https://github.com/YuezhiMao/python_qinput/tree/master)

**shift_coords.py** 
- creates shifted xyz files and save them in shifted_files 
- writes Qchem input files to get SCF energy (1st derivatives) and save Q-Chem input files for force calculation in a directory called get_forces_qchem_input_files
- to compile: `python shift_coords.py [options] [xyzfile] `          

Run all input files on Q-Chem at the same time and move all output files under one directory 

**get_freq.py** 
- reads in all output files, find the SCF energy and write new grad files
- copies all grad files and re-write name the files accordingly 
- saves all frequencies in a csv file under a directory called [rootname]_freq_compare
- saves all enthalpy and entropy in another csv file under a directory called [rootname]_freq_compare
- to compile: `python get_freq.py [option] [qchem force calculation output files dir] [xyzfile]`      

Use **parse_enthalpy_entropy.py** to parse values of enthalpy and entropy from output file for comparison purpose, it can be found at [python_qinput/luna](https://github.com/YuezhiMao/python_qinput/tree/luna)
Make sure that the Q-Chem's output file of frequency job is in [rootname]_freq_compare
- to compile: `python parse_enthalpy_entropy.py [rootname]_freq_compare`

**compare_results.py**
- compares frequencies generated from this program versus from Q-Chem and returns a frequency_RMSE.csv file
- compares enthalpy and entropy generated from this program versus from Q-Chem and returns a enthalpy_entropy_RMSE.csv file
- to compile: `python compare_results.py [rootname]_freq_compare`

