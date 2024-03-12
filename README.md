shift_coords.py - create shifted xyz files and save them in shifted_files
                - write Qchem input files to get SCF energy (1st derivatives) and save them in get_forces_qchem_input_files
    To compile: python shift_coords.py [options] [xyzfile]            

Run all input files on Qchem and move all output files under one directory 

get_frequecy.py - read in all output files, find the SCF energy and write new .grad files 
                - copy all grad files and write name the files accordingly 
                - print out the frequency
    To compile: python get_frequecy.py [qchem output files dir] [xyzfile]
