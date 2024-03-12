#!/usr/bin/env python
import numpy as np
from scipy.linalg import sqrtm
from sympy import *
from optparse import OptionParser
import os, re 
import argparse

def parse_xyz_file(filename):
    AtomList = []
    CoordList = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        l = re.search('^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
        if l!=None:
            AtomList.append(l.group(1))
            coord = float(l.group(2)),float(l.group(3)),float(l.group(4))
            CoordList.append(coord)
    fr.close()
    Coords = np.array(CoordList)
    return AtomList, Coords
def write_xyz_file(directory, outfile, AtomList, Coords):
    filepath = os.path.join(directory, outfile)
    with open(filepath, 'w') as fw:
        fw.write("%d\n" % len(AtomList))
        fw.write("\n")
        for iAtom in range(len(AtomList)):
            x, y, z = Coords[iAtom]
            fw.write("%-3s %15.10f %15.10f %15.10f\n" % (AtomList[iAtom], x, y, z))
def shift_one_coord(input_path, xyzfile, nameroot):
    AtomList, Coords = parse_xyz_file(xyzfile)  # Ensure this function is defined
    shifted_files = []

    for i in range(len(AtomList)):
        for j in range(3): 
            for sign in ['+', '-']:
                shifted_coords = np.copy(Coords)
                shift_amount = 0.001 if sign == '+' else -0.001
                shifted_coords[i][j] += shift_amount

                outfile = f"{nameroot}_shift_atom_{i+1}_coord_{j+1}_{sign}.xyz"
                write_xyz_file(input_path, outfile, AtomList, shifted_coords)
                shifted_files.append(outfile)

    return shifted_files
def main():
    UseMsg = '''
    python [script] [options] [xyzfile]
    '''
    parser = argparse.ArgumentParser(description="Make input file for single point geometry calculation.",usage=UseMsg)
    parser.add_argument("xyzfile", help="name of the XYZ file")
    parser.add_argument("-m","--method",default="B3LYP", help="the methods (density functional to use)")
    parser.add_argument("-b","--basis", default="aug-cc-pVTZ", help="the target basis (default: aug-cc-pVTZ)")
    parser.add_argument("-charge","--charge", type=int, default=0, help="total charge of the system (default: 0)")
    parser.add_argument("-mult", "--multiplicity", type=int, default=1, help="total multiplicity of the system (default: 1)")
    parser.add_argument("-o", "--output_path", default="get_forces_qchem_input_files", help="the directory to store the shifted XYZ files (default: get_forces_qchem_input_files)")
    parser.add_argument("-i", "--input_path", default="shifted_files", type=str, help="the directory storing the generated inputs (default: shifted_files)")
    parser.add_argument("-a", "--all", action='store_true', default=False, help="run all the xyz files under the xyz_path")
    args = parser.parse_args()

    if not os.path.isfile(args.xyzfile):
        print(f"Error: File {args.xyzfile} not found.")
        return

    nameroot = os.path.splitext(args.xyzfile)[0]
    os.makedirs(args.input_path, exist_ok=True) #make a dir for all shifted files
    os.makedirs(args.output_path, exist_ok=True) #make a dir for all the output from make_input_sp_geom
    shifted_files = shift_one_coord(args.input_path, args.xyzfile, nameroot)

    command = f"make_input_sp_geom --force -m {args.method} -b {args.basis} --charge {args.charge} --mult {args.multiplicity} -i {args.output_path} -a {args.input_path}"
    os.system(command)

if __name__ == "__main__":
    main()
