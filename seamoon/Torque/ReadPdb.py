aa_3_1 = {'ALA' : 'A', 'CYS' : 'C', 'ASP' : 'D', 'GLU' : 'E', 'PHE' : 'F', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LYS' : 'K', 'LEU' : 'L', 'MET' : 'M', 'ASN' : 'N', 'PRO' : 'P', 'GLN' : 'Q', 'ARG' : 'R', 'SER' : 'S', 'THR' : 'T', 'VAL' : 'V', 'TRP' : 'W', 'TYR' : 'Y',}
import numpy as np

def read_pdb(path_file):
    
    with open(path_file, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        
        positions = [[]]
        for i in range(len(lines)):
            if len(lines[i])!=0 and (lines[i][0]=="ATOM" or lines[i][0]=="HETATM") and (lines[i][2]=="CA" or lines[i][2]=="PCA"):
                positions[-1].append([float(lines[i][6]), float(lines[i][7]), float(lines[i][8])])
            if lines[i][0]=="ENDMDL":
                positions.append([])
    return positions