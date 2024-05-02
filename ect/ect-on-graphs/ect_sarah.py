# ECT computation for embedded graphs
# Sarah McGuire Feb 2024


import argparse
import numpy as np
import networkx as nx

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path



# include functions from ect_utils
from ect_utils import ECC, collect_all_points


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--species', type=str, default='all',
    help='the species directory to compute ect of')
args = vars(parser.parse_args())

if args['species']=='all':
    #mypath = '../data/ALLleaves/'
    #mypath_output = '../data/ALLleaves_ECT/'
    #print('Using all species folders', mypath)
    mypath = '../data/MPEG7original'
    mypath_output = '../data/MPEG7original_ECT/'
    print('Using MPEG data', mypath)
else:
    mypath = os.path.join('../data/ALLleaves/', args['species'])
    mypath_output = os.path.join('../data/ALLleaves_ECT/', args['species'])
    print('Using only species ' , args['species'] ,' folder', mypath)


# Compute bounding box
r = collect_all_points(mypath = '../data/ALLleaves')
print('r=',r)
#r = 2.9092515639765497
# ADD in list of species already computed ECT for
#done_species = ['Alstroemeria', 'Apple', 'Ivy', 'Passiflora']
done_species = []


#mypath = '../data/ALLleaves/Alstroemeria'
# loop through file system
classes=[]
for path, subdirs, files in os.walk(mypath):
    classes.extend(subdirs)
    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if (d[0] != '.' and d not in done_species)]
    print('Computing ECT of files in ', path, '...')
    print(len(files))
    for name in files:
        input_filedir = os.path.join(path, name)
        output_filedir = os.path.join(mypath_output+ input_filedir[len(mypath):])
    
        sample = np.load(input_filedir)
    
        # Create graph of sample outline
        G = nx.Graph()
        for i in range(np.shape(sample)[0]-1):
            G.add_edge(i, i+1)
        G.add_edge(0,np.shape(sample)[0]-1)

        # Get the vertex positions
        pos = {}
        valuesX = sample[:,0]
        valuesY = sample[:,1]
        for i in range(np.shape(sample)[0]):
            pos[i] = (valuesX[i],valuesY[i])

        # Select directions around the circle
        numCircleDirs = 32
        circledirs =  np.linspace(0, 2*np.pi, num=numCircleDirs, endpoint=False)

        # Choose number of thresholds for the ECC
        numThresh = 48

        # Compute the ECT of sample p for numCircleDirs, numThresh
        ECT_preprocess = {}
        for i, angle in enumerate(circledirs):

            outECC = ECC(G, pos, theta=angle, r=r, numThresh = numThresh)

            ECT_preprocess[i] = (angle,outECC)

        # Making a matrix M[i,j]: (numThresh x numCircleDirs)
        M = np.empty([numThresh,numCircleDirs])
        for j in range(M.shape[1]):
            M[:,j] = ECT_preprocess[j][1]


        # NPY file to save
        Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)
        np.save(output_filedir, M) 

    print('----------------\n completed subdirectory ', path ,'\n-------------', )