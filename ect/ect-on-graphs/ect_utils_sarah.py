# Functions and utils for ECT computation
# Sarah McGuire Feb 2024

import os
from itertools import compress, combinations
import numpy as np
import math

from numba import jit

@jit
def find_combos(newV):
    #res = list(combinations(newV, 2))
    res = []
    n = len(newV)
    for i in range(n):
        for j in range(i+1, n):
            res.append((newV[i], newV[j]))
    return res

def lower_edges(v, G, pos_list, omega):
        """
        Function compute the number of lower edges of a vertex v for a specific direction (included by the use of sorted v_list).
        """
        L = [n for n in G.neighbors(v)]
        gv = np.dot(pos_list[v],omega)
        Lg = [np.dot(pos_list[v],omega) for v in L]
        return sum(n >= gv for n in Lg) # includes possible duplicate counts 

 

def ECC(G, pos, theta, r, numThresh):
    """
    Function to compute the Euler Characteristic of a graph with coordinates for each vertex (pos), 
    using a specified number of thresholds and bounding box defined by radius r.
    """
        
    omega = (np.cos(theta), np.sin(theta))
    
    # list of vertices and vertex positions
    v_list = list(pos.keys())
    pos_list = list(pos.values())

    # function g 
    def g(v): 
        return np.dot(pos_list[v],omega)
    # sort the v_list using g(v)
    v_list.sort(key=g, reverse = True) 

        
    def count_duplicate_edges(newV):
        """
        Function to count the number of duplicate counted edges from lower_edges. These duplicate edges are added to the EC value.
        """
        res = find_combos(newV)
        count=0
        for v,w in res:
            if G.has_edge(v,w) and g(v)==g(w):
                count+=1
        return count
    
    # thresholds for filtration, r should be defined from global bounding box
    r_threshes = np.linspace(r, -r, numThresh)
    
    # Full ECC vector
    ecc=[]
    ecc.append(0)

    
    for i in range(numThresh):

        #set of new vertices that appear in this threshold band
        if i==numThresh-1:
            newV =list(compress(v_list,[r_threshes[i]>g(v) for v in v_list]))
        else:
            newV =list(compress(v_list,[r_threshes[i]>g(v)>=r_threshes[i+1] for v in v_list]))
  
        x = ecc[i]#previous value of ECC (not index i-1 becuase of extra 0 appended to begining)
        if newV: # if there's new vertices to count
            v_count=0
            e_count=0
            for v in newV:
                k = lower_edges(v, G, pos_list, omega)
                v_count+=1 #add 1 to vertex count
                e_count+=k #subtract the the number of lower edges
            #check for duplicate edges counted
            dupl = count_duplicate_edges(newV)
            # after counting all new vertices and edges
            ecc.append(x+v_count-e_count+dupl)
        else:
            ecc.append(x)
    ecc = ecc[1:] #Drop the initial 0 value
    #print('ECC for direction', omega, '= ', ecc)
    
    return ecc


def bounding_box(points):
    """
    Function to find a bounding box of a set of points.
    """
    x_coord, y_coord = zip(*points)
    return [(min(x_coord), min(y_coord)), (max(x_coord), max(y_coord))]


# collect list of all points in dataset
def collect_all_points(mypath = '../data/ALLleaves'):
    global_pos_list = []
    #loop through file system
    for path, subdirs, files in os.walk(mypath):
        #print(subdirs)
        files = [f for f in files if not f[0] == '.']
        subdirs[:] = [d for d in subdirs if not d[0] == '.']
        
        for name in files:
            input_filedir = os.path.join(path, name)
            leaf = np.load(input_filedir)
            valuesX = leaf[:,0]
            valuesY = leaf[:,1]
            for i in range(np.shape(leaf)[0]):
                global_pos_list.append((valuesX[i],valuesY[i]))
    print('Loaded all data points, computing global bounding box...')
    x_box,y_box = zip(*bounding_box(global_pos_list)) # build a bounding box
    # bounding box size (use to get a radius for the bounding circle)
    print('Completed bounding box, starting ect computations...')
    dist = math.dist((x_box[0],y_box[0]),(x_box[1],y_box[1]))
    r = dist/2
    # bounding circle center
    #center = ((x_box[0]+x_box[1])/2, (y_box[0]+y_box[1])/2)
    return r


