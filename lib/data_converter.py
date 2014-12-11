# Functions performing various data conversions for the ChaLearn AutoML challenge

# Main contributors: Arthur Pesah and Isabelle Guyon, August-October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import numpy as np
from scipy.sparse import *

# Note: to check for nan values np.any(map(np.isnan,X_train))

def file_to_array (filename, verbose=False):
	''' Converts a file to a list of list of STRING
	It differs from np.genfromtxt in that the number of columns doesn't need to be constant'''
	data =[]
	with open(filename, "r") as data_file:
		if verbose: print ("Reading {}...".format(filename))
		lines = data_file.readlines()
		if verbose: print ("Converting {} to correct array...".format(filename))
		data = [lines[i].strip().split() for i in range (len(lines))]
	return data

def read_first_line (filename):
	''' Read fist line of file'''
	data =[]
	with open(filename, "r") as data_file:
		line = data_file.readline()
		data = line.strip().split()
	return data  
 
def num_lines (filename):
	''' Count the number of lines of file'''
	return sum(1 for line in open(filename))

def binarization (array):
	''' Takes a binary-class datafile and turn the max value (positive class) into 1 and the min into 0'''
	array = np.array(array, dtype=float) # conversion needed to use np.inf after
	if len(np.unique(array)) > 2:
		raise ValueError ("The argument must be a binary-class datafile. {} classes detected".format(len(np.unique(array))))
	
	# manipulation which aims at avoid error in data with for example classes '1' and '2'.
	array[array == np.amax(array)] = np.inf
	array[array == np.amin(array)] = 0
	array[array == np.inf] = 1
	return np.array(array, dtype=int)

def sparse_file_to_sparse_list (filename, verbose=True):
	''' Converts a sparse data file to a sparse list, so that :
	sparse_list[i][j] = (a,b) means matrix[i][a]=b'''
	data_file = open(filename, "r")
	if verbose: print ("Reading {}...".format(filename))
	lines = data_file.readlines()
	if verbose: print ("Converting {} to correct array")
	data = [lines[i].split(' ') for i in range (len(lines))]
	if verbose: print ("Converting {} to sparse list".format (filename))
	return [[tuple(map(int, data[i][j].rstrip().split(':'))) for j in range(len(data[i])) if data[i][j] != '\n'] for i in range (len(data))] 

def sparse_list_to_csr_sparse (sparse_list, nbr_features, verbose=True):
	''' This function takes as argument a matrix of tuple representing a sparse matrix and the number of features. 
	sparse_list[i][j] = (a,b) means matrix[i][a]=b
	It converts it into a scipy csr sparse matrix'''
	nbr_samples = len(sparse_list)
	dok_sparse = dok_matrix ((nbr_samples, nbr_features)) # construction easier w/ dok_sparse...
	if verbose: print ("\tConverting sparse list to dok sparse matrix")
	for row in range (nbr_samples):
		for column in range (len(sparse_list[row])):
			(feature,value) = sparse_list[row][column]
			dok_sparse[row, feature-1] = value
	if verbose: print ("\tConverting dok sparse matrix to csr sparse matrix")
	return dok_sparse.tocsr() # ... but csr better for shuffling data or other tricks

def multilabel_to_multiclass (array):
	array = binarization (array)
	return np.array([np.nonzero(array[i,:])[0][0] for i in range (len(array))])
	
def convert_to_num(Ybin, verbose=True):
	''' Convert binary targets to numeric vector (typically classification target values)'''
	if verbose: print("\tConverting to numeric vector")
	Ybin = np.array(Ybin)
	if len(Ybin.shape) ==1:
         return Ybin
	classid=range(Ybin.shape[1])
	Ycont = np.dot(Ybin, classid)
	if verbose: print Ycont
	return Ycont
 
def convert_to_bin(Ycont, nval, verbose=True):
    ''' Convert numeric vector to binary (typically classification target values)'''
    if verbose: print ("\t_______ Converting to binary representation")
    Ybin=[[0]*nval for x in xrange(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])]=1
        Ybin[i] = line
    return Ybin
    
def tp_filter(X, Y, feat_num=1000, verbose=True):
    ''' TP feature selection in the spirit of the winners of the KDD cup 2001
    Only for binary classification and sparse matrices'''
        
    if issparse(X) and len(Y.shape)==1 and (sum(Y)/Y.shape[0])<0.1: 
        if verbose: print("========= Filtering features...")
        Posidx=Y>0
        #npos = sum(Posidx)
        #Negidx=Y<=0
        #nneg = sum(Negidx)
            
        nz=X.nonzero()
        mx=X[nz].max()
        if X[nz].min()==mx: # sparse binary
            if mx!=1: X[nz]=1
            tp=csr_matrix.sum(X[Posidx,:], axis=0)
            #fn=npos-tp
            #fp=csr_matrix.sum(X[Negidx,:], axis=0)
            #tn=nneg-fp
        else:
            tp=np.sum(X[Posidx,:]>0, axis=0)
            #tn=np.sum(X[Negidx,:]<=0, axis=0)
            #fn=np.sum(X[Posidx,:]<=0, axis=0)
            #fp=np.sum(X[Negidx,:]>0, axis=0)

        tp=np.ravel(tp)
        idx=sorted(range(len(tp)), key=tp.__getitem__, reverse=True)   
        return idx[0:feat_num]
    else:
        feat_num = X.shape[1]
        return range(feat_num)
    
def replace_missing(X):
    # This is ugly, but
    try:
        if X.getformat()=='csr':
            return X
    except:
        p=len(X)
        nn=len(X[0])*2
        XX = np.zeros([p,nn])
        for i in range(len(X)):
            line = X[i]
            line1 = [0 if np.isnan(x) else x for x in line]
            line2 = [1 if np.isnan(x) else 0 for x in line] # indicator of missingness
            XX[i] = line1 + line2
    return XX
