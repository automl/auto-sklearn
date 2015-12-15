# -*- encoding: utf-8 -*-
cimport numpy as np
import numpy as np

import scipy.sparse


from libc.stdio cimport *


def log_function(*args):
    print(args)

'''    read_sparse_file (filename, num_features, num_points)

    Reads the data from a file containing sparse data into a scipy.sparse.csr_matrix (type float32).
    
    The number of features and the number of points have to be known ahead of time.
    Each pair is "index:value" and the row in the file equals the row in the matrix.
    The labels are assumed to be in a different file, in contrast to the SVMlight file format where the "target value" would be the first entry    in each line!
    The optional argument initial_length should estimate the number of elements    expected in the data set. If it is to small all arrays will be dynamically enlarged in steps of initial_length. If chosen to small, unnecessary reallocations slow things down. If to big, memory might be wasted while    loading the data.
    The optional argument offset should be -1 if the indices in the file start at 1! This is the case for the automl challenge, hence the default value.
    So far, only ascii #32 (space) is recognized as a whitespace. If the entries are tab-separated (or any other chararcter), this could easily be implemented here.
    
'''
def read_sparse_file(filename, int num_points,int num_features, int initial_length = 8192, int offset = -1, double max_memory_in_mb = 1048576):

    #cdef np.ndarray[float, ndim=1] 
    data = np.zeros(initial_length,dtype=np.float32)
    #cdef np.ndarray[int, ndim=1]
    indices = np.zeros(initial_length, dtype=np.int32)
    #cdef np.ndarray[int, ndim=1] 
    indptr = np.zeros(num_points+1, dtype=np.int32)

    # we have to dynamically enlarge the arrays, so we need to keep track of how many entries we already have
    cdef int num_entries = 0
    
    
    # variables for I/O
    cdef char* fname
    cdef FILE* cfile
    cdef ssize_t read
    cdef char whitespace

    #variables for the indices and values
    cdef int i=0,j
    cdef float v


    filename_byte_string = filename.encode("UTF-8")
    fname = filename_byte_string
    cfile = fopen(fname, "r")
    if cfile == NULL:
        raise RuntimeError("Couldn't find file {}".format(filename))

    while True:
        # read the column and the value and store it
        read =fscanf(cfile, "%i:%f",&j,&v)
        # stop at EOF
        if read == -1:
            break

        data[num_entries] = v
        indices[num_entries] = j+offset
        num_entries += 1
        #enlarge the array if necessary
        if num_entries == data.shape[0]:
            if ((data.nbytes + indices.nbytes) < max_memory_in_mb*1024*1024):
                data.resize(data.shape[0]+initial_length)
                indices.resize(data.shape[0])
            else:
                break
        
        # check if we hit a endline next to recognize the next row
        # It is cumbersome, but a way to reliably do it!
        whitespace = fgetc(cfile)
        while whitespace == 32:
            whitespace = fgetc(cfile)

        if whitespace == '\n':
            i+=1
            # stop if num_points have been read
            if i >= num_points:
                break
            indptr[i] = num_entries
        else:
            ungetc(whitespace, cfile)

    fclose(cfile)
    
    
    #cut arrays to size
    data.resize(num_entries)
    indices.resize(num_entries)
    
    # fix the end of indptr
    for j in range (i,num_points+1):
        indptr[j] = num_entries
    return (scipy.sparse.csr_matrix((data, indices, indptr),
                                    shape=[num_points, num_features]))



'''
    see read_sparse_file, only difference: the value of every index present is 1, so there are no index:value pairs, but just indices.
    
'''
def read_sparse_binary_file(filename, int num_points, int num_features, int initial_length = 8192, int offset = -1, double max_memory_in_mb = 1048576):

    data = np.zeros(initial_length,dtype=np.bool)
    indices = np.zeros(initial_length, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] indptr = np.zeros(num_points+1, dtype=np.int32)
    
    # we have to dynamically enlarge the arrays, so we need to keep track of how many entries we already have
    cdef int num_entries = 0
    
    # variables for I/O
    cdef char* fname
    cdef FILE* cfile
    cdef ssize_t read
    cdef char whitespace

    #variables for the indices and values
    cdef int i=0,j

    filename_byte_string = filename.encode("UTF-8")
    fname = filename_byte_string
    cfile = fopen(fname, "r")
    if cfile == NULL:
        raise RuntimeError("Couldn't find file {}".format(filename))

    while True:
        # read the column and the value and store it
        read =fscanf(cfile, "%d",&j)
        # stop at EOF
        if read == -1:
            break

        data[num_entries] = True
        indices[num_entries] = j+offset
        num_entries += 1
        
        #enlarge the array if necessary
        if num_entries == data.shape[0]:
            if ((data.nbytes + indices.nbytes) < max_memory_in_mb*1024*1024):
                data.resize(data.shape[0]+initial_length)
                indices.resize(data.shape[0])
            else:
                break
        
        # check if we hit a endline next to recognize the next row
        # It is cumbersome, but a way to reliably do it!
        whitespace = fgetc(cfile)
        while (whitespace==32):
            whitespace = fgetc(cfile)

        if whitespace == '\n':
            i+=1
            # stop if num_points have been read
            if (i >=num_points): break
            indptr[i] = num_entries
        else:
            ungetc(whitespace, cfile)

    fclose(cfile)
    
    
    #cut arrays to size
    data.resize(num_entries)
    indices.resize(num_entries)
    
    # fix the end of indptr
    for j in range (i,num_points+1):
        indptr[j] = num_entries

    return (scipy.sparse.csr_matrix((data, indices, indptr),
                                    shape=[num_points, num_features],
                                    dtype=np.bool))


'''    read_dense_file (filename, num_features, num_points)

    Reads the data from a file containing dense data into a numpy array (type float32)
    
    The number of features and the number of points have to be known ahead of time.
    
    The function does not check for EOF or missing values, so be cautious!
'''
def read_dense_file(filename, int num_points, int num_features,double max_memory_in_mb = 1048576):

    nbits = np.finfo(np.float32).nexp + np.finfo(np.float32).nmant+1
    num_points = long(min(num_points,max_memory_in_mb*1024*1024*8/nbits/num_features))
    
    cdef np.ndarray[float, ndim=2] data = np.zeros([num_points, num_features],dtype=np.float32)
    
    # variables for I/O
    cdef char* fname
    cdef FILE* cfile

    #variable for the indices and values
    cdef int i=0,j=0
    cdef float v

    filename_byte_string = filename.encode("UTF-8")
    fname = filename_byte_string
    cfile = fopen(fname, "r")
    if cfile == NULL:
        raise RuntimeError("Couldn't find file {}".format(filename))

    for i in range(num_points):
        for j in range(num_features):
            fscanf(cfile, "%f",&v)
            data[i,j] = v
    fclose(cfile)

    return(data)


def read_dense_file_unknown_width(filename, num_points, max_memory_in_mb = 1048576):
    # variables for I/O
    cdef char* fname
    cdef FILE* cfile
    cdef int rc;

    #variables for the indices and values
    cdef int num_cols=0

    filename_byte_string = filename.encode("UTF-8")
    fname = filename_byte_string
    cfile = fopen(fname, "r")
    if cfile == NULL:
        raise RuntimeError("Couldn't find file {}".format(filename))


    #count the number of columns in the first line
    rc = fgetc(cfile)
    while True:
        # read a column
        while (rc != ' ') and (rc != '\n'):
            rc=fgetc(cfile)
        
        while (rc == ' '):
            rc=fgetc(cfile)
        num_cols+=1
        
        if (rc == '\n'): break
    fclose(cfile)

    data = read_dense_file(filename, num_points,num_cols, max_memory_in_mb)

    # if only one predictor is present, convert it into a 1D array
    if data.shape[1] == 1:
        return (data.flatten())

    return (data)


# function copied from the reference implementation
# no need to really optimize them because they don't take long

def read_first_line (filename):
    """
    Read fist line of file
    :param filename:
    :return:
    """
    data =[]
    with open(filename, "r") as data_file:
        line = data_file.readline()
        data = line.strip().split()
    return data  


def file_to_array (filename, verbose=False):
    """
    Converts a file to a list of list of STRING
    It differs from np.genfromtxt in that the number of columns doesn't need to be constant
    :param filename:
    :param verbose:
    :return:
    """

    data =[]
    with open(filename, "r") as data_file:
        if verbose:
            log_function("Reading {}...".format(filename))
        lines = data_file.readlines()

        if verbose:
            log_function("Converting {} to correct array...".format(filename))

        data = [lines[i].strip().split() for i in range (len(lines))]
    return data
