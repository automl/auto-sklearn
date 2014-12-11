# Score library for NUMPY arrays
# ChaLearn AutoML challenge

# For regression: 
# solution and prediction are vectors of numerical values of the same dimension

# For classification:
# solution = array(p,n) of 0,1 truth values, samples in lines, classes in columns
# prediction = array(p,n) of numerical scores between 0 and 1 (analogous to probabilities)

# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

from sklearn import metrics
import numpy as np
import scipy as sp
import os
from sklearn.preprocessing import *
from sys import stderr
from sys import version
swrite = stderr.write
from os import getcwd as pwd
from pip import get_installed_distributions as lib
from glob import glob
import platform
import psutil

if (os.name == "nt"):
    filesep = '\\'
else:
    filesep = '/'

# ========= Useful functions ==============

def read_array(filename):
    ''' Read array and convert to 2d np arrays '''
    array = np.genfromtxt(filename, dtype=float)
    if len(array.shape)==1:
        array = array.reshape( -1, 1 ) 
    return array
 
def sanitize_array(array):
    ''' Replace NaN and Inf (there should not be any!)'''
    a=np.ravel(array)
    maxi = np.nanmax((filter(lambda x: x != float('inf'), a))) # Max except NaN and Inf
    mini = np.nanmin((filter(lambda x: x != float('-inf'), a))) # Mini except NaN and Inf
    array[array==float('inf')]=maxi
    array[array==float('-inf')]=mini
    mid = (maxi + mini)/2
    array[np.isnan(array)]=mid
    return array
 
def normalize_array (solution, prediction):
    ''' Use min and max of solution as scaling factors to normalize prediction,
    then threshold it to [0, 1]. Binarize solution to {0, 1}. 
    This allows applying classification scores to all cases.
    In principle, this should not do anything to properly formatted 
    classification inputs and outputs.'''
    # Binarize solution
    sol=np.ravel(solution) # convert to 1-d array
    maxi = np.nanmax((filter(lambda x: x != float('inf'), sol))) # Max except NaN and Inf
    mini = np.nanmin((filter(lambda x: x != float('-inf'), sol))) # Mini except NaN and Inf
    if maxi == mini:
        print('Warning, cannot normalize')
        return [solution, prediction]
    diff = maxi - mini
    mid = (maxi + mini)/2.
    new_solution = np.copy(solution)
    new_solution[solution>=mid] = 1
    new_solution[solution<mid] = 0
    # Normalize and threshold predictions (takes effect only if solution not in {0, 1})
    new_prediction = (np.copy(prediction) - float(mini))/float(diff)
    new_prediction[new_prediction>1] = 1 # and if predictions exceed the bounds [0, 1]
    new_prediction[new_prediction<0] = 0
    # Make probabilities smoother
    #new_prediction = np.power(new_prediction, (1./10))
    return [new_solution, new_prediction]
    
def binarize_predictions(array, task='binary.classification'):
    ''' Turn predictions into decisions {0,1} by selecting the class with largest 
    score for multiclass problems and thresholding at 0.5 for other cases.'''
    # add a very small random value as tie breaker (a bit bad because this changes the score every time)
    # so to make sure we get the same result every time, we seed it    
    #eps = 1e-15
    #np.random.seed(sum(array.shape))
    #array = array + eps*np.random.rand(array.shape[0],array.shape[1])
    bin_array = np.zeros(array.shape)
    if (task != 'multiclass.classification') or (array.shape[1]==1): 
        bin_array[array>=0.5] = 1
    else:        
        sample_num=array.shape[0]
        for i in range(sample_num):
            j = np.argmax(array[i,:])
            bin_array[i,j] = 1        
    return bin_array

def acc_stat (solution, prediction):
	''' Return accuracy statistics TN, FP, TP, FN
     Assumes that solution and prediction are binary 0/1 vectors.'''
     # This uses floats so the results are floats
	TN = sum(np.multiply((1-solution), (1-prediction)))
	FN = sum(np.multiply(solution, (1-prediction)))
	TP = sum(np.multiply(solution, prediction))
	FP = sum(np.multiply((1-solution), prediction))
 	#print "TN =",TN
	#print "FP =",FP
	#print "TP =",TP
	#print "FN =",FN
	return (TN, FP, TP, FN) 
 
def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''    
    m=len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i=a.argsort()
    sa=a[i]
    # Find unique values
    uval=np.unique(a)     
    # Test whether there are ties 
    R=np.arange(m, dtype=float)+1 # Ranks with base 1
    if len(uval)!=m:
        # Average the ranks for the ties 
        oldval=sa[0]
        newval=sa[0]
        k0=0
        for k in range(1,m):
            newval=sa[k]
            if newval==oldval:
                # moving average
                R[k0:k+1]=R[k-1]*(k-k0)/(k-k0+1)+R[k]/(k-k0+1)
            else:
                k0=k;
                oldval=newval
    # Invert the index
    S=np.empty(m)
    S[i]=R
    return S
    
def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape)==0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0]/(j[0]+1.))*i[1]+(1./(j[0]+1))*j[1]), enumerate(x))[1]
    R=np.array(R)    
    if len(R.shape)==1: return average(R)
    if axis==1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))
            
 
# ======= All metrics used for scoring in the challenge ========

### REGRESSION METRICS (work on raw solution and prediction)
# These can be computed on all solutions and predictions (classification included)

def r2_metric(solution, prediction, task='regression'):
    ''' 1 - Mean squared error divided by variance '''
    mse = mvmean((solution-prediction)**2)
    var = mvmean((solution-mvmean(solution))**2)
    score = 1 - mse / var
    return mvmean(score)

def a_metric (solution, prediction, task='regression'):
    ''' 1 - Mean absolute error divided by mean absolute deviation '''
    mae = mvmean(np.abs(solution-prediction))         # mean absolute error
    mad = mvmean(np.abs(solution-mvmean(solution))) # mean absolute deviation
    score = 1 - mae / mad
    return mvmean(score)
 
### END REGRESSION METRICS 

### CLASSIFICATION METRICS (work on solutions in {0, 1} and predictions in [0, 1])
# These can be computed for regression scores only after running normalize_array

def bac_metric (solution, prediction, task='binary.classification'):
    ''' Compute the normalized balanced accuracy. The binarization and 
    the normalization differ for the multi-label and multi-class case. '''
    label_num = solution.shape[1]
    score = np.zeros(label_num)
    bin_prediction = binarize_predictions(prediction, task)
    [tn,fp,tp,fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum (eps, tp)
    pos_num = sp.maximum (eps, tp+fn)
    tpr = tp / pos_num # true positive rate (sensitivity)
    if (task != 'multiclass.classification') or (label_num==1):
        tn = sp.maximum (eps, tn)
        neg_num = sp.maximum (eps, tn+fp)
        tnr = tn / neg_num # true negative rate (specificity)
        bac = 0.5*(tpr + tnr)
        base_bac = 0.5     # random predictions for binary case
    else: 
        bac = tpr
        base_bac = 1./label_num # random predictions for multiclass case
    bac = mvmean(bac)     # average over all classes
    # Normalize: 0 for random, 1 for perfect
    score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    return score
    

def pac_metric (solution, prediction, task='binary.classification'):
    ''' Probabilistic Accuracy based on log_loss metric. 
    We assume the solution is in {0, 1} and prediction in [0, 1].
    Otherwise, run normalize_array.''' 
    debug_flag=False
    [sample_num, label_num] = solution.shape
    if label_num==1: task='binary.classification'
    eps = 1e-15
    the_log_loss = log_loss(solution, prediction, task)
    # Compute the base log loss (using the prior probabilities)    
    pos_num = 1.* sum(solution) # float conversion!
    frac_pos = pos_num / sample_num # prior proba of positive class
    the_base_log_loss = prior_log_loss(frac_pos, task)
    # Alternative computation of the same thing (slower)    
    # Should always return the same thing except in the multi-label case
    # For which the analytic solution makes more sense
    if debug_flag:
        base_prediction = np.empty(prediction.shape)
        for k in range(sample_num): base_prediction[k,:] = frac_pos
        base_log_loss = log_loss(solution, base_prediction, task)  
        diff = np.array(abs(the_base_log_loss-base_log_loss))
        if len(diff.shape)>0: diff=max(diff)
        if(diff)>1e-10: 
            print('Arrggh {} != {}'.format(the_base_log_loss,base_log_loss))
    # Exponentiate to turn into an accuracy-like score.
    # In the multi-label case, we need to average AFTER taking the exp 
    # because it is an NL operation
    pac = mvmean(np.exp(-the_log_loss)) 
    base_pac = mvmean(np.exp(-the_base_log_loss))
    # Normalize: 0 for random, 1 for perfect    
    score = (pac - base_pac) / sp.maximum(eps, (1 - base_pac))
    return score
 
def f1_metric (solution, prediction, task='binary.classification'):
    ''' Compute the normalized f1 measure. The binarization differs 
        for the multi-label and multi-class case. 
        A non-weighted average over classes is taken.
        The score is normalized.'''
    label_num = solution.shape[1]
    score = np.zeros(label_num)
    bin_prediction = binarize_predictions(prediction, task)
    [tn,fp,tp,fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    true_pos_num = sp.maximum (eps, tp+fn)
    found_pos_num = sp.maximum (eps, tp+fp)
    tp = sp.maximum (eps, tp)
    tpr = tp / true_pos_num      # true positive rate (recall)
    ppv = tp / found_pos_num     # positive predictive value (precision)
    arithmetic_mean = 0.5 * sp.maximum (eps, tpr+ppv)
    # Harmonic mean:
    f1 = tpr*ppv/arithmetic_mean
    # Average over all classes
    f1 = mvmean(f1)
    # Normalize: 0 for random, 1 for perfect
    if (task != 'multiclass.classification') or (label_num==1):
    # How to choose the "base_f1"?
    # For the binary/multilabel classification case, one may want to predict all 1.
    # In that case tpr = 1 and ppv = frac_pos. f1 = 2 * frac_pos / (1+frac_pos)
    #     frac_pos = mvmean(solution.ravel())
    #     base_f1 = 2 * frac_pos / (1+frac_pos)
    # or predict random values with probability 0.5, in which case
    #     base_f1 = 0.5
    # the first solution is better only if frac_pos > 1/3.
    # The solution in which we predict according to the class prior frac_pos gives
    # f1 = tpr = ppv = frac_pos, which is worse than 0.5 if frac_pos<0.5
    # So, because the f1 score is used if frac_pos is small (typically <0.1)
    # the best is to assume that base_f1=0.5
        base_f1 = 0.5
    # For the multiclass case, this is not possible (though it does not make much sense to
    # use f1 for multiclass problems), so the best would be to assign values at random to get 
    # tpr=ppv=frac_pos, where frac_pos=1/label_num
    else:
        base_f1=1./label_num
    score = (f1 - base_f1) / sp.maximum(eps, (1 - base_f1))
    return score
    
def auc_metric(solution, prediction, task='binary.classification'):
    ''' Normarlized Area under ROC curve (AUC).
    Return Gini index = 2*AUC-1 for  binary classification problems.
    Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
    for the predictions. If solution and prediction are not vectors, the AUC
    of the columns of the matrices are computed and averaged (with no weight).
    The same for all classification problems (in fact it treats well only the
    binary and multilabel classification problems).'''
    #auc = metrics.roc_auc_score(solution, prediction, average=None)
    # There is a bug in metrics.roc_auc_score: auc([1,0,0],[1e-10,0,0]) incorrect
    label_num=solution.shape[1]
    auc=np.empty(label_num)
    for k in range(label_num):
        r_ = tiedrank(prediction[:,k])
        s_ = solution[:,k]
        if sum(s_)==0: print('WARNING: no positive class example in class {}'.format(k+1))
        npos = sum(s_==1)
        nneg = sum(s_<1)
        auc[k] = (sum(r_[s_==1]) - npos*(npos+1)/2) / (nneg*npos)
    return 2*mvmean(auc)-1

    
### END CLASSIFICATION METRICS 
    
# ======= Specialized scores ========
# We run all of them for all tasks even though they don't make sense for some tasks
    
def nbac_binary_score(solution, prediction):
    ''' Normalized balanced accuracy for binary and multilabel classification '''
    return bac_metric (solution, prediction, task='binary.classification')
    
def nbac_multiclass_score(solution, prediction):
    ''' Multiclass accuracy for binary and multilabel classification '''
    return bac_metric (solution, prediction, task='multiclass.classification')
    
def npac_binary_score(solution, prediction):
    ''' Normalized balanced accuracy for binary and multilabel classification '''
    return pac_metric (solution, prediction, task='binary.classification')
    
def npac_multiclass_score(solution, prediction):
    ''' Multiclass accuracy for binary and multilabel classification '''
    return pac_metric (solution, prediction, task='multiclass.classification')

def f1_binary_score(solution, prediction):
    ''' Normalized balanced accuracy for binary and multilabel classification '''
    return f1_metric (solution, prediction, task='binary.classification')
    
def f1_multiclass_score(solution, prediction):
    ''' Multiclass accuracy for binary and multilabel classification '''
    return f1_metric (solution, prediction, task='multiclass.classification')
    
def log_loss(solution, prediction, task = 'binary.classification'):
    ''' Log loss for binary and multiclass. '''
    [sample_num, label_num] = solution.shape
    eps = 1e-15
    
    pred = np.copy(prediction) # beware: changes in prediction occur through this
    sol = np.copy(solution)
    if (task == 'multiclass.classification') and (label_num>1):
        # Make sure the lines add up to one for multi-class classification
        norma = np.sum(prediction, axis=1)
        for k in range(sample_num):
            pred[k,:] /= sp.maximum (norma[k], eps) 
        # Make sure there is a single label active per line for multi-class classification
        sol = binarize_predictions(solution, task='multiclass.classification')
        # For the base prediction, this solution is ridiculous in the multi-label case
    
    # Bounding of predictions to avoid log(0),1/0,...
    pred = sp.minimum (1-eps, sp.maximum (eps, pred))
    # Compute the log loss    
    pos_class_log_loss = - mvmean(sol*np.log(pred), axis=0)
    if (task != 'multiclass.classification') or (label_num==1):
        # The multi-label case is a bunch of binary problems.
        # The second class is the negative class for each column.
        neg_class_log_loss = - mvmean((1-sol)*np.log(1-pred), axis=0)
        log_loss = pos_class_log_loss + neg_class_log_loss
        # Each column is an independent problem, so we average.
        # The probabilities in one line do not add up to one.
        # log_loss = mvmean(log_loss) 
        # print('binary {}'.format(log_loss))
        # In the multilabel case, the right thing i to AVERAGE not sum
        # We return all the scores so we can normalize correctly later on
    else:
        # For the multiclass case the probabilities in one line add up one.
        log_loss = pos_class_log_loss
        # We sum the contributions of the columns.
        log_loss = np.sum(log_loss) 
        #print('multiclass {}'.format(log_loss))
    return log_loss
    
def prior_log_loss(frac_pos, task = 'binary.classification'):
    ''' Baseline log loss. For multiplr classes ot labels return the volues for each column'''
    eps = 1e-15   
    frac_pos_ = sp.maximum (eps, frac_pos)
    if (task != 'multiclass.classification'): # binary case
        frac_neg = 1-frac_pos
        frac_neg_ = sp.maximum (eps, frac_neg)
        pos_class_log_loss_ = - frac_pos * np.log(frac_pos_)
        neg_class_log_loss_ = - frac_neg * np.log(frac_neg_)
        base_log_loss = pos_class_log_loss_ + neg_class_log_loss_
        # base_log_loss = mvmean(base_log_loss)
        # print('binary {}'.format(base_log_loss))
        # In the multilabel case, the right thing i to AVERAGE not sum
        # We return all the scores so we can normalize correctly later on
    else: # multiclass case
        fp = frac_pos_ / sum(frac_pos_) # Need to renormalize the lines in multiclass case
        # Only ONE label is 1 in the multiclass case active for each line
        pos_class_log_loss_ = - frac_pos * np.log(fp)
        base_log_loss = np.sum(pos_class_log_loss_) 
    return base_log_loss
        
# sklearn implementations for comparison
def log_loss_(solution, prediction):
    return metrics.log_loss(solution, prediction)
    
def r2_score_(solution, prediction):
	return metrics.r2_score(solution, prediction)

def a_score_(solution, prediction):
	mad = float(mvmean(abs(solution-mvmean(solution)))) 
	return 1 - metrics.mean_absolute_error(solution, prediction)/mad
 
def auc_score_(solution, prediction):
    auc = metrics.roc_auc_score(solution, prediction, average=None)
    return mvmean(auc)
    
### SOME I/O functions
    
def ls(filename):
    return sorted(glob(filename))
 
def write_list(lst):
    for item in lst:
        swrite(item + "\n") 
        
def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def get_info (filename):
    ''' Get all information {attribute = value} pairs from the public.info file'''
    info={}
    with open (filename, "r") as info_file:
        lines = info_file.readlines()
        features_list = list(map(lambda x: tuple(x.strip("\'").split(" = ")), lines))
        for (key, value) in features_list:
            info[key] = value.rstrip().strip("'").strip(' ')
            if info[key].isdigit(): # if we have a number, we want it to be an integer
                info[key] = int(info[key])
    return info   

def show_io(input_dir, output_dir):  
	''' show directory structure and inputs and autputs to scoring program'''      
	swrite('\n=== DIRECTORIES ===\n\n')
	# Show this directory
	swrite("-- Current directory " + pwd() + ":\n")
	write_list(ls('.'))
	write_list(ls('./*'))
	write_list(ls('./*/*'))
	swrite("\n")
	
	# List input and output directories
	swrite("-- Input directory " + input_dir + ":\n")
	write_list(ls(input_dir))
	write_list(ls(input_dir + '/*'))
	write_list(ls(input_dir + '/*/*'))
	write_list(ls(input_dir + '/*/*/*'))
	swrite("\n")
	swrite("-- Output directory  " + output_dir + ":\n")
	write_list(ls(output_dir))
	write_list(ls(output_dir + '/*'))
	swrite("\n")
        
    # write meta data to sdterr
	swrite('\n=== METADATA ===\n\n')
	swrite("-- Current directory " + pwd() + ":\n")
	try:
		metadata = yaml.load(open('metadata', 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
	except:
		swrite("none\n");
	swrite("-- Input directory " + input_dir + ":\n")
	try:
		metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
		swrite("\n")
	except:
		swrite("none\n");
	
def show_version(scoring_version):
	''' Python version and library versions '''
	swrite('\n=== VERSIONS ===\n\n')
 	# Scoring program version
	swrite("Scoring program version: " + str(scoring_version) + "\n\n")
	# Python version
	swrite("Python version: " + version + "\n\n")
	# Give information on the version installed
	swrite("Versions of libraries installed:\n")
	map(swrite, sorted(["%s==%s\n" % (i.key, i.version) for i in lib()]))
 
def show_platform():
    ''' Show information on platform'''
    swrite('\n=== SYSTEM ===\n\n')
    try:
        linux_distribution = platform.linux_distribution()
    except:
        linux_distribution = "N/A"
    swrite("""
    dist: %s
    linux_distribution: %s
    system: %s
    machine: %s
    platform: %s
    uname: %s
    version: %s
    mac_ver: %s
    memory: %s
    number of CPU: %s
    """ % (
    str(platform.dist()),
    linux_distribution,
    platform.system(),
    platform.machine(),
    platform.platform(),
    platform.uname(),
    platform.version(),
    platform.mac_ver(),
    psutil.virtual_memory(),
    str(psutil.cpu_count())
    ))
 
def compute_all_scores(solution, prediction):
    ''' Compute all the scores and return them as a dist'''
    missing_score = -0.999999
    scoring = {'BAC (multilabel)':nbac_binary_score, 
               'BAC (multiclass)':nbac_multiclass_score, 
               'F1  (multilabel)':f1_binary_score, 
               'F1  (multiclass)':f1_multiclass_score, 
               'Regression ABS  ':a_metric, 
               'Regression R2   ':r2_metric, 
               'AUC (multilabel)':auc_metric, 
               'PAC (multilabel)':npac_binary_score, 
               'PAC (multiclass)':npac_multiclass_score}
    # Normalize/sanitize inputs
    [csolution, cprediction] = normalize_array (solution, prediction)
    solution = sanitize_array (solution); prediction = sanitize_array (prediction)
    # Compute all scores
    score_names = sorted(scoring.keys())
    scores = {}   
    for key in score_names:
        scoring_func = scoring[key] 
        try:
            if key=='Regression R2   ' or key=='Regression ABS  ':
                scores[key] = scoring_func(solution, prediction)
            else:
                scores[key] = scoring_func(csolution, cprediction)
        except:
            scores[key] = missing_score
    return scores
    
def write_scores(fp, scores):
    ''' Write scores to file opened under file pointer fp'''
    for key in scores.keys():
        fp.write("%s --> %s\n" % (key, scores[key]))
        print(key + " --> " + str(scores[key]))
        
def show_all_scores(solution, prediction):
    ''' Compute and display all the scores for debug purposes'''
    scores = compute_all_scores(solution, prediction)
    for key in scores.keys():
        print(key + " --> " + str(scores[key]))
 
############################### TEST PROGRAM ##########################################
if __name__=="__main__":
    # This shows a bug in metrics.roc_auc_score
#    print('\n\nBug in sklearn.metrics.roc_auc_score:')
#    print('auc([1,0,0],[1e-10,0,0])=1')
#    print('Correct (ours): ' +str(auc_metric(np.array([[1,0,0]]).transpose(),np.array([[1e-10,0,0]]).transpose())))
#    print('Incorrect (sklearn): ' +str(metrics.roc_auc_score(np.array([1,0,0]),np.array([1e-10,0,0]))))
    
    # This checks the binary and multi-class cases are well implemented 
    # In the 2-class case, all results should be identical, except for f1 because
    # this is a score that is not symmetric in the 2 classes.
    eps = 1e-15
    print('\n\nBinary score verification:')
    print('\n\n==========================')
    
    sol0 = np.array([[1, 0],[1, 0],[0, 1],[0, 1]])
    
    comment = ['PERFECT']
    Pred = [sol0]
    Sol = [sol0]
    
    comment.append('ANTI-PERFECT, very bad for r2_score')
    Pred.append(1-sol0)
    Sol.append(sol0)
    
    comment.append('UNEVEN PROBA, BUT BINARIZED VERSION BALANCED (bac and auc=0.5)') 
    Pred.append(np.array([[0.7, 0.3],[0.4, 0.6],[0.49, 0.51],[0.2, 0.8]])) # here is we have only 2, pac not 0 in uni-col
    Sol.append(sol0)
    
    comment.append('PROBA=0.5, TIES BROKEN WITH SMALL VALUE TO EVEN THE BINARIZED VERSION')
    Pred.append(np.array([[0.5+eps, 0.5-eps],[0.5-eps, 0.5+eps],[0.5+eps, 0.5-eps],[0.5-eps, 0.5+eps]]))
    Sol.append(sol0)
    
    comment.append('PROBA=0.5, TIES NOT BROKEN (bad for f1 score)')
    Pred.append(np.array([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]))
    Sol.append(sol0)
    
 
    sol1 = np.array([[1, 0],[0, 1],[0, 1]])
   
    comment.append('EVEN PROBA, but wrong PAC prior because uneven number of samples')  
    Pred.append(np.array([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]))
    Sol.append(sol1)
    
    comment.append('Correct PAC prior; score generally 0. But 100% error on positive class because of binarization so f1 (1 col) is at its worst.')
    p=len(sol1)
    Pred.append(np.array([sum(sol1)*1./p]*p))
    Sol.append(sol1)
    
    comment.append('All positive')
    Pred.append(np.array([[1, 1],[1, 1],[1, 1]]))
    Sol.append(sol1)
    
    comment.append('All negative') 
    Pred.append(np.array([[0, 0],[0, 0],[0, 0]]))
    Sol.append(sol1)
       
    for k in range(len(Sol)):
        sol = Sol[k]
        pred= Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        print('------ 2 columns ------')
        show_all_scores(sol, pred)
        print('------ 1 column  ------')
        sol=np.array([sol[:,0]]).transpose()
        pred=np.array([pred[:,0]]).transpose()
        show_all_scores(sol, pred)
        
    print('\n\nMulticlass score verification:')
    print('\n\n==========================')    
    sol2 = np.array([[1, 0, 0],[0, 1, 0],[1, 0, 0], [1, 0, 0]])
    
    comment = ['Three classes perfect']
    Pred = [sol2]
    Sol = [sol2]
    
    comment.append('Three classes all wrong')  
    Pred.append(np.array([[0, 1, 0],[0, 0, 1],[0, 1, 0], [0, 0, 1]]))
    Sol.append(sol2)
    
    comment.append('Three classes equi proba')  
    Pred.append(np.array([[1/3, 1/3, 1/3],[1/3, 1/3, 1/3],[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]))
    Sol.append(sol2)
    
    comment.append('Three classes some proba that do not add up')  
    Pred.append(np.array([[0.2, 0, 0.5],[0.8, 0.4, 0.1],[0.9, 0.1, 0.2], [0.7, 0.3, 0.3]]))
    Sol.append(sol2)
    
    comment.append('Three classes predict prior')  
    Pred.append(np.array([[ 0.75,  0.25,  0.  ],[ 0.75,  0.25,  0.  ],[ 0.75,  0.25,  0.  ], [ 0.75,  0.25,  0.  ]]))
    Sol.append(sol2)
    
    for k in range(len(Sol)):
        sol = Sol[k]
        pred= Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)
        
    print('\n\nMulti-label score verification: 1) all identical labels')
    print('\n\n=======================================================')
    print('\nIt is normal that for more then 2 labels the results are different for the multiclass scores.')
    print('\nBut they should be indetical for the multilabel scores.')
    num=2
        
    sol=np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0], [0, 0, 0]])
    sol3 = sol[:,0:num]
    if num==1: 
        sol3=np.array([sol3[:,0]]).transpose()
    
    comment = ['{} labels perfect'.format(num)]
    Pred = [sol3]
    Sol = [sol3]
    
    comment.append('All wrong, in the multi-label sense')  
    Pred.append(1-sol3)
    Sol.append(sol3)
    
    comment.append('All equi proba: 0.5')  
    sol=np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    if num==1:
        Pred.append(np.array([sol[:,0]]).transpose())
    else:
        Pred.append(sol[:,0:num])
    Sol.append(sol3)
    
    comment.append('All equi proba, prior: 0.25')   
    sol=np.array([[ 0.25,  0.25,  0.25  ],[ 0.25,  0.25,  0.25  ],[ 0.25,  0.25,  0.25  ], [ 0.25,  0.25,  0.25  ]])
    if num==1:
        Pred.append(np.array([sol[:,0]]).transpose())
    else:
        Pred.append(sol[:,0:num])
    Sol.append(sol3)

    comment.append('Some proba')  
    sol=np.array([[0.2, 0.2, 0.2],[0.8, 0.8, 0.8],[0.9, 0.9, 0.9], [0.7, 0.7, 0.7]])
    if num==1:
        Pred.append(np.array([sol[:,0]]).transpose())
    else:
        Pred.append(sol[:,0:num])
    Sol.append(sol3)
    
    comment.append('Invert both solution and prediction')  
    if num==1:
        Pred.append(np.array([sol[:,0]]).transpose())
    else:
        Pred.append(sol[:,0:num])
    Sol.append(1-sol3)
    
    for k in range(len(Sol)):
        sol = Sol[k]
        pred= Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)
        
    print('\n\nMulti-label score verification:')
    print('\n\n==========================')
        
    sol4 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1], [0, 0, 1]])
    
    comment = ['Three labels perfect']
    Pred = [sol4]
    Sol = [sol4]
    
    comment.append('Three classes all wrong, in the multi-label sense')  
    Pred.append(1-sol4)
    Sol.append(sol4)
    
    comment.append('Three classes equi proba')  
    Pred.append(np.array([[1/3, 1/3, 1/3],[1/3, 1/3, 1/3],[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]))
    Sol.append(sol4)
    
    comment.append('Three classes some proba that do not add up')  
    Pred.append(np.array([[0.2, 0, 0.5],[0.8, 0.4, 0.1],[0.9, 0.1, 0.2], [0.7, 0.3, 0.3]]))
    Sol.append(sol4)
    
    comment.append('Three classes predict prior')  
    Pred.append(np.array([[ 0.25,  0.25,  0.5  ],[ 0.25,  0.25,  0.5  ],[ 0.25,  0.25,  0.5  ], [ 0.25,  0.25,  0.5  ]]))
    Sol.append(sol4)
    
    for k in range(len(Sol)):
        sol = Sol[k]
        pred= Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)
        
