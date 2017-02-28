'''
Copyright 2017 Egor Kobylkin

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# A placeholder for a fully automated command-line wrapper for auto-sklearn binary classification
# It takes the dataset file as the only parameter and predicts in a 'yes or now' kind of situation
# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""
Created on Mon Feb 27 19:11:59 PST 2017
@author: Egor Kobylkin
"""

import time, os, psutil, math, pandas as pd, shutil, traceback
from time import time,sleep,strftime
import numpy as np
import multiprocessing

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.constants import *

import argparse

# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()

parser.add_argument('filename', nargs=1, help='pandas HDFS dataframe .h5 with cust_id, category and data columns')

args = parser.parse_args()

# if the memory limit is lower the model can fail and the whole process will crash
memory_limit = 15000 # MB
global max_classifier_time_budget
max_classifier_time_budget = 1200 # but 10 minutes is usually more than enough

def p(text):
    print ('[REDLY] '+text+" # "+strftime("%H:%M")+" #")

def time_single_estimator(clf_name, clf_class, X, y, max_clf_time):
    if ('libsvm_svc' == clf_name  # doesn't even scale to a 100k rows
            or 'qda' == clf_name ): # crashes
        return 0
    p(clf_name+" starting")
    default = clf_class.get_hyperparameter_search_space().get_default_configuration()
    clf=clf_class(**default._values)
    t0 = time()
    try:
        clf.fit(X,y)
    except Exception as e:
        print(e)
    classifier_time = time() - t0 # keep time even if classifier crashed
    print(clf_name+" training time: "+str(classifier_time))
    if max_clf_time.value < int(classifier_time):
        max_clf_time.value = int(classifier_time) 
    # no return statement here because max_clf_time is a managed object 

def max_estimators_fit_duration(X,y,max_classifier_time_budget,sample_factor=1):
    p("constructing preprocessor pipeline and transforming sample dataset")
    # we don't care about the data here but need to preprocess, otherwise the classifiers crash
    default_cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
                            include={ 'imputation': 'most_frequent'
                                        , 'rescaling': 'standardize' }
                                        ).get_default_configuration()
    preprocessor = SimpleClassificationPipeline(default_cs, random_state=42)
    preprocessor.fit(X,y)
    X_tr,dummy = preprocessor.pre_transform(X,y)

    p("running estimators on a subset")
    # going over all default classifiers used by auto-sklearn
    clfs=autosklearn.pipeline.components.classification._classifiers

    processes = []
    with multiprocessing.Manager() as manager:
        max_clf_time=manager.Value('i',3) # default 3 sec
        for clf_name,clf_class in clfs.items() :
            pr = multiprocessing.Process( target=time_single_estimator, name=clf_name
                    , args=(clf_name, clf_class, X_tr, y, max_clf_time))
            pr.start()
            processes.append(pr)
        for pr in processes:
            pr.join(max_classifier_time_budget) # will block for max_classifier_time_budget or
            # until the classifier fit process finishes. After max_classifier_time_budget 
            # we will terminate all still running processes here. 
            if pr.is_alive():
                p("terminating "+pr.name+" process due to timeout")    
                pr.terminate()
        result_max_clf_time=max_clf_time.value

    p("test classifier fit completed")
    
    per_run_time_limit = int(sample_factor*result_max_clf_time) 
    return max_classifier_time_budget if per_run_time_limit > max_classifier_time_budget else per_run_time_limit

def read_dataframe_h5(filename):
    with pd.HDFStore(filename,  mode='r') as store:
        df=store.select('data')
    p("read dataset from the store")
    return df

def x_y_dataframe_split(dataframe, id=False):
    p("dataframe split")
    X = dataframe.drop(['cust_id','category'], axis=1)
    y = pd.np.array(dataframe['category'], dtype='int')
    if id:
        row_id = dataframe['cust_id']
        return X,y,row_id
    else:
        return X,y

filename = str(args.filename[0])
dataframe = read_dataframe_h5(filename)
print(dataframe['category'].unique()) 
p("filling missing values with the most frequent values")
# we need to "protect" NAs here for the dataset separation later
dataframe['category'] = dataframe['category'].fillna(-1) 
dataframe = dataframe.fillna(dataframe.mode().iloc[0])
print(dataframe['category'].unique()) 
p("factorizing the X")    
# we need this list of original dtypes for the Autosklearn fit, create it before categorisation or split
col_dtype_dict = {c:( 'Numerical' if np.issubdtype(dataframe[c].dtype, np.number) else 'Categorical' )
                                     for c in dataframe.columns if c not in ['cust_id','category']}
# http://stackoverflow.com/questions/25530504/encoding-column-labels-in-pandas-for-machine-learning
# http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn?rq=1
# https://github.com/automl/auto-sklearn/issues/121#issuecomment-251459036
for c in dataframe.select_dtypes(exclude=[np.number]).columns:
    if c not in ['cust_id','category']:
        dataframe[c]=dataframe[c].astype('category').cat.codes
df_unknown = dataframe[ dataframe.category == -1 ]  # 'None' gets categorzized into -1
df_known = dataframe[ dataframe.category != -1 ] # preparing for multiclass labeling
del dataframe

X,y = x_y_dataframe_split(df_known)
per_run_time_limit = max_estimators_fit_duration(X.values,y,max_classifier_time_budget)

# this is how much time budget should be reserved for each autosklearn model run
print(str(per_run_time_limit))
