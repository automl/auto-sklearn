'''
Copyright 2017 Egor Kobylkin
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# -*- coding: utf-8 -*-
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""
Created on Mon Feb 27 19:11:59 PST 2017
@author: ekobylkin
"""

import time, os, psutil, math, pandas as pd, shutil, traceback
from time import time,sleep,strftime
import numpy as np
import multiprocessing

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.constants import *

from sklearn.cross_validation import train_test_split
import sklearn.metrics

import argparse

# https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()

parser.add_argument('filename', nargs=1, help='pandas HDFS dataframe .h5 with cust_id, category and data columns')

args = parser.parse_args()

work_dir = './redly_tmp'
result_filename = 'redly-os-result.csv'
atsklrn_tempdir=os.path.join(work_dir, 'atsklrn_tmp')
shutil.rmtree(atsklrn_tempdir,ignore_errors=True) # cleanup - remove temp directory

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

def define_pool_size(memory_limit):
    # some classifiers can use more than one core - so keep this at half memory and cores
    max_pool_size = int(math.ceil(psutil.virtual_memory().total / (memory_limit * 1000000)))
    half_of_cores = int(math.ceil(psutil.cpu_count()/2.0))
    return half_of_cores if max_pool_size > half_of_cores else max_pool_size

def calculate_time_left_for_this_task(pool_size,per_run_time_limit):
    half_cpu_cores = pool_size
    queue_factor = 30
    if queue_factor*half_cpu_cores < 100: # 100 models to test overall
        queue_factor=100/half_cpu_cores

    time_left_for_this_task = int(queue_factor*per_run_time_limit)
    return time_left_for_this_task

def spawn_autosklearn_classifier(X_train, y_train, seed, dataset_name, time_left_for_this_task, per_run_time_limit, feat_type):
    c = AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task, per_run_time_limit=per_run_time_limit,
            ml_memory_limit=memory_limit,
            shared_mode=True, tmp_folder=atsklrn_tempdir, output_folder=atsklrn_tempdir,
            delete_tmp_folder_after_terminate=False, delete_output_folder_after_terminate=False,
            initial_configurations_via_metalearning=0, ensemble_size=0,
            seed=seed)
    sleep(seed)
    try:
        p("Starting seed="+str(seed))
        c.fit(X_train, y_train, metric='f1_metric', feat_type=feat_type, dataset_name = dataset_name)
        p("####### Finished seed="+str(seed))
    except Exception:
        p("Exception in seed="+str(seed)+".  ")
        traceback.print_exc()
    raise


def train_multicore(X, y, feat_type, pool_size=1, per_run_time_limit=60):
    time_left_for_this_task = calculate_time_left_for_this_task(pool_size,per_run_time_limit)
    
    p("Overal run time is about " + str(2*math.ceil(time_left_for_this_task/60.0)) + " minute(s)")
    p("Max time allowance for a model " + str(math.ceil(per_run_time_limit/60.0)) + " minute(s)")

    processes = []
    for i in range(2,pool_size+2): # reserve seed 1 for the ensemble building
        seed = i
        pr = multiprocessing.Process( target=spawn_autosklearn_classifier
                , args=(X,y,i,'foobar',time_left_for_this_task,per_run_time_limit,feat_type))
        pr.start()
        processes.append(pr)
    for pr in processes:
        pr.join()
    
    p("Multicore fit completed")

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

# prepare a sample to measure approx classifier run time and select features
max_sample_size=100000 # so that the classifiers fit method completes in a reasonable time  
dataset_size=df_known.shape[0]

if dataset_size > max_sample_size :
    sample_factor = dataset_size/float(max_sample_size)
    p("sample factor ="+str(sample_factor))
    X_sample,y_sample = x_y_dataframe_split(df_known.sample(max_sample_size,random_state=42))
    p("factorizing the X_sample")    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=33000, random_state=42) # no need for larger test
else :
    sample_factor = 1
    X_sample,y_sample = X,y  
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

per_run_time_limit = max_estimators_fit_duration(X_train.values,y_train,max_classifier_time_budget,sample_factor)
p("per_run_time_limit="+str(per_run_time_limit))
pool_size = define_pool_size(memory_limit)    
p("Process pool size="+str(pool_size))
feat_type= [col_dtype_dict[c] for c in X.columns]
p("starting autosklearn classifiers fiting")
train_multicore(X_train.values, y_train, feat_type, pool_size, per_run_time_limit)

p("Building ensemble")
seed = 1
c = AutoSklearnClassifier(
    time_left_for_this_task=300,per_run_time_limit=150,ml_memory_limit=20240,ensemble_size=50,ensemble_nbest=200,
    shared_mode=True, tmp_folder=atsklrn_tempdir, output_folder=atsklrn_tempdir,
    delete_tmp_folder_after_terminate=False, delete_output_folder_after_terminate=False,
    initial_configurations_via_metalearning=0,
    seed=seed)
c.fit_ensemble(
    task = BINARY_CLASSIFICATION
    ,y = y_train
    ,metric = F1_METRIC
    ,precision = '32'
    ,dataset_name = 'foobar' 
    ,ensemble_size=10
    ,ensemble_nbest=15)

sleep(20)
p("Ensemble built")

p("Show models")
print(c.show_models())
p("Predicting")
y_hat = c.predict(X_test.values)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

if df_unknown.shape[0]==0:
    p("nothing to predict. Prediction dataset is empty.")
    exit()

p("Re-fitting on full known dataset. This can take long for a large set.")
try:
    c.refit(X.values, y)
except Exception as e:
    p("Refit failed, restarting")
    print(e)
    try:
        X=X.values
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        c.refit(X, y)
    except Exception as e:
        p("Second refit failed, exiting")
        print(e)
        exit()

X_unknown,y_unknown,row_id_unknown = x_y_dataframe_split(df_unknown, id=True) 
p("Predicting. This can take a long time for a large prediction set.")
y_pred = c.predict(X_unknown.values)
p("Prediction done")

# http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats
# https://github.com/automl/ChaLearn_Automatic_Machine_Learning_Challenge_2015/blob/master/004_yolanda.py
result_df = pd.DataFrame({'cust_id':row_id_unknown,'prediction':pd.Series(y_pred,index=row_id_unknown.index)})
p("Exporting the data")
result_df.to_csv(result_filename, index=False, header=True) 
p("Redly Script Completed!")

