import cPickle
import glob
import os
import yaml

from openml.apiconnector import APIConnector
import numpy as np

from AutoML2015.models.paramsklearn import get_configuration_space
import AutoML2015.metalearning.files
from HPOlibConfigSpace.configuration_space import Configuration
from HPOlibConfigSpace.hyperparameters import InstantiatedFloatHyperparameter, \
    InstantiatedIntegerHyperparameter, InstantiatedCategoricalHyperparameter, \
    InstantiatedConstant

openml_cache_dir = None
input_csv_file = "/home/feurerm/mhome/projects/automl_competition_2015" \
                 "/experiments/openml_dataset_phase1.csv"
#openml_cache_dir = "/home/feurerm/mhome/projects/automl_competition_2015/"
#input_csv_file = "/home/feurerm/mhome/projects/automl_competition_2015" \
#                 "/experiments/libsvm_datasets_phase1.csv"

output_dir = os.path.dirname(AutoML2015.metalearning.files.__file__)

experiment_dir = "/home/feurerm/mhome/projects/automl_competition_2015" \
                 "/experiments/metadata_for_phase_1"

datasets = []
with open(input_csv_file) as fh:
    for line in fh:
        datasets.append(int(float(line)))


output_list = []

api = APIConnector(authenticate=False, cache_directory=openml_cache_dir)

for did in datasets:
    dataset = api.get_cached_dataset(did)
    X, y, categorical = dataset.get_pandas(
        target=dataset.default_target_attribute, include_row_id=False,
        include_ignore_attributes=False)

    if np.sum(categorical) == len(categorical):
        print "Do not use dataset %d because it will be empty after removing " \
              "categorical variables." % dataset.id
        continue

    output_dict = {}
    output_dict['name'] = dataset.id
    output_dict['file'] = 'dataset/%d/description.xml' % dataset.id
    output_dict['metafeature_file'] = '%d.arff' % dataset.id
    output_list.append(output_dict)


with open(os.path.join(output_dir, 'datasets.yaml'), 'a') as fh:
    yaml.safe_dump(output_list, fh)


# TODO add regression
outputs_per_metric = {'bac_metric': list(),
                      'auc_metric': list(),
                      'f1_metric': list(),
                      'pac_metric': list()}

for did in datasets:
    dataset = api.get_cached_dataset(did)
    # For phase 1, do not use categorical features
    X, y, categorical = dataset.get_pandas(
        target=dataset.default_target_attribute, include_row_id=False,
        include_ignore_attributes=False)

    if np.sum(categorical) == len(categorical):
        print "Do not use dataset %d because it will be empty after removing " \
              "categorical variables." % dataset.id
        continue

    data_manager_info_dummy = {'task': 'binary.classification', 'is_sparse': 0}
    configuration_space = get_configuration_space(data_manager_info_dummy)
    directory_content = os.listdir(experiment_dir)

    # Create output lists for this dataset
    for key in outputs_per_metric:
        outputs_per_metric[key].append({'name': dataset.id,
                                        'experiments': list()})

    # Only directories can have experiment directories inside
    for dc in directory_content:
        dc = os.path.join(experiment_dir, dc)
        if not os.path.isdir(dc):
            continue

        dataset_exp_dir = os.path.join(dc, "%d" % dataset.id)
        if not os.path.isdir(dataset_exp_dir):
            continue

        # Find SMAC directory
        smac_pkls = glob.glob(os.path.join(dataset_exp_dir,
            "smac_2_08_00-master_*", "smac_2_08_00-master.pkl"))

        for smac_pkl in smac_pkls:
            with open(smac_pkl) as fh:
                runs = []
                trials = cPickle.load(fh)
                for trial in trials["trials"]:
                    test_result = trial['test_result']
                    if not np.isfinite(test_result):
                        continue
                    params = trial['params']
                    for key in params:
                        try:
                            params[key] = float(params[key])
                        except:
                            pass

                    break_ = False
                    for key in params:
                        if "sparse_filtering" in key:
                            break_ = True
                    if break_:
                        break

                    configuration = Configuration(configuration_space, **params)
                    parameters = {}
                    for ihp in configuration:
                        if isinstance(ihp, InstantiatedIntegerHyperparameter):
                            value_ = int(float(ihp.value))
                        elif isinstance(ihp, InstantiatedFloatHyperparameter):
                            value_ = float(ihp.value)
                        elif isinstance(ihp, InstantiatedCategoricalHyperparameter):
                            try:
                                if abs(int(float(ihp.value)) -
                                       float(ihp.value)) < 0.000001:
                                    value_ = int(ihp.value)
                                else:
                                    value_ = str(ihp.value)
                            except Exception as e:
                                #print e
                                value_ = str(ihp.value)
                        elif isinstance(ihp, InstantiatedConstant):
                            value_ = ihp.hyperparameter.value
                        else:
                            raise ValueError(ihp)
                        parameters[ihp.hyperparameter.name] = value_

                    # We only have the results for each of the CV folds,
                    # therefore we must average these values first.
                    values_per_metric = {metric_: list() for metric_ in
                                         outputs_per_metric}

                    # Extract the results of the different metrics
                    for key in trial['test_additional_data']:
                        additional_data = trial['test_additional_data'][key]
                        additional_data = additional_data.split(";")

                        for pair in additional_data:
                            pair = pair.strip()
                            try:
                                metric, value = pair.split(":")
                                metric = metric.strip()
                                value = value.strip()

                                if metric == 'duration':
                                    continue

                                values_per_metric[metric] = float(value)

                            except Exception as e:
                                print e
                                print smac_pkl
                                print trial
                                print

                    for metric in outputs_per_metric:
                        # Cast the duration to float, because otherwise
                        # it is a np.float, which cannot be
                        # represented by yaml
                        run = {'configuration': parameters,
                               'result': float(np.mean(values_per_metric[
                                               metric])),
                               'duration': float(trial["duration"])}
                        outputs_per_metric[metric][-1]['experiments'].\
                            append(run)

for metric in outputs_per_metric:
    with open(os.path.join(output_dir, "%s.experiments.yaml" % metric), "a") as fh:
        yaml.safe_dump(outputs_per_metric[metric], fh)
