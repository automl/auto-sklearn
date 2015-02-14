from csv import DictWriter
import numpy as np
import os


from pyMetaLearn.metafeatures import metafeatures
from pyMetaLearn.metafeatures.metafeature import DatasetMetafeatures
from AutoML2015.data.data_converter import predict_RAM_usage
from AutoML2015.wrapper.openml_wrapper import create_mock_data_manager, \
    remove_categorical_features
from openml.apiconnector import APIConnector


#input_csv_file = "/home/feurerm/mhome/projects/automl_competition_2015" \
#                 "/experiments/openml_dataset_phase1.csv"
input_csv_file = "/home/feurerm/mhome/projects/automl_competition_2015/" \
                 "experiments/libsvm_datasets_phase1.csv"
output_dir = "/home/feurerm/ihome/projects/automl_competition_2015" \
             "/experiments/metafeatures_for_phase_1"


datasets = []
with open(input_csv_file) as fh:
    for line in fh:
        datasets.append(int(line))

api = APIConnector(authenticate=False,
                   cache_directory="/home/feurerm/mhome/projects/automl_competition_2015/")

all_metafeatures = []
for did in datasets:
    commands = []

    dataset = api.get_cached_dataset(did)

    X, y, categorical = dataset.get_pandas(
        target=dataset.default_target_attribute, include_row_id=False,
        include_ignore_attributes=False)
    X = X.values
    y = y.values
    num_missing_values = np.sum(~np.isfinite(X))
    num_values = X.shape[0] * X.shape[1]
    sparsity = float(num_missing_values) / float(num_values)

    estimated_ram = float(predict_RAM_usage(X, categorical)) / 1024 / 1024
    sparse = False
    if np.sum(categorical) > 0 and estimated_ram > 1000:
        sparse = True

    if np.sum(categorical) == len(categorical):
        print "Do not use dataset %d because it will be empty after removing " \
              "categorical variables." % dataset.id
        continue

    mf_filename = os.path.join(output_dir, "%d.arff" % dataset.id)

    if os.path.exists(mf_filename):
        with open(mf_filename) as fh:
            mf = DatasetMetafeatures.load(fh)

    else:
        X, categorical = remove_categorical_features(X, categorical)
        D = create_mock_data_manager(X, y, categorical, None, None)
        mf = metafeatures.calculate_all_metafeatures(
            X, y, categorical, dataset.name)

        mf.dump(mf_filename)

    all_metafeatures.append(mf)

# Write the calculation times as a csv file to disc (can be viewed in
# LibreOffice calc afterwards)
calculation_times = dict()
for i, mf in enumerate(all_metafeatures):
    calculation_times[mf.dataset_name + "_" + str(i)] = dict()
    for value in mf.metafeature_values:
        calculation_times[mf.dataset_name + "_" + str(i)][value.name] = \
            value.time

csv_file = os.path.join(output_dir, "calculation_times.csv")
with open(csv_file, "w") as fh:
    dw = DictWriter(fh, ["name"] +
        sorted([mfv.name for mfv in all_metafeatures[0].metafeature_values]))
    dw.writeheader()
    for key in calculation_times:
        calculation_times[key]["name"] = key
        dw.writerow(calculation_times[key])