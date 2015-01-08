import csv
from itertools import product
import os
from StringIO import StringIO

from openml import apiconnector

from AutoML2015.models.autosklearn import get_configuration_space
from AutoSklearn.components.classification import _classifiers
from AutoSklearn.components.preprocessing import _preprocessors
from HPOlibConfigSpace.converters import pcs_parser

input_csv_file = "/home/feurerm/projects/openml/datasets/datasets_iteration002.csv"
output_dir = "/home/feurerm/ihome/projects/automl_competition_2015" \
             "/experiments/base"
smac = "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_08"

api = apiconnector.APIConnector()

datasets = []
with open(input_csv_file) as fh:
    reader = csv.reader(fh)
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        did = int(row[0])
        type_ = row[1]
        status = row[2]
        target = row[3]

        if row[4] in ["TRUE", "True", "1"]: target_correct = True
        elif row[4] in ["FALSE", "False", "", "?", "0"]: target_correct = False
        else: raise ValueError(row, row[4])

        ignore = row[5]
        if row[6] in ["TRUE", "True", "1"]: ignore_correct = True
        elif row[6] in ["FALSE", "False", "", "?", "0"]: ignore_correct = False
        else: raise ValueError(row, row[6])

        row_id = row[7]
        if row[8] in ["TRUE", "True", "1"]: row_id_correct = True
        elif row[8] in ["FALSE", "False", "", "?", "0"]: row_id_correct = False
        else: raise ValueError(row, row[8])

        if row[9] in ["TRUE", "True", "1"]: attributes_correct = True
        elif row[9] in ["FALSE", "False", "", "?", "0"]: attributes_correct = False
        else: raise ValueError(row, row[9])

        if row[10] in ["TRUE", "True", "1"]: use = True
        elif row[10] in ["FALSE", "False", "", "?", "0"]: use = False
        else: raise ValueError(row, row[10])

        # Sanity checks:
        if type_ not in ["binary.classification",
                         "multiclass.classification",
                         "multilabel.classification", "regression", "?", ""]:
            print "%s (row %d) is not a valid machine learning type."

        # Remove all datasets by Quan Sun
        """
        if did in [1024, 717, 724, 729, 731, 736, 743, 748, 750, 755, 762, 767,
                   774, 779, 786, 793, 798, 801, 806, 813, 818, 820, 825, 832,
                   837, 844, 849, 851, 863, 868, 870, 875, 882, 887, 894, 899,
                   902, 907, 914, 919, 921, 926, 933, 938, 940, 945, 957, 964,
                   969, 971, 976, 983, 988, 990, 995, 1002, 1007, 1014, 1019,
                   1021, 1026, 715, 722, 727, 734, 739, 741, 746, 753, 758, 760,
                   765, 772, 777, 784, 789, 791, 796, 804, 809, 811, 816, 823,
                   828, 830, 835, 842, 847, 854, 859, 861, 866, 873, 878, 880,
                   885, 892, 897, 900, 905, 912, 917, 924, 929, 931, 936, 943,
                   948, 950, 955, 962, 967, 974, 979, 981, 986, 993, 998, 1000,
                   1005, 1012, 1017, 713, 718, 720, 725, 732, 737, 744, 749,
                   751, 756, 763, 768, 770, 775, 782, 787, 794, 799, 802, 807,
                   814, 819, 821, 826, 833, 838, 840, 845, 852, 857, 864, 869,
                   871, 876, 888, 890, 895, 903, 908, 910, 915, 922, 927, 934,
                   939, 941, 946, 953, 958, 960, 965, 972, 977, 984, 989, 991,
                   996, 1003, 1008, 1010, 1015, 1022, 714, 719, 721, 726, 733,
                   738, 740, 745, 752, 757, 764, 769, 771, 776, 783, 788, 790,
                   795, 803, 808, 810, 815, 822, 827, 834, 839, 841, 846, 853,
                   858, 860, 865, 872, 877, 884, 889, 891, 896, 904, 909, 911,
                   916, 923, 928, 930, 935, 942, 947, 954, 959, 961, 966, 973,
                   978, 980, 985, 992, 997, 1004, 1009, 1011, 1016, 1023, 716,
                   723, 728, 730, 735, 742, 747, 754, 759, 761, 766, 773, 778,
                   780, 785, 792, 797, 800, 805, 812, 817, 824, 829, 831, 836,
                   843, 848, 850, 855, 862, 867, 874, 879, 881, 886, 893, 898,
                   901, 906, 913, 918, 920, 925, 932, 937, 944, 949, 951, 956,
                   963, 968, 970, 975, 982, 987, 994, 999, 1001, 1006, 1013,
                   1018, 1020, 1025]:
            continue
        """

        if type_ == "binary.classification" and status == "active" and \
                target_correct is True and ignore_correct is True and \
                row_id_correct is True and use is True:
            dataset = api.get_cached_dataset(did)


            qualities = api.download_dataset_qualities(did)['oml:quality']
            qualities = {q['oml:name']: q['oml:value'] for q in qualities}
            num_features = int(qualities["NumberOfFeatures"])
            num_instances = int(qualities["NumberOfInstances"])
            num_classes = int(qualities["NumberOfClasses"])

            if num_instances < 1000:
                continue
            print dataset.name, dataset.id, \
                "#features", num_features, \
                "#instances", num_instances, \
                "#classes", num_classes

            datasets.append(dataset)


classifiers = _classifiers
preprocessors = [p for p in _preprocessors if p not in
                 ["imputation", "rescaling"]] + ["None"]
metrics = ["bac_metric", "auc_metric", "f1_metric", "pac_metric"]

print "#datasets", len(datasets)
print "#classifiers", len(classifiers)
print "#preprocessors", len(preprocessors)
print "#metrics", len(metrics)
print len(datasets) * len(classifiers) * len(preprocessors) * len(metrics)

# Add possibility to restart by classifier or preprocessor...
for dataset in datasets:
    commands = []
    for classifier, preprocessor, metric in \
            product(classifiers, preprocessors, metrics):
        directory_name = "%d-%s-%s-%s" % (dataset.id,
                                          classifier,
                                          preprocessor,
                                          metric)
        output_dir_ = os.path.join(output_dir, directory_name)
        if not os.path.exists(output_dir_):
            os.mkdir(output_dir_)
        else:
            continue

        # Create configuration space, we can put everything in here because by
        # default it is not sparse and not multiclass and not multilabel
        configuration_space = get_configuration_space(
            {'task': "binary.classification", 'is_sparse': 0},
            include_classifiers=[classifier],
            include_preprocessors=[preprocessor])
        with open(os.path.join(output_dir_, "params.pcs"), "w") as fh:
            cs = pcs_parser.write(configuration_space)
            fh.write(cs)

        config = StringIO()
        config.write("[HPOLIB]\n")
        config.write("dispatcher = runsolver_wrapper.py\n")
        config.write("function = python -m AutoML2015.wrapper.openml_wrapper "
                     "--dataset %d --metric %s --task_type %s\n"
                     % (dataset.id, metric, "binary.classification"))
        config.write("number_of_jobs = 100\n")
        config.write("number_cv_folds = 10\n")
        config.write("runsolver_time_limit = 1800\n")
        config.write("memory_limit = 4000\n")
        config.write("result_on_terminate = 1.0\n")
        config.write("[SMAC]\n")
        config.write("runtime_limit = 172800\n")
        config.write("p = params.pcs\n")

        with open(os.path.join(output_dir_, "config.cfg"), "w") as fh:
            fh.write(config.getvalue())

        commands.append("HPOlib-run -o %s --cwd %s "
                        "--HPOLIB:temporary_output_directory "
                        "/tmp/${JOB_ID}.${SGE_TASK_ID}.aad_core.q"
                        % (smac, output_dir_))

    for seed in range(1000, 10001, 1000):
        with open(os.path.join(output_dir, "commands_seed-%d_did-%d.txt"
                % (seed, dataset.id)), "w") as fh:
            for command in commands:
                fh.write(command + " --seed %d --HPOLIB:optimizer_loglevel 10" % seed)
                fh.write("\n")
