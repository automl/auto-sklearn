from collections import OrderedDict
from openml import apiconnector

api = apiconnector.APIConnector()

datasets = OrderedDict()
cached_datasets = api.get_list_of_cached_datasets()

for dataset_ in api.get_dataset_list():
    did = dataset_["did"]
    datasets[did] = dataset_

with open("/home/feurerm/projects/automl_competition_2015/datasets.csv", "w") as fh:
    fh.write("did,type,status,target,correct,ignore,correct,row_id,correct,"
             "attributes_correct,use?,comment\n")
    for i in range(1, 1168):
        did = i
        print i
        dataset_ = datasets.get(did)

        comment = ""

        if did in [273, 274, 1092, 292, 293, 350, 351, 354, 357, 373, 315]:
            comment = "Sparse format or string type"
        elif did in [489, 496]:
            comment = "data type"
        # Starting from 579, there are spaces as delimiters
        elif did in [1038, 1042, 1047, 1048, 1057, 1073, 1077, 1078, 1079,
                     1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088,
                     1095, 1100, 1037, 358, 486] + range(579, 659) + \
                    [664, 671, 673, 674, 676, 689, 691, 700, 704, 709, 1039,
                     1043, 1051, 1090, 1091, 1093, 1094, 1096, 1098, 1099]:
            comment = "Scipy.io.arff errors..."
        elif did in [1101, 310, 1102, 311, 1109, 316, 374, 376, 379, 380,
                           383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393,
                           394, 395, 396, 397, 398, 399, 400, 401, 545]:
            comment = "Decoding problems..."
        elif did in range(70, 79) + range(115, 163) + range(244, 273):
            comment = "Huge BNG files..."

        if not comment:
            if did in cached_datasets:
                dataset = api.get_cached_dataset(did)
                status = dataset_["status"]
                target = dataset.default_target_attribute
                ignore = dataset.ignore_attributes
                row_id = dataset.row_id_attribute
                use = ""
                comment = ""
            else:
                try:
                    dataset = api.download_dataset(did)
                    status = dataset_["status"]
                    target = dataset.default_target_attribute
                    ignore = dataset.ignore_attributes
                    row_id = dataset.row_id_attribute
                    use = ""
                    comment = ""
                    print "Downloaded dataset %d" %did
                except:
                    status = ""
                    target = ""
                    ignore = ""
                    row_id = ""
                    use = ""
                    comment = ""
        else:
            status = dataset_["status"] if dataset_ is not None else ""
            target = ""
            ignore = ""
            row_id = ""
            use = False

        fh.write("%d,,%s,%s,,%s,,%s,,,%s,%s\n" % (did, status, target, ignore,
                                                 row_id, str(use), comment))
