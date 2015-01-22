"""Create a csv file which contains all datasets IDs to use for metadata
generation in the automl phase1.

The input csv file must have the following columns:
* did: Dataset ID
* type: Task type; one of multiclass.classification,
  multilabel.classification, binary.classification
* status: OpenML status for that dataset
* target: target attribute for that dataset
* correct: If the target attribute is correct
* ignore: the ignore attributes of OpenML
* correct: If the ignore attribute is correct
* row_id: The row ID of the OpenML dataset
* correct: If the row ID is correct
* attributes_correct: Sometimes the attributes are marked to be of a wrong
  type. This happens mostly for categorical attributes, which are represented as
  numerical ones. This should be True if all attributes are correct.
* use?: True if this dataset is checked and can be safely used
* comment

It then outputs .csv file with a single column. This column contains the
OpenML dataset IDs which should be used in the first automl phase."""

from argparse import ArgumentParser
import csv
from openml.apiconnector import APIConnector


def get_dataset_list(fh):
    datasets = []
    api = APIConnector()
    reader = csv.reader(fh)

    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        did = int(row[0])
        type_ = row[1]
        status = row[2]
        target = row[3]

        if row[4] in ["TRUE", "True", "1"]:
            target_correct = True
        elif row[4] in ["FALSE", "False", "", "?", "0"]:
            target_correct = False
        else:
            raise ValueError(row, row[4])

        ignore = row[5]
        if row[6] in ["TRUE", "True", "1"]:
            ignore_correct = True
        elif row[6] in ["FALSE", "False", "", "?", "0"]:
            ignore_correct = False
        else:
            raise ValueError(row, row[6])

        row_id = row[7]
        if row[8] in ["TRUE", "True", "1"]:
            row_id_correct = True
        elif row[8] in ["FALSE", "False", "", "?", "0"]:
            row_id_correct = False
        else:
            raise ValueError(row, row[8])

        if row[9] in ["TRUE", "True", "1"]:
            attributes_correct = True
        elif row[9] in ["FALSE", "False", "", "?", "0"]:
            attributes_correct = False
        else:
            raise ValueError(row, row[9])

        if row[10] in ["TRUE", "True", "1"]:
            use = True
        elif row[10] in ["FALSE", "False", "", "?", "0"]:
            use = False
        else:
            raise ValueError(row, row[10])

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

        if type_ in ["binary.classification"] and status == "active" and \
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

    return datasets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="Path to the input .csv file.")
    parser.add_argument("output", help="Path of the output file.")
    args = parser.parse_args()

    with open(args.input) as fh:
        datasets = get_dataset_list(fh)

    with open(args.output, "w") as fh:
        for dataset in datasets:
            fh.write(str(dataset.id))
            fh.write("\n")