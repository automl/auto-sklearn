#!/usr/bin/env python
import csv
import os
import itertools


def get_empty_iterator():
    return itertools.cycle([None])


def get_plot_markers():
    return itertools.cycle(['o', 's', 'x', '^', 'p', 'v', '>', '<', '8', '*',
                            '+', 'D'])


def get_plot_linestyles():
    return itertools.cycle(['-', '--', '-.', ':', ])


def get_single_linestyle():
    return itertools.cycle(['-'])


def get_plot_colors():
    # color brewer, 2nd qualitative 9 color scheme (http://colorbrewer2.org/)
    return itertools.cycle(["#000000",    # Black
                            "#e41a1c",    # Red
                            "#377eb8",    # Blue
                            "#4daf4a",    # Green
                            "#984ea3",    # Purple
                            "#ff7f00",    # Orange
                            "#ffff33",    # Yellow
                            "#a65628",    # Brown
                            "#f781bf",    # Pink
                            "#999999",    # Grey
                            ])


def get_defaults():
    default = {"linestyles": get_single_linestyle(),
               "colors": get_plot_colors(),
               "markers": get_empty_iterator(),
               "markersize": 6,
               "labelfontsize": 12,
               "linewidth": 1,
               "titlefontsize": 15,
               "gridcolor": 'lightgrey',
               "gridalpha": 0.5,
               "dpi": 100,
               "legendsize": 12
               }
    return default


def fill_with_defaults(def_dict):
    defaults = get_defaults()
    for key in defaults:
        if key not in def_dict:
            def_dict[key] = defaults[key]
        elif def_dict[key] is None:
            def_dict[key] = defaults[key]
        else:
            pass
    return def_dict


def read_csv(fn, has_header=True, data_type=str):
    data = list()
    header = None
    with open(fn, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if header is None and has_header:
                header = row
                continue
            data.append(list(map(data_type, [i.strip() for i in row])))
    return header, data


def get_file_and_name_list(argument_list, match_file, len_name=1):
    """
    argument_list: [<whatisthis> <file>*]*
    match_file: string which only appears in file and not in whatisthis
    len_name: len of names describing file(s) (if >1 return list of tuples)
    """
    assert 0 < len_name == int(len_name)
    name_list = list()
    file_list = list()
    len_desc = 0
    for i in range(len(argument_list)):
        # This if statement basically says all elems in arg_list should be a file.
        if match_file not in argument_list[i] and len_desc == len_name:
            # We have all names, but next argument is not a file
            raise ValueError("You need at least one %s file per Experiment, %s has none"
                             % (match_file, name_list[-1]))
        elif match_file not in argument_list[i] and len_desc < len_name:
            # We start with a new name desc   desc=description?
            if len_name > 1 and len_desc == 0:
                name_list.append(list([argument_list[i], ]))
            elif len_name > 1 and len_desc > 0:
                name_list[-1].append(argument_list[i])
            else:
                name_list.append(argument_list[i])

            len_desc += 1

            if len_desc == len_name:
                # We have all desc for this file
                file_list.append(list())
            continue
        else:
            if os.path.exists(argument_list[i]):
                len_desc = 0
                file_list[-1].append(os.path.abspath(argument_list[i]))
            else:
                raise ValueError("%s is not a valid file" % argument_list[i])

    return file_list, name_list
