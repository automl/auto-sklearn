from argparse import ArgumentParser
from collections import OrderedDict
import os
import shlex
import subprocess

import HPOlibConfigSpace.hyperparameters
import ParamSklearn.classification
import ParamSklearn.regression

# Some macros
COND = "conditional"
CAT = "categorical"
CONT = "continuous"
CONST = "constant"
UN = "unparameterized"

template_string = \
"""
\documentclass{article} %% For LaTeX2
\usepackage[a4paper, left=5mm, right=5mm, top=5mm, bottom=5mm]{geometry}

%%\\usepackage[landscape]{geometry}
\\usepackage{multirow}           %% import command \multicolmun
\\usepackage{tabularx}           %% Convenient table formatting
\\usepackage{booktabs}           %% provides \\toprule, \midrule and \\bottomrule

\\begin{document}

%s

\\end{document}
"""

caption_str = "Number of Hyperparameters for each possible %s " \
              "for a dataset with these properties: %s"

table_str = \
"""
\\begin{table}[t!]
\\centering
\\scriptsize
\\caption{ %s }
\\begin{tabularx}{\\textwidth}{ X X X X X X }
\\toprule
name & \#$\lambda$ & cat (cond) & cont (cond) & const & un \\\\
\\toprule
\\\\
%s
\\\\
\\toprule
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def get_dict(task_type="classifier", **kwargs):
    assert task_type in ("classifier", "regressor")

    if task_type == "classifier":
        cs = ParamSklearn.classification.ParamSklearnClassifier.get_hyperparameter_search_space(dataset_properties=kwargs)
    elif task_type == "regressor":
        cs = ParamSklearn.regression.ParamSklearnRegressor.get_hyperparameter_search_space(dataset_properties=kwargs)
    else:
        raise ValueError("'task_type' is not in ('classifier', 'regressor')")

    preprocessor = None
    estimator = None

    for h in cs.get_hyperparameters():
        if h.name == "preprocessor:__choice__":
            preprocessor = h
        elif h.name == (task_type + ':__choice__'):
            estimator = h

    if estimator is None:
        raise ValueError("No classifier found")
    elif preprocessor is None:
        raise ValueError("No preprocessor found")

    estimator_dict = OrderedDict()
    for i in sorted(estimator.choices):
        estimator_dict[i] = OrderedDict()
        estimator_dict[i][COND] = OrderedDict()
        for t in (CAT, CONT, CONST):
            estimator_dict[i][t] = 0
            estimator_dict[i][COND][t] = 0
        estimator_dict[i][UN] = 0

    preprocessor_dict = OrderedDict()
    for i in sorted(preprocessor.choices):
        preprocessor_dict[i] = OrderedDict()
        preprocessor_dict[i][COND] = OrderedDict()
        for t in (CAT, CONT, CONST):
            preprocessor_dict[i][t] = 0
            preprocessor_dict[i][COND][t] = 0
        preprocessor_dict[i][UN] = 0

    for h in cs.get_hyperparameters():
        if h.name == "preprocessor:__choice__" or \
                h.name == (task_type + ':__choice__'):
            continue
        # walk over both dicts
        for d in (estimator_dict, preprocessor_dict):
            est = h.name.split(":")[1]
            if est not in d:
                continue
            if isinstance(h, HPOlibConfigSpace.hyperparameters.UniformIntegerHyperparameter):
                d[est][CONT] += 1
            elif isinstance(h, HPOlibConfigSpace.hyperparameters.UniformFloatHyperparameter):
                d[est][CONT] += 1
            elif isinstance(h, HPOlibConfigSpace.hyperparameters.CategoricalHyperparameter):
                d[est][CAT] += 1
            elif isinstance(h, HPOlibConfigSpace.hyperparameters.Constant):
                d[est][CONST] += 1
            elif isinstance(h, HPOlibConfigSpace.hyperparameters.UnParametrizedHyperparameter):
                d[est][UN] += 1
            else:
                raise ValueError("Don't know that type: %s" % type(h))

    for h in cs.get_conditions():
        if h.parent.name == (task_type + ':__choice__') or h.parent.name == \
                "preprocessor:__choice__":
            # ignore this condition
            # print "IGNORE", h
            continue

        # walk over both dicts and collect hyperparams
        for d in (estimator_dict, preprocessor_dict):
            est = h.child.name.split(":")[1]
            if est not in d:
                #print "Could not find %s" % est
                continue

            #print "####"
            #print vars(h)
            #print h.parent
            #print type(h)
            if isinstance(h.child, HPOlibConfigSpace.hyperparameters.UniformIntegerHyperparameter):
                d[est][COND][CONT] += 1
            elif isinstance(h.child, HPOlibConfigSpace.hyperparameters.UniformFloatHyperparameter):
                d[est][COND][CONT] += 1
            elif isinstance(h.child, HPOlibConfigSpace.hyperparameters.CategoricalHyperparameter):
                d[est][COND][CAT] += 1
            elif isinstance(h.child, HPOlibConfigSpace.hyperparameters.Constant):
                d[est][COND][CONST] += 1
            elif isinstance(h.child, HPOlibConfigSpace.hyperparameters.UnParametrizedHyperparameter):
                d[est][COND][UN] += 1
            else:
                raise ValueError("Don't know that type: %s" % type(h))
    print preprocessor_dict
    return (estimator_dict, preprocessor_dict)


def build_table(d):
    lines = list()
    for est in d.keys():
        sum_ = 0
        t_list = list([est.replace("_", " "), ])
        for t in (CAT, CONT):
            sum_ += d[est][t]
            t_list.append("%d (%d)" % (d[est][t], d[est][COND][t]))
        t_list.append("%d" % d[est][CONST])
        t_list.append("%d" % d[est][UN])
        sum_ += d[est][CONST] + d[est][UN]
        t_list.insert(1, "%d" % sum_)
        lines.append(" & ".join(t_list))
    return "\\\\ \n".join(lines)


def main():
    parser = ArgumentParser()

    # General Options
    parser.add_argument("-s", "--save", dest="save", default=None,
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--type", dest="task_type", default="classifier",
                        choices=("classifier", ), help="Type of dataset")
    parser.add_argument("--sparse", dest="sparse", default=False,
                        action="store_true", help="dataset property")
    prop = parser.add_mutually_exclusive_group(required=True)
    prop.add_argument("--multilabel", dest="multilabel", default=False,
                      action="store_true", help="dataset property")
    prop.add_argument("--multiclass", dest="multiclass", default=False,
                      action="store_true", help="dataset property")
    prop.add_argument("--binary", dest="binary", default=False,
                      action="store_true", help="dataset property")

    args, unknown = parser.parse_known_args()

    props = {"sparse": args.sparse,
             "multilabel": args.multilabel,
             "multiclass": args.multiclass}
    est_dict, preproc_dict = get_dict(task_type=args.task_type, **props)

    est_table = build_table(est_dict)
    preproc_table = build_table(preproc_dict)

    est_table = table_str % (caption_str % (args.task_type, str(props)), est_table)
    preproc_table = table_str % (caption_str % ("preprocessor", str(props)), preproc_table)

    tex_doc = template_string % "\n".join([est_table, preproc_table])
    if args.save is None:
        print tex_doc
    else:
        fh = open(args.save, "w")
        fh.write(tex_doc)
        fh.close()
        proc = subprocess.Popen(shlex.split('pdflatex %s' % args.save))
        proc.communicate()
        try:
            os.remove(args.save.replace(".tex", ".aux"))
            os.remove(args.save.replace(".tex", ".log"))
        except OSError:
            # This is fine
            pass


if __name__ == "__main__":
    main()