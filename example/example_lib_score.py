# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
from sys import stderr

import numpy as np

from autosklearn.metrics.libscores import show_all_scores

swrite = stderr.write

if (os.name == 'nt'):
    filesep = '\\'
else:
    filesep = '/'


def main():
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

    sol0 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

    comment = ['PERFECT']
    Pred = [sol0]
    Sol = [sol0]

    comment.append('ANTI-PERFECT, very bad for r2_score')
    Pred.append(1 - sol0)
    Sol.append(sol0)

    comment.append(
        'UNEVEN PROBA, BUT BINARIZED VERSION BALANCED (bac and auc=0.5)')
    Pred.append(np.array([[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])
                )  # here is we have only 2, pac not 0 in uni-col
    Sol.append(sol0)

    comment.append(
        'PROBA=0.5, TIES BROKEN WITH SMALL VALUE TO EVEN THE BINARIZED VERSION')
    Pred.append(np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                          [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]]))
    Sol.append(sol0)

    comment.append('PROBA=0.5, TIES NOT BROKEN (bad for f1 score)')
    Pred.append(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
    Sol.append(sol0)

    sol1 = np.array([[1, 0], [0, 1], [0, 1]])

    comment.append(
        'EVEN PROBA, but wrong PAC prior because uneven number of samples')
    Pred.append(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
    Sol.append(sol1)

    comment.append(
        'Correct PAC prior; score generally 0. But 100% error on positive class because of binarization so f1 (1 col) is at its worst.')
    p = len(sol1)
    Pred.append(np.array([sum(sol1) * 1. / p] * p))
    Sol.append(sol1)

    comment.append('All positive')
    Pred.append(np.array([[1, 1], [1, 1], [1, 1]]))
    Sol.append(sol1)

    comment.append('All negative')
    Pred.append(np.array([[0, 0], [0, 0], [0, 0]]))
    Sol.append(sol1)

    for k in range(len(Sol)):
        sol = Sol[k]
        pred = Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        print('------ 2 columns ------')
        show_all_scores(sol, pred)
        print('------ 1 column  ------')
        sol = np.array([sol[:, 0]]).transpose()
        pred = np.array([pred[:, 0]]).transpose()
        show_all_scores(sol, pred)

    print('\n\nMulticlass score verification:')
    print('\n\n==========================')
    sol2 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

    comment = ['Three classes perfect']
    Pred = [sol2]
    Sol = [sol2]

    comment.append('Three classes all wrong')
    Pred.append(np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]))
    Sol.append(sol2)

    comment.append('Three classes equi proba')
    Pred.append(np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3],
                          [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]))
    Sol.append(sol2)

    comment.append('Three classes some proba that do not add up')
    Pred.append(np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                          [0.7, 0.3, 0.3]]))
    Sol.append(sol2)

    comment.append('Three classes predict prior')
    Pred.append(np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                          [0.75, 0.25, 0.]]))
    Sol.append(sol2)

    for k in range(len(Sol)):
        sol = Sol[k]
        pred = Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)

    print('\n\nMulti-label score verification: 1) all identical labels')
    print('\n\n=======================================================')
    print(
        '\nIt is normal that for more then 2 labels the results are different for the multiclass scores.')
    print('\nBut they should be indetical for the multilabel scores.')
    num = 2

    sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    sol3 = sol[:, 0:num]
    if num == 1:
        sol3 = np.array([sol3[:, 0]]).transpose()

    comment = ['{} labels perfect'.format(num)]
    Pred = [sol3]
    Sol = [sol3]

    comment.append('All wrong, in the multi-label sense')
    Pred.append(1 - sol3)
    Sol.append(sol3)

    comment.append('All equi proba: 0.5')
    sol = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]])
    if num == 1:
        Pred.append(np.array([sol[:, 0]]).transpose())
    else:
        Pred.append(sol[:, 0:num])
    Sol.append(sol3)

    comment.append('All equi proba, prior: 0.25')
    sol = np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25]])
    if num == 1:
        Pred.append(np.array([sol[:, 0]]).transpose())
    else:
        Pred.append(sol[:, 0:num])
    Sol.append(sol3)

    comment.append('Some proba')
    sol = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                    [0.7, 0.7, 0.7]])
    if num == 1:
        Pred.append(np.array([sol[:, 0]]).transpose())
    else:
        Pred.append(sol[:, 0:num])
    Sol.append(sol3)

    comment.append('Invert both solution and prediction')
    if num == 1:
        Pred.append(np.array([sol[:, 0]]).transpose())
    else:
        Pred.append(sol[:, 0:num])
    Sol.append(1 - sol3)

    for k in range(len(Sol)):
        sol = Sol[k]
        pred = Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)

    print('\n\nMulti-label score verification:')
    print('\n\n==========================')

    sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

    comment = ['Three labels perfect']
    Pred = [sol4]
    Sol = [sol4]

    comment.append('Three classes all wrong, in the multi-label sense')
    Pred.append(1 - sol4)
    Sol.append(sol4)

    comment.append('Three classes equi proba')
    Pred.append(np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3],
                          [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]))
    Sol.append(sol4)

    comment.append('Three classes some proba that do not add up')
    Pred.append(np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                          [0.7, 0.3, 0.3]]))
    Sol.append(sol4)

    comment.append('Three classes predict prior')
    Pred.append(np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                          [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]]))
    Sol.append(sol4)

    for k in range(len(Sol)):
        sol = Sol[k]
        pred = Pred[k]
        print('****** ({}) {} ******'.format(k, comment[k]))
        show_all_scores(sol, pred)

if __name__ == '__main__':
    main()