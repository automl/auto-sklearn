## Reproduce results of Efficient and Robust Automated Machine Learning (Feurer et al.)

### 1- Get performance of the best single model over time
An example of run on [`kr-vs-kp` dataset](https://www.openml.org/t/3):

```bash
python score_auto_sklearn.py --working-directory log_output --time-limit 100 \
       --per-run-time-limit 30 \
       --task-id 3 -s 1 --nb_conf_metalearning 0
```
Scores are stored at {working-directory}/{seed}/{task_id}/score_single_best.csv.


### 2- Get overall performance of Auto-Sklearn (with ensemble)
```bash
python score_ensemble.py --input-directory log_output/1/3/ -s 1 --output-file scores_ens.txt
```
