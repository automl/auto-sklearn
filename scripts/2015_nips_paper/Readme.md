## Reproduce results of Efficient and Robust Automated Machine Learning (Feurer et al.)
This folder contains all necessary scripts in order to reproduce the results shown in
Figure 3 of Efficient and Robust Automated Machine Learning (Feurer et al.). The scripts
can be modified to include different datasets, change the runtime, etc. The scripts only
only handles classification tasks, and balanced accuracy is used as the score metric.

### 1. Creating commands.txt
To run the experiment, first create commands.txt by running:
```bash
cd setup
bash create_commands.sh
```
The script can be modified to run experiments with different settings, i.e. 
different runtime and/or different tasks.

### 2. Executing commands.txt
Run each commands in commands.txt:
```bash
cd run
bash run_commands.sh
```
Each command line in commands.txt first executes model fitting, and then creating the
single best and ensemble trajectories. Therefore, the commands can be run in parallel
on a cluster by modifying run_commands.sh.

### 3. Plotting the results
To plot the results, run:
```bash
cd plot
bash plot_ranks.py
```



