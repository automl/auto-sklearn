## Reproduce results of Efficient and Robust Automated Machine Learning (Feurer et al.)

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
To run the commands in parallel or in a cluster,
modify run_commands.sh.

### 3. Plotting the results
To plot the results, run:
```bash
cd plot
bash plot_ranks.py
```


