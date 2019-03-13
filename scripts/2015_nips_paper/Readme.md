## Reproduce results of Efficient and Robust Automated Machine Learning (Feurer et al.)

### 1. Creating commands.txt
To run the experiment, first create commands.txt by running:
```bash
bash create_commands.sh
```
The script can be modified to run experiments with different settings, i.e. 
different runtime and/or different tasks.

### 2. Executing commands.txt
Run each commands in commands.txt. Here, we use job.sh to run experiments in 
the meta-slurm cluster. To run experiments in other environments, one can simply
modify or write a new script to execute each line of commands.txt.
```bash
bash job.sh
```

### 3. Plotting the results
To plot the results, execute:
```bash
bash plot_experiments.sh
```


