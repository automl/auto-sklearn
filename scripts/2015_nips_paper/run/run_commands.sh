#!/usr/bin/env bash

# Run all commands in commands.txt. Each command line executes first the model fitting,
# and then creates the trajectory of a single model and the ensemble. Therefore, each
# line can be executed separately and in parallel, for example, on a cluster environment.
cat "../commands.txt" | while read line; do eval "$line"; done