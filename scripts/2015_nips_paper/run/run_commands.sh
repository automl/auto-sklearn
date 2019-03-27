#!/usr/bin/env bash

# Run all commands in commands.txt. Modify this to run on a cluster.
cat "../commands.txt" | while read line; do eval "$line"; done