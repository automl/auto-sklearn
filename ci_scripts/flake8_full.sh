#!/bin/bash

echo "Running flake8 on whole sections of the package"
flake8 --max-line-length=100 --show-source autosklearn/util