#!/bin/bash

flake8 --max-line-length=100 --show-source \
    autosklearn \
    test \
    examples \
    || exit 1

echo -e "No problem detected by flake8\n"
