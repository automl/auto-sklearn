#!/bin/bash

# Default ignore is 'W504', 'E24', 'E126', 'W503', 'E123',
# 'E704', 'E121', 'E226'
# The I* are flake order
# Add flake 8 order
flake8 --application-import-names=autosklearn --ignore=I100,I101,I201,I202,W504,E24,E126,W503,E123,E704,E121,E226 --max-line-length=100 --show-source \
    autosklearn \
    test \
    examples \
    || exit 1

# Support for incremental flake-8 order
flake8 --application-import-names=autosklearn --max-line-length=100 --show-source \
    autosklearn/data \
    autosklearn/ensembles \
    autosklearn/metrics \
    autosklearn/util \
    || exit 1
echo -e "No problem detected by flake8\n"
