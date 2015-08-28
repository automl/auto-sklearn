#!/bin/bash

source ~/Develop/venv/system/bin/activate


cd ../au
yapf -ri ./
isort -rc --atomic -e -w 79 -af -m2 -cs -tc -s init.py ./
pyformat -r -a -i -j 4 --exclude init.py ./


deactivate
