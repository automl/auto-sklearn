#!/usr/bin/env bash

cat "../commands.txt" | while read line; do eval "$line"; done