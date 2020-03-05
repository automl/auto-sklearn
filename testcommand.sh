#!/usr/bin/env bash
pytest -n 4 --durations=20 --timeout=600 --timeout-method=thread -v $1
