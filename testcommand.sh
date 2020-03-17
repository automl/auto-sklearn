#!/usr/bin/env bash
pytest -n 8 --durations=300 --timeout=1800 --timeout-method=thread -v $1
