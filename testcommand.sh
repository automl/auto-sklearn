#!/usr/bin/env bash
pytest -n 8 --durations=300 --timeout=300 --dist load --timeout-method=thread -v $1
