#!/usr/bin/env bash
pytest -n 3 --durations=20 --timeout=300 --dist load --timeout-method=thread --fulltrace -v $1
