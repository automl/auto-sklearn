from typing import Literal

PYNISHER_CONTEXT = Literal["spawn", "fork", "forkserver"]
valid_pynisher_contexts = ["spawn", "fork", "forkserver"]

BUDGET_TYPE = Literal["subsample", "iterations", "mixed"]
valid_budget_types = ["subsample", "iterations", "mixed"]

RESAMPLING_STRATEGY = Literal[
    "holdout",
    "holdout-iterative-fit",
    "cv",
    "partial-cv",
    "partial-cv-iterative-fit",
    "cv-iterative-fit",
    "test",
]
valid_resampling_strategies = [
    "holdout",
    "holdout-iterative-fit",
    "cv",
    "partial-cv",
    "partial-cv-iterative-fit",
    "cv-iterative-fit",
    "test",
]
