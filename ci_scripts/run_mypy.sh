#MYPYPATH=smac
MYPYOPTS=""

MYPYOPS="$MYPYOPS --ignore-missing-imports --follow-imports skip"
# We would like to have the following options set, but for now we have to use the ones above to get started
#MYPYOPTS="--ignore-missing-imports --strict"
#MYPYOPTS="$MYPYOPS --disallow-any-unimported"
#MYPYOPTS="$MYPYOPS --disallow-any-expr"
#MYPYOPTS="$MYPYOPS --disallow-any-decorated"
#MYPYOPTS="$MYPYOPS --disallow-any-explicit"
#MYPYOPTS="$MYPYOPS --disallow-any-generics"
MYPYOPTS="$MYPYOPS --disallow-untyped-decorators"
MYPYOPTS="$MYPYOPS --disallow-incomplete-defs"
MYPYOPTS="$MYPYOPS --disallow-untyped-defs"
# Add the following once the scenario is removed from teh main code or typed
# https://mypy.readthedocs.io/en/stable/command_line.html#configuring-warnings
# MYPYOPTS="$MYPYOPS --warn-unused-ignores"

mypy $MYPYOPTS --show-error-codes \
    autosklearn/data/ \
    autosklearn/ensembles \
	autosklearn/metrics \
	autosklearn/util
