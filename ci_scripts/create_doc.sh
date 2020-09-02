# This script is mostly adopted from https://github.com/openml/openml-python/blob/develop/ci_scripts/create_doc.sh

set -euo pipefail

# Check if DOCPUSH is set
if ! [[ -z ${DOCPUSH+x} ]]; then

    if [[ "$DOCPUSH" == "true" ]]; then

        # install documentation building dependencies
        pip install --upgrade matplotlib seaborn setuptools pytest mypy coverage sphinx pillow sphinx-gallery sphinx_bootstrap_theme cython numpydoc nbformat nbconvert mock

        # $1 is the branch name
        # $2 is the global variable where we set the script status

        # delete any previous documentation folder
        if [ -d doc/$1 ]; then
            rm -rf doc/$1
        fi

        # create the documentation
        cd doc && make html 2>&1

        if ! { [ $1 = "master" ] || [ $1 = "development" ]; }; then
            { echo "Not one of the allowed branches"; exit 0; }
        fi

        # create directory with branch name
        # the documentation for dev/stable from git will be stored here
        mkdir $1

        # get previous documentation from github
        git clone https://github.com/automl/auto-sklearn.git --branch gh-pages --single-branch

        # copy previous documentation
        cp -r auto-sklearn/. $1
        rm -rf auto-sklearn

        # if the documentation for the branch exists, remove it
        if [ -d $1/$1 ]; then
            rm -rf $1/$1
        fi

        # copy the updated documentation for this branch
        mkdir $1/$1
        cp -r build/html/. $1/$1

        # takes a variable name as an argument and assigns the script outcome to a
        # variable with the given name. If it got this far, the script was successful
        function set_return() {
            # $1 is the variable where we save the script outcome
            local __result=$1
            local  status='success'
            eval $__result="'$status'"
        }

        set_return "$2"
    fi
fi
# Workaround for travis failure
set +u
