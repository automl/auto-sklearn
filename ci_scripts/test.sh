set -e

run_tests() {
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR

    cwd=`pwd`
    test_dir=$cwd/test/

    cd $TEST_DIR

    python -c 'import autosklearn; print("Auto-sklearn imported from: %s" % autosklearn.__file__)'

    test_params=""
    if [[ "$COVERAGE" == "true" ]]; then
        test_params="--cov=$MODULE"
    fi

    python -m pytest $test_dir -v $test_params

    cd $cwd
}

run_examples() {
    cwd=`pwd`
    examples_dir=$cwd/examples/

    # Get into a temp directory to run test from the installed scikit learn and
    # check if we do not leave artifacts
    mkdir -p $EXAMP_DIR
    cd $EXAMP_DIR

    python -c 'import autosklearn; print("Auto-sklearn imported from: %s" % autosklearn.__file__)'
    for example in `find $examples_dir -name '*.py'`
    do
        echo '***********************************************************'
        echo "Running example $example"
        python $example
    done

    cd $cwd
}

if [[ "$RUN_FLAKE8" ]]; then
    echo '***********************************************************'
    echo '***********************************************************'
    echo 'Running flake8'
    echo '***********************************************************'
    source ci_scripts/run_flake8.sh
fi

if [[ "$RUN_MYPY" ]]; then
    echo '***********************************************************'
    echo '***********************************************************'
    echo 'Running Mypy'
    echo '***********************************************************'
    source ci_scripts/run_mypy.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    echo '***********************************************************'
    echo '***********************************************************'
    echo 'Running unittests'
    echo '***********************************************************'
    run_tests
fi

if [[ "$EXAMPLES" ]]; then
    echo '***********************************************************'
    echo '***********************************************************'
    echo 'Running examples'
    echo '***********************************************************'
    run_examples
fi


