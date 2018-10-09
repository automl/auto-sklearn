set -e

run_tests() {
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR

    cwd=`pwd`
    examples_dir=$cwd/examples
    test_dir=$cwd/test/

    cd $TEST_DIR
    if [[ "$COVERAGE" == "true" ]]; then
        nosetests --no-path-adjustment -sv --with-coverage --cover-package=$MODULE $test_dir
    elif [[ "$EXAMPLES" == "true" ]]; then
        for example in `find $examples_dir -name '*.py'`
        do
            python $example
        done
    else
        nosetests --no-path-adjustment -sv $test_dir
    fi
}

if [[ "$RUN_FLAKE8" ]]; then
    source ci_scripts/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi


