set -e

run_tests() {
    # Get into a temp directory to run test from the installed scikit learn and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR

    cwd=`pwd`
    examples_dir=$cwd/examples/
    package_dir=$cwd/autosklearn/
    test_dir=$cwd/test/

    cd $TEST_DIR
    nose_params=""
    if [[ "$COVERAGE" == "true" ]]; then
        nose_params="--with-coverage --cover-package=$MODULE"
    fi

    nosetests $package_dir $test_dir $examples_dir --no-path-adjustment -sv --exe --with-doctest $nose_params

    if [[ "$EXAMPLES" == "true" ]]; then
        for example in `find $examples_dir -name '*.py'`
        do
            python $example
        done
    fi
}

if [[ "$RUN_FLAKE8" ]]; then
    source ci_scripts/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi


