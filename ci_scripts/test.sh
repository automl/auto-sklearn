set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cwd=`pwd`
test_dir=$cwd/test/

cd $TEST_DIR

if [[ "$COVERAGE" == "true" ]]; then
    nosetests --no-path-adjustment -sv --with-coverage --cover-package=$MODULE $test_dir
else
    nosetests --no-path-adjustment -sv $test_dir
fi
