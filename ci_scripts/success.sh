set -e

if [[ "$COVERAGE" == "true" ]]; then

    python setup.py install
    cp $TEST_DIR/.coverage $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR
    codecov
    coverage report

fi