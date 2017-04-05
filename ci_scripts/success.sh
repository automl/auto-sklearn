set -e

if [[ "$COVERAGE" == "true" ]]; then
    cp $TEST_DIR/.coverage $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR
    codecov
fi