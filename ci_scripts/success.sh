set -e

if [[ "$COVERAGE" == "true" ]]; then
    codecov
fi