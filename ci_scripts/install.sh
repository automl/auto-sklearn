set -e

# Install general requirements the way setup.py suggests
pip install pytest==4.6.* pep8 codecov pytest-cov flake8 flaky mypy flake8-import-order matplotlib

# Install the packages in the correct order specified by the requirements.txt file
cat requirements.txt | xargs -n 1 -L 1 pip install

# Debug output to know all exact package versions!
    pip freeze

if [[ "$TEST_DIST" == "true" ]]; then
    pip install twine
    python setup.py sdist
    # Find file which was modified last as done in https://stackoverflow.com/a/4561987
    dist=`find dist -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
    echo "Installing $dist"
    pip install "$dist"
    twine_output=`twine check "$dist"`
    if [[ "$twine_output" != "Checking $dist: PASSED" ]]; then
        echo $twine_output
        exit 1
    else
        echo "Check with Twine: OK: $twine_output"
    fi
else
    python setup.py check -m -s
    python setup.py install
fi

# Install openml dependency for metadata generation unittest
pip install openml
mkdir ~/.openml
echo "apikey = 610344db6388d9ba34f6db45a3cf71de" > ~/.openml/config
