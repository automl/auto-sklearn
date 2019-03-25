pip --version
# Install general requirements the way setup.py suggests
pip install nose pep8 codecov coverage flake8

# Install the packages in the correct order specified by the requirements.txt file
cat requirements.txt | xargs -n 1 -L 1 pip install

# Debug output to know all exact package versions!
    pip freeze

if [[ "$TEST_DIST" == "true" ]]; then
    python setup.py sdist
    # Find file which was modified last as done in https://stackoverflow.com/a/4561987
    dist=`find dist -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
    echo "Installing $dist"
    pip install "$dist"
else
    python setup.py check -m -s
    python setup.py install
fi

# Install openml dependency for metadata generation unittest
pip install openml
mkdir ~/.openml
echo "apikey = 610344db6388d9ba34f6db45a3cf71de" > ~/.openml/config
