# First, create a temporary directory for zipping files
rm .zip -rf
rm build -rf
rm submission.zip
mkdir .zip
mkdir build

# Build an installable for the submission project
python setup.py sdist
pip install dist/AutoML2015-0.1dev.tar.gz -t .zip -t .zip -b build --no-deps

# Add dependencies
pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev -t .zip/AutoML2015 -b build --no-deps
pip install git+https://bitbucket.org/mfeurer/autosklearn#egg=AutoSklearn -t .zip/AutoML2015 -b build --no-deps

# Clean up the submission directory
find -name "*.egg-info" -exec rm -rf {} \;
find .zip -name "*.pyc" -exec rm -rf {} \;
find .zip -name "*~" -exec rm -rf {} \;

# Zip it
cd .zip/AutoML2015
zip -r ../../submission.zip *
cd ../..

