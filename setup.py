import setuptools

setuptools.setup(name="ParamSklearn",
                 description="Scikit-Learn wrapper for automatic "
                             "hyperparameter configuration.",
                 version="0.1dev",
                 packages=setuptools.find_packages(),
                 install_requires=["numpy==1.9.0",
                                   "scipy==0.14.0",
                                   "scikit-learn==0.15.2",
                                   "nose",
                                   "HPOlibConfigSpace",
                                   "GPy==0.6.0"],
                 test_requires=["mock"],
                 test_suite="nose.collector",
                 package_data={'': ['*.txt', '*.md']},
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 license="BSD",
                 platforms=['Linux'],
                 classifiers=[],
                 url="github.com/mfeurer/paramsklearn")
