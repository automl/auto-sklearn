import setuptools

setuptools.setup(name="ParamSklearn",
                 description="Scikit-Learn wrapper for automatic "
                             "hyperparameter configuration.",
                 version="0.16.1.0",
                 packages=setuptools.find_packages(),
                 install_requires=["numpy>=1.6.0",
                                   "scipy>=0.14.0",
                                   "scikit-learn==0.16.1",
                                   "nose",
                                   "HPOlibConfigSpace"],
                 test_requires=["mock"],
                 test_suite="nose.collector",
                 package_data={'': ['*.txt', '*.md']},
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 license="BSD",
                 platforms=['Linux'],
                 classifiers=[],
                 url="github.com/mfeurer/paramsklearn")
