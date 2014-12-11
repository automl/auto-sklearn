import setuptools

setuptools.setup(name="AutoSklearn",
                 description="Scikit-Learn wrapper for automatic "
                             "hyperparameter configuration.",
                 version="0.1dev",
                 packages=setuptools.find_packages(),
                 install_requires=["numpy",
                                   "scipy",
                                   "scikit-learn==0.15.2",
                                   "nose",
                                   "HPOlibConfigSpace"],
                 test_suite="nose.collector",
                 package_data={'': ['*.txt', '*.md']},
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 license="BSD",
                 platforms=['Linux'],
                 classifiers=[],
                 url="github.com/mfeurer/autosklearn")
