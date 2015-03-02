import setuptools
from distutils.extension import Extension

#for basename in ["chelper_functions", "cdata_manager"]:
#    os.system("cython ./AutoML2015/data/%s.pyx"%basename)
#    os.system('gcc -w -shared -fPIC -fwrapv -O2 -fno-strict-aliasing -I/usr/include/python2.7 -o AutoML2015/data/%s.so AutoML2015/data/%s.c'%(basename, basename))

setuptools.setup(name="AutoSklearn",
                 description="Code to participate in the AutoML 2015 challenge.",
                 version="0.1dev",
                 # When we have a package, uncomment this
                 ext_modules=[Extension("autosklearn.data.cdata_manager", ["autosklearn/data/cdata_manager.c"]),
                              Extension("autosklearn.data.chelper_functions", ["autosklearn/data/chelper_functions.c"])],
                 packages=setuptools.find_packages(),
                 install_requires=["numpy",
                                   "pyyaml",
                                   "scipy",
                                   "scikit-learn==0.15.2",
                                   "nose",
                                   "lockfile",
                                   "HPOlibConfigSpace",
                                   "ParamSklearn",
                                   "pymetalearn",
                                   "cma"],
                 test_suite="nose.collector",
                 package_data={'': ['*.txt', '*.md', 'metadata']},
                 author="Matthias Feurer",
                 author_email="feurerm@informatik.uni-freiburg.de",
                 license="BSD",
                 platforms=['Linux'],
                 classifiers=[],
                 url='www.automl.org')




