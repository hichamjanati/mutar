#! /usr/bin/env python
import os

from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension(
        "mutar.solver_mtw_cd",
        ['mutar/solver_mtw_cd.pyx'],
    ),
]


# get __version__ from _version.py
ver_file = os.path.join('mutar', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'mutar'
DESCRIPTION = 'Multi-Task Regression in Python'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'H. Janati'
MAINTAINER_EMAIL = 'hicham.janati@inria.fr'
URL = 'https://github.com/hichamjanati/mutar'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/hichamjanati/mutar'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scikit-learn', 'numba>=0.40.1']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      include_dirs=[numpy.get_include()],
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      ext_modules=cythonize(extensions))
