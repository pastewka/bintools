import os
import glob
import sys
import re

from numpy.distutils.core import setup, Extension

setup(name='matscipy',
      description='Binarization tools',
      maintainer='Lars Pastewka',
      maintainer_email='lars.pastewka@imtek.uni-freiburg.de',
      license='LGPLv2.1+',
      package_dir={'bintools': 'bintools'},
      packages=['bintools'],
      ext_modules=[
        Extension(
            '_bintools',
            ['c/bintoolsmodule.c'],
            )
        ],
      )
