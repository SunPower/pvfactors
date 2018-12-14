#!/usr/bin/env python

from setuptools import setup
import versioneer

DESCRIPTION = ('2D View Factor Model to calculate the irradiance incident on '
               + 'various surfaces of PV arrays')
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()
DISTNAME = 'pvfactors'
AUTHOR = 'SunPower'
MAINTAINER_EMAIL = 'marc.abouanoma@sunpowercorp.com'
URL = 'https://github.com/SunPower/pvfactors'
PACKAGES = ['pvfactors']
LICENSE = 'BSD 3-Clause'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['numpy>=1.13.1',
                    'scipy>=0.19.1',
                    'pandas>=0.23.3',
                    'shapely>=1.6.1',
                    'pvlib>=0.6.1b0',
                    'matplotlib>=2.1.0',
                    'future>=0.16.0',
                    'six>=1.11.0']

TESTS_REQUIRE = ['pytest>=3.2.1', 'pytest-mock>=1.10.0']

setup(name=DISTNAME,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      packages=PACKAGES,
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      license=LICENSE
      )
