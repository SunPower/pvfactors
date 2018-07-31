#!/usr/bin/env python

from setuptools import setup
import versioneer

DESCRIPTION = ('2D View Factor Model to calculate the irradiance incident on '
               + 'PV arrays')
DISTNAME = 'pvfactors'
AUTHOR = 'anomam'
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

INSTALL_REQUIRES = ['numpy>=1.11.1',
                    'scipy>=0.17.1',
                    'pandas>=0.18.1',
                    'shapely>=1.5.16',
                    'pvlib>=0.5.0',
                    'matplotlib>=2.1.0',
                    'future>=0.16.0',
                    'six>=1.11.0']

TESTS_REQUIRE = ['pytest>=3.2.1']

setup(name=DISTNAME,
      description=DESCRIPTION,
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
