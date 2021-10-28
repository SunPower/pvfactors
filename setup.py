""" Install pvfactors """

from setuptools import setup
import versioneer

DESCRIPTION = (
    '2D View Factor Model to calculate the irradiance incident on ' +
    'various surfaces of PV arrays')
with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = list(f)

DISTNAME = 'pvfactors'
AUTHOR = 'SunPower'
MAINTAINER_EMAIL = 'marc.abouanoma@sunpowercorp.com'
URL = 'https://github.com/SunPower/pvfactors'
PACKAGES = ['pvfactors', 'pvfactors.geometry', 'pvfactors.irradiance',
            'pvfactors.viewfactors']
LICENSE = 'BSD 3-Clause'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

TESTS_REQUIRES = ['pytest>=3.2.1', 'pytest-mock>=1.10.0', 'mock']

setup(name=DISTNAME,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      packages=PACKAGES,
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES,
      extras_require={
          'testing': TESTS_REQUIRES,
          'docs': ['Sphinx~=4.0', 'sphinx_rtd_theme', 'nbsphinx',
                   'sphinxcontrib-napoleon', 'sphinxcontrib_github_alt',
                   'ipykernel']
      },
      license=LICENSE
      )
