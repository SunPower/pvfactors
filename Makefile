build-docs:
	python setup.py build_sphinx

build-package:
	python setup.py bdist_wheel sdist
