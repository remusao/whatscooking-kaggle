
envtmpdir=doc/build

.PHONY: tests, doc

check:
	python setup.py test

doc:
	sphinx-build -b html -d ${envtmpdir}/doctrees doc/source  ${envtmpdir}/html

gendoc:
	sphinx-apidoc -o doc/source/ whatscooking

install:
	pip install . --upgrade

clean:
	python setup.py clean
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -frv *.egg
	rm -frv whatscooking.egg-info
	rm -frv dist
	rm -frv .tox
	rm -frv build
	rm -frv htmlcov
