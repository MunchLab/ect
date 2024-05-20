.PHONY: tests

help:
	@Echo       clean: runs autopep to improve formatting of code
	@Echo		reqs: replace requirements.txt file for use by collaborators to create virtual environments and for use in documentation
	@Echo       tests: runs unit tests
	@Echo       docs: clears files from docs folder, rebuilds docs from source folder
	@Echo       release: runs build to create tar and wheel for distribution
	@Echo       all: runs clean build and docs folders, creates new html folders in build, moves relevant files to docs, and runs unittests and autopep.

reqs:
	@pip freeze > requirements.txt

clean:
	# Running autopep8
	@autopep8 -r --in-place ect/

tests:
	# Running unittests
	@python -m unittest

release:
	python setup.py sdist bdist_wheel

html:
	# Running sphinx-build to build html files in build folder.
	rm -r docs
	mkdir docs
	sphinx-build -M html doc_source docs
	rsync -a docs/html/ docs/
	rm -r docs/html
	
all:
	# Running autopep8
	@make clean
	
	# Generate documentation 
	@make html
	
	# Running unittests
	@make tests