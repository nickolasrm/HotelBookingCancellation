install:
	pip install -r src/requirements.txt

install-test:
	make install
	pip install -r src/requirements-test.txt

install-dev:
	make install-test
	pip install -r src/requirements-dev.txt
	git init
	pre-commit install

lint:
	pre-commit

lint-all:
	pre-commit run --all-files

test:
	pytest
