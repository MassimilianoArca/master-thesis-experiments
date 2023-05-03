ifdef OS
   export PYTHON_COMMAND=python
else
   export PYTHON_COMMAND=python3.8
endif


version:
	poetry run python -c "from ml3_repo_manager.version import make_version; make_version()"

setup:
	$(PYTHON_COMMAND) -m pip install poetry
	poetry env use $(PYTHON_COMMAND)
	poetry run pip install --upgrade pip
	make add-private-pypi
	cp .hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
	cp .hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
	cp .hooks/post-merge .git/hooks/post-merge && chmod +x .git/hooks/post-merge

install:
	poetry lock
	poetry install

isort:
	poetry run isort --skip-glob=.tox .

format:
	poetry run black master_thesis_experiments

lint:
	make isort
	make format
	poetry run pylint  --extension-pkg-whitelist='pydantic' master_thesis_experiments
	poetry run flake8 master_thesis_experiments
	poetry run mypy --ignore-missing-imports --install-types --non-interactive --package master_thesis_experiments

test:
	poetry run pytest --verbose --color=yes --cov=master_thesis_experiments

env:
	poetry shell

requirements:
	poetry export --without-hashes --with-credentials -f requirements.txt

minimum-requirements:
	poetry export --without-hashes --with-credentials -f requirements.txt | grep -e ml3-repo-manager -e pyyaml -e -- > requirements.txt

publish:
	poetry build
	poetry publish -r private-pypi
	rm -rf dist/

class-diagram:
	poetry run pyreverse -o svg --colorized --module-names y --all-ancestors master_thesis_experiments