# Update version in pyproject.toml

# Run it once to link pre-commit hooks
link_git_hooks:
	ln -s ../../pre-commit.bash .git/hooks/pre-commit
	chmod ug+x .git/hooks/pre-commit

# Git uses hooks to run check_code before each commit
check_code:
	Black abtoolkit
	pylint abtoolkit
	python -m unittest -v

build_package:
	pip install --upgrade build
	python3 -m build

load_package_to_testpypi:
	pip install --upgrade twine
	python3 -m twine upload --repository testpypi dist/*

load_package_to_pypi:
	pip install --upgrade twine
	python3 -m twine upload dist/*
