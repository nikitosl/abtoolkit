# Update version in pyproject.toml
ls:
	ls -la

build_package:
	pip install --upgrade build
	python3 -m build

load_package_to_testpypi:
	pip install --upgrade twine
	python3 -m twine upload --repository testpypi dist/*

load_package_to_pypi:
	pip install --upgrade twine
	python3 -m twine upload dist/*