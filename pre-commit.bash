#!/usr/bin/env bash

echo "Running pre-commit hook"
source venv/bin/activate
make check_code

# $? stores exit value of the last command
if [ $? -ne 0 ];
then
    echo "Tests must pass before commit!"
    exit 1
else
    echo "All pre-commit checks passed!"
fi