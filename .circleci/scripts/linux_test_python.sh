#!/bin/bash

# Lint Test
pylint -rn --rcfile=./mms/tests/pylintrc mms/.

# Execute python tests
python -m pytest --cov-report html:htmlcov --cov=mms/ mms/tests/unit_tests/