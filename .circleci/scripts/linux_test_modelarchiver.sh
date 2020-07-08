#!/bin/bash

cd model-archiver/

# Lint test
pylint -rn --rcfile=./model_archiver/tests/pylintrc model_archiver/.

# Execute python unit tests
python -m pytest --cov-report html:results_units --cov=./ model_archiver/tests/unit_tests


# Install model archiver module
pip install .

# Execute integration tests
python -m pytest model_archiver/tests/integ_tests
# ToDo - Report for Integration tests ?