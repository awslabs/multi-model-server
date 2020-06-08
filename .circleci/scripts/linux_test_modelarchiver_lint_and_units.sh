#!/bin/bash

cd model-archiver/

# Lint test
pylint -rn --rcfile=./model_archiver/tests/pylintrc model_archiver/.

# Execute python unit tests
python -m pytest model_archiver/tests/unit_tests