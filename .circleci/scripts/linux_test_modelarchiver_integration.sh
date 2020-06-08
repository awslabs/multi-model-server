#!/bin/bash

cd model-archiver/

# Install model archiver module
pip install .

# Execute integration tests
pytest model_archiver/tests/integ_tests