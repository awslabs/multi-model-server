#!/bin/bash

python -m pytest --cov-report html:htmlcov --cov=mms/ mms/tests/unit_tests/