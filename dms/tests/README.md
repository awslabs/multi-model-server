# Testing DMS

## Pre-requisites

You will need some additional Python modules to run the unit tests and integration tests.

```bash
sudo pip install mock pytest
```

You will also need the source for the project, so clone the project first.

```bash
git clone https://github.com/awslabs/deep-model-server.git
cd deep-model-server
```

## Unit Tests

You can run the unit tests with the following:

```bash
python -m pytest dms/tests/unit_tests/
```
## CI Tests

You can run the integration tests with the following:
 
```bash
python -m pytest dms/test/integration_tests/
```