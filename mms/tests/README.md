# Testing MMS

## Pre-requisites

You will need some additional Python modules to run the unit tests and integration tests.

```bash
sudo pip install mock pytest
```

You will also need the source for the project, so clone the project first.

```bash
git clone https://github.com/awslabs/mxnet-model-server.git
cd mxnet-model-server
```

## Unit Tests

You can run the unit tests with the following:

```bash
python -m pytest mms/tests/unit_tests/
```
## CI Tests

You can run the integration tests with the following:
 
```bash
python -m pytest mms/test/integration_tests/
```