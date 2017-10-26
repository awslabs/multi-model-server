# Testing DMS

## Unit Tests

You will need some additional Python modules to run the unit tests.

```bash
sudo pip install mock pytest
```

You will also need the source for the project, so clone the project first.

```bash
git clone https://github.com/awslabs/deep-model-server.git
cd deep-model-server
```

Then you can run the unit tests with the following:

```bash
python -m pytest mms/tests/unit_tests/
```
## CI Tests

TODO: add comments about where they're hosted, etc.
