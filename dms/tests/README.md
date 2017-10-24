# Testing DMS

## Unit Tests

You will need some additional Python modules to run the unit tests.

```bash
sudo pip install mock pytest
```

You will also need the source for the project, so clone the project first.

```bash
git clone --recursive https://github.com/deep-learning-tools/deep-model-server.git
cd deep-model-server
```

Then you can run the unit tests with the following:

```bash
python -m pytest mms/tests/unit_tests/
```
## CI Tests

TODO: add instructions when assets (cat.jpg) are available.
