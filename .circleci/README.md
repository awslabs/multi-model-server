# Multi Model Server CircleCI build
Model Server uses CircleCI builds. This folder contains the config and scripts that are needed for CircleCI.

## config.yml
config.yml contains MMS build logic which will be used by CircleCI.

## Docker images
MMS uses customized docker image for its CircleCI build. To make sure MMS is compatible with
 both Python2 and Python3, we use two build projects. We published two code build docker
 images on docker hub:
* --------/mms-build:python2.7
* --------/mms-build:python3.6

Following files in the _images_ folder are used to create the docker images
* Dockerfile.python2.7 - Dockerfile for --------/mms-build:python2.7
* Dockerfile.python3.6 - Dockerfile for --------/mms-build:python3.6

## CircleCI cli local
To make it easy for developers to debug build issues locally, MMS supports CircleCI cli for running a job in a container on your machine.

#### Dependencies
1. CircleCI cli from - https://circleci.com/docs/2.0/local-cli/#installation
2. PyYAML (pip install PyYaml)
3. docker

#### Command
Developers can use the following command to build MMS locally:
**_./run_circleci_tests.py <workflow_name> <job_name>_**
```bash
$ cd multi-model-server
$ ./run_circleci_tests.py build_and_test python_tests
```

> To avoid Pull Request build failure on github, developer should always make sure local build can pass.
