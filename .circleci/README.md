# Multi Model Server CircleCI build
Model Server uses CircleCI for builds. This folder contains the config and scripts that are needed for CircleCI.

## config.yml
_config.yml_ contains MMS build logic which will be used by CircleCI.

## Workflows and Jobs
Currently, following _workflows_ are available -
1. **build_and_test_python27**
2. **build_and_test_python36**

Following _jobs_ are executed under each workflow
1. **build** - Builds _frontend/model-server.jar_ and executes tests from gradle
2. **modelarchiver** - Builds and tests modelarchiver module
3. **python_tests** - Executes pytests from _mms/tests/unit_tests/_
4. **benchmark** - Executes latency benchmark using resnet-18 model
5. (NEW!) **api_tests** - Executes newman test suite for API testing
6. (NEW!) **performance_regression** - Executes performance tests using taurus/jmeter 

> Checkout _workflows_ and _jobs_ section in _config.yml_ for an up to date list 

## scripts
Instead of using inline commands inside _config.yml_, job steps are configured as shell scripts.  
This is easier for maintenance and reduces chances of error in config.yml

## images (WIP)
MMS uses customized docker image for its CircleCI build.  
To make sure MMS is compatible with both Python2 and Python3, we use two build projects.  
We have published two docker images on docker hub for code build
* prashantsail/mms-build:python2.7
* prashantsail/mms-build:python3.6

Following files in the _images_ folder are used to create the docker images
* Dockerfile.python2.7 - Dockerfile for prashantsail/mms-build:python2.7
* Dockerfile.python3.6 - Dockerfile for prashantsail/mms-build:python3.6

## Local CircleCI cli
To make it easy for developers to debug build issues locally, MMS supports CircleCI cli for running a job in a container on your machine.

#### Dependencies
1. CircleCI cli [Quick Install](https://circleci.com/docs/2.0/local-cli/#quick-installation)
2. PyYAML (pip install PyYaml)
3. docker (installed and running)

#### Command
Developers can use the following command to build MMS locally:  
**_./run_circleci_tests.py <workflow_name> <job_name>_**
```bash
$ cd multi-model-server
$ ./run_circleci_tests.py build_and_test_python27 python_tests
$ ./run_circleci_tests.py build_and_test_python36 api_tests
```

###### Checklist
> 1. Make sure docker is running before you start local execution.  
> 2. Docker containers to have **at least 4GB RAM, 2 CPU**.  
> 3. If you are on a network with low bandwidth availability, explicitly pull the docker images beforehand using -  
> docker pull prashantsail/mms-build:python2.7  
> docker pull prashantsail/mms-build:python3.6  

> To avoid Pull Request build failures on github, developers should always make sure that their local builds pass.