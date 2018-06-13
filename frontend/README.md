Model Server REST API endpoint
==============================

## Quick Start

### Building frontend

You can build frontend using gradle:

```sh
$ cd frontend
$ ./gradlew build
```

You will find a jar file in frontend/server/build/libs file.

### Starting frontend

Frontend web server using a configuration file to controll the behavior of the frontend web server.
An sample config.properties can be found in frontend/server/src/test/resources/config.properties.
This configure will load a noop model by default. The noop model file is located in frontend/modelarchive/src/test/resources/model/noop-v0.1.model.

```sh
$ export MMS_CONFIG_FILE=~/source/mxnet-model-server/frontend/server/src/test/resources/config.properties
$ cd frontend/server
$ java -jar build/libs/server-1.0.jar
```
