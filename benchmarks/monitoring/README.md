# Metrics Monitoring Suite

The Metrics Monitoring suite helps in monitoring the process, sub-process and system wide metrics while performance test
cases are running. It allows to specify the pass/fail criteria for metrics in the test case.
We use Taurus test automation framework to run the test cases and metrics monitoring.


## Taurus

[Taurus](https://gettaurus.org) is Apache 2.0 Licenced, test automation framework which extends and abstracts
different popular load testing frameworks such as JMeter, Locust etc. Taurus simplifies creating, running and analyzing the 
performance tests.

## Installation
Refer the Taurus installation guide [here](https://gettaurus.org/install/Installation/).


## How to run tests
To run a metrics monitoring test case, you need to specify below in Taurus test case yaml.
1. Load test case scenario
2. Metrics to monitor
3. Pass/fail criteria

#### 1. Load test case scenario
You can specify the test scenarios, in the scenario section of the yaml.

There are multiple ways you can specify your test scenario.
1. Use existing Jmeter/locust test script

    Taurus provides support for different executors such as JMeter so you can use test script written in those frameworks as it is.
    Details about executor types are provided [here](https://gettaurus.org/docs/ExecutionSettings/).
    Details about how to run an existing JMeter script are provided [here](https://gettaurus.org/docs/JMeter/). 
    
    To get you started quickly, we have provided a sample JMeter Script and a Taurus yaml file here and here.
    Use shell command below to run the test case:
    ```
    bzt call_jmx.yaml
    ```
    
    Here is how the Taurus yaml looks like.
    ```yaml code
    
    ```

2. Write a Taurus script

    You can also write a Taurus script for a specific executor type.
    To get you started quickly, we have provided a Taurus script with JMeter executor.
    
    ```
    bzt inference_test.yaml
    ```
    ```yaml code
    
    ```


#### 2. Metrics to monitor
You can specify the different metrics to monitor in services/monitoring section of the yaml.

Metrics can be monitored in two ways:
1. Standalone monitoring server

    If your server is hosted on different machine, you will be using this method. Before running the test case
    you have to start a metric_monitoring_server.py script. It will be communicating with Taurus test client over sockets.
    The address and port of the monitoring script should be specified in test case yaml. 
    
    To start monitoring script on server use command below:
    ```
    sudo python metric_monitoring_server.py
    ```
    
    Sample test yaml:
    ```yaml code
    
    ```

2. Taurus local monitoring plugin

    If your test client is running on the server itself, you may want to use this method.
    We have provided a custom Taurus plugin as metrics_monitoring_taurus.py. Make sure that the benchmarks/monitoring folder 
    is in PYTHON_PATH. You need to specify the monitoring class.
    
    Sample test yaml:
    ```yaml code
    
    ```

#### 3. Pass/Fail criteria
You can specify the pass/fail criteria for the test cases.
Read more about it [here](https://gettaurus.org/docs/PassFail/)

Sample test yaml:
```yaml code

```
