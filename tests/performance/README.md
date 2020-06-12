# Performance Regression Suite

This test suite helps in running the load tests and monitoring the process, sub-process and system wide metrics. It allows to specify the pass/fail criteria for metrics in the test case.
We use Taurus test automation framework to run the test cases and metrics monitoring.

## How to run the test suite
To run the test suite just invoke the [run_perfomance_suite.py](run_perfomance_suite.py). You will have to provide the artifacts-dir path to store the test case results.
You can specify test cases to be run by providing 'test-dir' (default='monitoring/tests') and 'pattern' (default='*.yaml'). For other options use '--help' option.   

The script starts the server monitoring agent, collects all the test cases, executes them and then produces Junit XML and HTML report in artifacts-dir.  

**Note**: The script assumes that Model Server is already started. The different JMeter test case parameters such as Model Server Host, Port, image path are specified in test yamls. Modify as per your setup.
In future, a global config file will be provided for commonly used parameters.


```bash
python run_perfomance_suite.py --artifacts-dir='<path>'
```

#### To know more about the Test Suite follow the guide below:

## Taurus

[Taurus](https://gettaurus.org) is Apache 2.0 Licenced, test automation framework which extends and abstracts
different popular load testing frameworks such as JMeter, Locust etc. Taurus simplifies creating, running and analyzing the 
performance tests.

## Installation
Refer the Taurus installation guide [here](https://gettaurus.org/install/Installation/).


## How to create and run tests
To run a metrics monitoring test case, you need to specify below in Taurus test case yaml.
1. Load test case scenario
2. Metrics to monitor
3. Pass/fail criteria


#### 1. Load test case scenario
You can specify the test scenarios, in the scenario section of the yaml.

There are multiple ways you can specify your test scenario.
1. Use existing JMeter/locust test script

    Taurus provides support for different executors such as JMeter. You can use test script written in those frameworks as it is.
    Details about executor types are provided [here](https://gettaurus.org/docs/ExecutionSettings/).
    Details about how to run an existing JMeter script are provided [here](https://gettaurus.org/docs/JMeter/). 
    
    To get you started quickly, we have provided a sample JMeter script and a Taurus yaml file [here](tests/register_and_inference.jmx) and [here](tests/call_jmx.yaml) .
    
    Here is how the call_jmx.yaml looks like. Adjust the module/jmeter/properties section as per your environment. 
    
    ```yaml
    execution:
    - concurrency: 1
      ramp-up: 1s
      hold-for: 40s
      scenario: Inference
    scenarios:
      Inference:
        script: register_and_inference.jmx
    modules:
      jmeter:
        properties:
          hostname : 127.0.0.1
          port : 8080
          management_port : 8081
          protocol : http
          input_filepath : kitten.jpg
    
    ```
    
    Use Taurus command below to run the test case:
    
    ```bash
    bzt call_jmx.yaml
    ```

2. Write a Taurus script

    You can also write a Taurus script for a specific executor type.
    For quick reference, we have provided a Taurus script with JMeter executor [here](tests/inference.yaml).
    
    Below is the yaml. 
    
    ```yaml
    execution:
    - concurrency: 4
      ramp-up: 1s
      hold-for: 20s
      scenario: Inference
    scenarios:
      Inference:
        requests:
        - follow-redirects: true
          label: Inference Request
          method: POST
          url: ${__P(protocol,http)}://${__P(hostname,127.0.0.1)}:${__P(port,8080)}/predictions/${model}
        store-cache: false
        store-cookie: false
        use-dns-cache-mgr: false
        variables:
          model: ${__P(model_name,squeezenet_v1.1)}
    
    modules:
      jmeter:
        properties:
          input_filepath : kitten.jpg
          model_name : squeezenet
    
    ```

    Use command Taurus command below to run the test yaml. Note this test script assumes squeezenet model is already registered.
    
    ```bash
    bzt inference.yaml
    ```

#### 2. Metrics to monitor
You can specify the different metrics to monitor in services/monitoring section of the yaml.

Metrics can be monitored in two ways:
1. Standalone monitoring server

    If your server is hosted on different machine, you will be using this method. Before running the test case
    you have to start a [metrics_monitoring_server.py](metrics_monitoring_server.py) script. It will be communicating with Taurus test client over sockets.
    The address and port(default=9009) of the monitoring script should be specified in test case yaml. 
    
    To install monitoring server dependencies, use the following command
    ```bash   
    pip install -r requirements.txt
    ```    
   
    To start monitoring script on server use command below:
    
    ```bash   
    python benchmarks/monitoring/metrics_monitoring_server.py
    ```
    
    Test yaml with monitoring section config. Complete yaml can be found [here](tests/inference_server_monitoring.yaml)
    
    ```yaml 
    
    services:
      - module: monitoring
        server-agent:
          - address: localhost:9009 # metric monitoring service address
            label: mms-inference-server  # if you specify label, it will be used in reports instead of ip:port
            interval: 1s    # polling interval
            logging: True # those logs will be saved to "SAlogs_192.168.0.1_9009.csv" in the artifacts dir
            metrics: # metrics should be supported by monitoring service
              - sum_cpu_percent # cpu percent used by all the mms server processes and workers
              - sum_memory_percent
              - sum_num_handles
              - server_workers # no of mms workers
    ```
    
    Use command Taurus command below to run the test yaml and observe the Metrics widget on CLI live report.
    
    ```bash
    bzt inference_server_monitoring.yaml
    ```


2. Taurus local monitoring plugin

    If your test client is running on the server itself, you may want to use this method.
    We have provided a custom Taurus plugin as [metrics_monitoring_taurus.py](metrics_monitoring_taurus.py). Make sure that the benchmarks/monitoring folder 
    is in PYTHONPATH. You need to specify the monitoring class.
    
    Use command below to add update PYTHONPATH.
    
    ```bash
     export MMS_HOME=<MMS_HOME_PATH>
     export PYTHONPATH=$MMS_HOME/benchmarks/monitoring:$PYTHONPATH
    ```
    
    Relevant test yaml sections. Test yaml can be found [here](tests/inference_taurus_local_monitoring.yaml)
    
    ```
    modules:
      server_local_monitoring:
        # metrics_monitoring_taurus and dependencies should be in python path
        class : metrics_monitoring_taurus.Monitor # monitoring class.
    
    services:
      - module: server_local_monitoring # should be added in modules section
        ServerLocalClient: # keyword from metrics_monitoring_taurus.Monitor
        - interval: 1s
          metrics:
          - cpu
          - disk-space
          - mem
          - sum_memory_percent
    
    ```

#### 3. Pass/Fail criteria
You can specify the pass/fail criteria for the test cases.
Read more about it [here](https://gettaurus.org/docs/PassFail/)

Relevant test yaml section:
```yaml
reporting:
- module: passfail
  criteria:
  - class: bzt.modules.monitoring.MonitoringCriteria
    subject: mms-inference-server/sum_num_handles
    condition: '>'
    threshold: 180
    timeframe: 1s
    fail: true
    stop: true

```

Test yamls can be found [here](tests/inference_server_monitoring_criteria.yaml) and [here](tests/inference_taurus_local_monitoring_criteria.yaml).
Use command below to run the test case

```bash
bzt inference_server_monitoring_criteria.yaml
bzt inference_taurus_local_monitoring_criteria.yaml
```
