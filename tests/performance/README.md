# Performance Regression Suite

This test suite helps in running the load tests and monitoring the process and system wide metrics. It allows to specify the pass/fail criteria for metrics in the test case.
We use Taurus with JMeter as a test automation framework to run the test cases and metrics monitoring.

## How to run the test suite
To run the test suite you need to execute the [run_performance_suite.py](run_performance_suite.py). You will have to provide the artifacts-dir path to store the test case results.
You can specify test cases to be run by providing 'test-dir' with default as '$MMS_HOME/tests/performance/tests' and 'pattern' with default as '*.yaml'. For other options use '--help' option.   

Script does the following:  
1. Optionally but by default starts the metrics monitoring server
2. Collects all the test yamls from test-dir satisfying the pattern
3. Executes test yamls
4. Generates Junit XML and HTML report in artifacts-dir.  

### A. Installation Prerequisites
1. Install Taurus. The Taurus needs Python3 but since your tests and MMS instance can run in different virtual environement or machine, 
you can configure system such that tests are running on Python3 and MMS instance can run on Python 2 or 3.  
Refer the [link](https://gettaurus.org/docs/Installation/) for more details on installation.
   ```bash   
    pip install bzt # Needs python3.6+
    ``` 
2. Install other dependencies.
   ```bash 
    export MMS_HOME=<MMS_HOME_PATH>
    pip install -r $MMS_HOME/tests/performance/requirements.txt
    ``` 
3. Make sure that `git` is installed and the test suites are run from the MMS working directory. This is used to compare performance regresssions across runs and the run artifacts are stored in a folder which have the commit SHA.

### B. Running the test suite
1. Make sure parameters set in the [tests/common/global_config.yaml](tests/performance/tests/global_config.yaml) are correct.
2. Run the test suite runner script
3. Check the console logs, $artifacts-dir$/<run-dir>/junit.html report, comparison.csv and other artifacts.

    **Steps are provided below**
    ```bash
    export MMS_HOME=<MMS_HOME_PATH>
    cd $MMS_HOME/tests/performance
    
     
    # Note that MMS server started and stopped by the individual test suite.
    # check variables
    # vi tests/common/global_config.yaml 

    
    python -m run_performance_suite  --pattern='*'
    ```

### C. Understanding the test suite artifacts and reports
1. The $artifacts-dir$/<run-dir>/junit.html contains the summary report of the test run. Note that each test yaml is treated as a 
test suite. Different criteria in the yaml are treated as test cases. If criteria is not specified in the yaml, test suite is marked as skipped with 0 test cases.
2. For each test suite a sub-directory is created with artifacts for it.  
3. The comparison.csv contains diff for monitoring metrics between an ongoing run and a previous run which was ran for same MMS server. 


## How to add test case to test suite.

To add test case follow steps below.
1. Add scenario
2. Add metrics to monitor
3. Add pass/fail criteria


#### 1. Add scenario
You can specify the test scenarios, in the scenario section of the yaml.
To get you started quickly, we have provided a sample JMeter script and a Taurus yaml file [here](tests/call_jmx/register_and_inference.jmx) and [here](tests/call_jmx/call_jmx.yaml) .
    
Here is how the sample call_jmx.yaml looks like. Note variables used by jmx script are specified in [tests/global_config.yaml](tests/performance/tests/global_config.yaml) file.
    
 ```yaml
 execution:
 - concurrency: 1
   ramp-up: 1s
   hold-for: 40s
   scenario: Inference

 scenarios:
   Inference:
     script: register_and_inference.jmx

 ```
    
To run this individual test using Taurus(bzt) run commands below:
    
 ```bash
 export MMS_HOME=<MMS_HOME_PATH>
 cd $MMS_HOME/tests/performance
 python -m run_performance_suite -p call_jmx
 ```

**Note**:
Taurus provides support for different executors such as JMeter. You can use test script written in those frameworks as it is.
Details about executor types are provided [here](https://gettaurus.org/docs/ExecutionSettings/).
Details about how to run an existing JMeter script are provided [here](https://gettaurus.org/docs/JMeter/). 


#### 2. Add metrics to monitor
You can specify different metrics in services/monitoring section of the yaml.
Metrics can be monitored in two ways:
1. Standalone monitoring server

    If your MMS server is hosted on different machine, you will be using this method. Before running the test case
    you have to start a [metrics_monitoring_server.py](metrics_monitoring_server.py) script. It will be communicating with Taurus test client over sockets.
    The address and port(default=9009) of the monitoring script should be specified in test case yaml. 
      
    **Note**: While running Test suite runner script, no need to manually start the monitoring server. The scripts optionally but by default starts and stops it in setup and teardown.
    
    To start monitoring server run commands below:
    ```bash 
    export MMS_HOME=<MMS_HOME_PATH>
    pip install -r $MMS_HOME/tests/performance/requirements.txt
    python $MMS_HOME/tests/performance/metrics_monitoring_server.py --start
    ```     
   
    Sample yaml with monitoring section config. Complete yaml can be found [here](tests/inference_server_monitoring/inference_server_monitoring.yaml)
    
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
    
    Use the command below to run the test yaml.
    
    ```bash
    export MMS_HOME=<MMS_HOME_PATH>
    cd $MMS_HOME/tests/performance
    python -m run_performance_suite -p inference_server_monitoring
    ```


2. Taurus local monitoring plugin

    If your test client is running on the server itself, you may want to use this method.
    We have provided a custom Taurus plugin as [metrics_monitoring_taurus.py](metrics_monitoring_taurus.py). 
    
    **Note**: To know the list of supported/available metrics check [here](metrics_monitoring_taurus.py)  
    **Note**: While running Test suite runner script, no need to manually update the PYTHONPATH. The scripts updates it.
    
    Use commands below to update PYTHONPATH so that plugin gets picked up by Taurus.
    
    ```bash
     export MMS_HOME=<MMS_HOME_PATH>
     export PYTHONPATH=$MMS_HOME/tests/performance:$PYTHONPATH
    ```
    
    Relevant test yaml sections. Test yaml can be found [here](tests/inference_taurus_local_monitoring/inference_taurus_local_monitoring.yaml)
    
    ```yaml
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

    Use command below to run the test yaml
    
    ```bash
    export MMS_HOME=<MMS_HOME_PATH>
    cd $MMS_HOME/tests/performance
    python -m run_performance_suite -p inference_taurus_local_monitoring
    ```

#### 3.1 Add pass/fail criteria
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
    diff_percent : 30

```

#### 3.2 Add pass/fail criteria with previous run
On completion, the test suite runner script compares the monitoring metrics with values from a previous run which was executed on same environment. 
Note that at least one test suite run on the same environment should have happened in order to do the comparison. The run results are stored in either a local folder or a S3 bucket based on the `compare-local` option
Metrics which have 'diff_percent' value specified in the pass/fail criterion are used for comparision with the previous run. See pass/fail criteria [section](#3-add-passfail-criteria)
Below are different options used by run_performance_suite script for coparison.
1. **artifacts-dir**:
This is an optional parameter. The default is './run_artifacts' directory.
A sub directory with '{env_name}_{git_commit_id}_{timestamp}' gets created in the artifacts dir.
2. **env-name**:
This is an optional parameter. The default is current hostname. This should be unique it is used while doing comparison between runs.
Comparison should happen between the runs of same environment.
3. **compare-local/no-compare-local**:
This is an optional parameter. The default is compare-local. If `compare-local` is set,  previous run results from the local `artifacts-dir` folder will be used used. 
`experimental` If no-compare-local is set,  previous run results from the public S3 bucket will be used. `no-compare-local is currently experimental`

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
    diff_percent : 30

```

#### 3.3 Metrics that you can use for passfail criteria  
  **System Metrics**
  > disk_used, memory_percent, read_count, write_count, read_bytes, write_byte

  | Syntax | Examples |
  | ------ | -------- |
  | system_{metricname} | system_disk_used, system_memory_percent, system_write_count |


  **Process Metrics**
  > cpu_percent, memory_percent, cpu_user_time, cpu_system_time, cpu_iowait_time, memory_rss, memory_vms, io_read_count, io_write_count, io_read_bytes, io_write_bytes, file_descriptors, threads

  - Frontend  
    It is a single java process

    | Syntax | Examples |
    | ------ | -------- |
    | frontend_{metricname} | frontend_cpu_percent, frontend_memory_percent, frontend_cpu_iowait_time, frontend_memory_rss, frontend_io_write_bytes, frontend_threads |

  - Workers  
    These are python processes. MMS can have more than one worker  
    Metrics for worker(s) are always available with an aggregate  
    > Aggregates  
    > sum, avg, min, max

    | Syntax | Examples |
    | ------ | -------- |
    | {aggregate}\_workers\_{metricname} | total_workers, sum_workers_memory_percent, avg_workers_iowait_time, min_workers_io_write_bytes, max_workers_threads |

  - All (Frontend + Workers)  
    We can also aggregate metrics for both frontend and worker processes together
  
    | Syntax | Examples |
    | ------ | -------- |
    | {aggregate}\_all\_{metricname} | sum_all_memory_percent, avg_all_iowait_time, min_all_io_write_bytes, max_all_threads |

  - Miscellaneous
     * total_processes - Total number of processes spawned for frontend & workers
     * total_workers - Total number of workers spawned
     * orphans - Total number of orphan processes

## Test Strategy
More details about our testing strategy and test cases can be found [here](TESTS.md) 

## TODOs
1. Auto threshold calculation
