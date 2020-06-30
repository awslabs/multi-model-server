# Performance Regression Suite

## Motivation
The goal of this test suite is to ensure that performance regressions are detected early on. Ideally, with every commit 
made into the source control system. 

The salient features of the performance regression suite are

* Non-intrusive - Does not need any code-changes or instrumentation on the server being monitored. 
* It can be used to monitor a wide variety of server metrics - memory, cpu, io - in addition to 
traditional API level metrics such as latency, throughput etc. 
* It is easy to add custom metrics. For example, in MMS server, `the number of workers spawned` would be an interesting 
metric to track. The platform allows for easy addition of these metrics.
* Test cases are specified in human readable yaml files. Every test case has a pass or fail status. This is determined 
by evaluating expressions specified in the test case. Every expression checks metrics against threshold values. For 
example, `memory consumed by all workers < 500M`, `number of worker processes < 3`.
* Test cases execute against compute environments. The threshold values are specific to the compute environment. It is
possible to specify multiple compute environments against which the test cases will run. It follows that each compute 
environment, will have its own threshold values.
* This suite leverages the open source [Taurus framework](https://gettaurus.org/). 
* This suite extends the Taurus framework in the following ways
   * Adds resource monitoring service. This allows MMS specific metrics to be added. 
   * Environments as described earlier.
   * Specification of pass/fail criterion between two commits. For example, memory consumed by workers should not 
   increase by more than 10% between two commits for the given test case.
   * Custom reporting of results.
   
The building blocks of the performance regression suite and flow is captured in the following drawing

![](assets/blocks.png) 

## Quickstart

### A. Installation
1. Install Taurus. Refer the [link](https://gettaurus.org/docs/Installation/) for more details on installation.
   ```bash   
    pip install bzt # Needs python3.6+
    ``` 
2. Install performance regression suite dependencies.
   ```bash 
    export MMS_HOME=<MMS_HOME_PATH>
    pip install -r $MMS_HOME/tests/performance/requirements.txt
    ``` 
3. Make sure that `git` is installed and the test suites are run from the MMS working directory.

### B. Running the test suite
1. Make sure parameters set in [tests/common/global_config.yaml](tests/performance/tests/global_config.yaml) are correct.
2. To run the test suite execute [run_performance_suite.py](run_performance_suite.py) with the following 
parameters

   * `--artifacts-dir` or `-a` is a directory where the test case results will be stored. The default value is 
`$MMS_HOME/tests/performance/run_artifacts`.  

   * `--test-dir` or `-t` is a directory containing the test cases. The default value is 
`$MMS_HOME/tests/performance/tests`.
 
   * `--pattern` or `-p` glob pattern picks up certain test cases for execution within the `test-dir`. The default value picks up 
all test cases.
 
    * `--exclude-pattern` or `-x` glob pattern excludes certain test cases for execution within the `test-dir`. 
The default value excludes nothing.
 
   * `--env-name` or `-e` specifies the environment name to use while running the test cases. The environment name is the name of 
the file (minus the extension) found inside the environments folder in each test case. They encapsulate parameter 
values which are specific to the execution environment. This is a mandatory parameter.   

   The script does the following:  
   1. Starts the metrics monitoring server.
   2. Collects all the tests from test-dir satisfying the pattern
   3. Executes the tests
   4. Generates artifacts in the artifacts-dir against each test case.  

3. Check the console logs, $artifacts-dir$/<run-dir>/performance_results.html report, comparison.csv, comparison.html 
and other artifacts.

**Steps are provided below**

```bash
export MMS_HOME=<MMS_HOME_PATH>
cd $MMS_HOME/tests/performance
 
# Note that MMS server started and stopped by the individual test suite.
# check variables such as MMS server PORT etc 
# vi tests/common/global_config.yaml 

#all tests
python -m run_performance_suite -e xlarge

#run a specific test 
python -m run_performance_suite -e xlarge -p inference_single_worker

```

### C. Understanding the test suite artifacts and reports
1. The $artifacts-dir$/<run-dir>/performance_results.html is a summary report of the test run. 
2. Each test yaml is treated as a test suite. Each criteria in the test suite is treated as a test case. 
If the test suite does not specify any criteria, then the test suite is reported as skipped with 0 test cases.
3. For each test suite, a sub-directory is created containing relevant run artifacts. Important files in this directory are
   * metrics.csv -- contains the values of the various system-monitored metrics over time
   * finals_stats.csv -- contains the values of the various api metrics over time  
4. The $artifacts-dir$/<run-dir>/comparison_results.html is a summary report which shows performance difference between
the last two commits.
5. The run completes with a console summary of the performance and comparision suites which have failed
![](assets/console.png) 

## Add a new test

Follow these three steps to add a new test case to the test suite.

1. Add scenario (a.k.a test suite)
2. Add metrics to monitor
3. Add pass/fail criteria (a.k.a test case)


#### 1. Add scenario (a.k.a test suite)
Create a folder for the test under `test_dir` location. A test generally comprises of a jmeter file - containing the 
load scenario and a yaml file which contains test scenarios specifying the conditions for failure or success. The
file-names should be identical to the folder name with their respective extensions. 

An example [jmeter script](tests/examples_starter/examples_starter.jmx) 
and a [scenario](tests/examples_starter/examples_starter.yaml) is provided as a template to get started.
    
Please note that various global configuration settings used by examples_starter.jmx script are specified in 
[tests/global_config.yaml](tests/performance/tests/global_config.yaml) file.
    
 ```tests/examples_starter/examples_starter.yaml
 execution:
 - concurrency: 1
   ramp-up: 1s
   hold-for: 40s
   scenario: Inference

 scenarios:
   Inference:
     script: examples_starter.jmx

 ```
    
To execute this test suite, run the following command
    
 ```bash
 export MMS_HOME=<MMS_HOME_PATH>
 cd $MMS_HOME/tests/performance
 python -m run_performance_suite -p examples_starter -e xlarge
 ```

**Note**:
Taurus provides support for different executors such as JMeter. Supported executor types can be found [here](https://gettaurus.org/docs/ExecutionSettings/).
Details about how to use an existing JMeter script are provided [here](https://gettaurus.org/docs/JMeter/). 


#### 2. Add metrics to monitor
Specify the metrics of interest in the services/monitoring section of the yaml.

1. Standalone monitoring server

   Use this technique if MMS and the tests execute on different machines. Before running the test cases, 
   please start the [metrics_monitoring_server.py](metrics_monitoring_server.py) script. It will communicate server 
   metric data with the test client over sockets. The monitoring server runs on port 9009 by default.
    
   To start the monitoring server, run the following commands on the MMS host:
    ```bash 
    export MMS_HOME=<MMS_HOME_PATH>
    pip install -r $MMS_HOME/tests/performance/requirements.txt
    python $MMS_HOME/tests/performance/metrics_monitoring_server.py --start
    ```     
      
   The monitoring section configuration is shown below. 
    
    ```yaml
    services:
      - module: monitoring
        server-agent:
          - address: <mms-host>:9009 # metric monitoring service address
            label: mms-inference-server  # Specified label will be used in reports instead of ip:port
            interval: 1s    # polling interval
            logging: True # those logs will be saved to "SAlogs_192.168.0.1_9009.csv" in the artifacts dir
            metrics: # metrics should be supported by monitoring service
              - sum_cpu_percent # cpu percent used by all the mms server processes and workers
              - sum_memory_percent
              - sum_num_handles
              - server_workers # no of mms workers
    ```
   The complete yaml can be found [here](tests/examples_remote_monitoring/examples_remote_monitoring.yaml)
    
   Use the command below to run the test suite.
    
    ```bash
    export MMS_HOME=<MMS_HOME_PATH>
    cd $MMS_HOME/tests/performance
    python -m run_performance_suite -p examples_remote_monitoring -e xlarge
    ```

2. Local monitoring plugin

   Use this technique if both MMS and the tests run on the same host.   
   The monitoring section configuration is shown below.
    
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
   The complete yaml can be found [here](tests/examples_local_monitoring/examples_local_monitoring.yaml).
    
   Use the command below to run the test suite.
    
    ```bash
    export MMS_HOME=<MMS_HOME_PATH>
    cd $MMS_HOME/tests/performance
    python -m run_performance_suite -p examples_local_monitoring -e xlarge
    ```

#### 3. Add pass/fail criteria (a.k.a test case)

1. **Specify the pass/fail criteria**. Each pass-fail criterion maps to a test case in the generated report. We leverage the
pass-fail module from Taurus to achieve this functionality. More details can be found [here](https://gettaurus.org/docs/PassFail/).

   A sample criterion is shown below

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

2. Specify the pass/fail criterion vis-a-vis a prior run. On completion, the test suite runner script compares the 
monitoring metrics with values from a previous run which was executed on same environment. The run results are stored 
in either a local folder or a S3 bucket based on the `compare-local` option. Metrics which have 'diff_percent' value 
specified in the pass/fail criterion are used for comparison with the previous run. 

   A sample criterion is shown below
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
    Note that 
    1. At least one test suite run on the same environment should have happened in order to do the comparison.
    2. The $artifacts-dir$/<run-dir>/comparison_results.html is a summary report which shows performance difference 
    between the last two commits.
    3. The test case fails if the diff_percent is greater than the specified value across runs.

3. Metrics available for pass-fail criteria  
  
   **System Metrics**
   > disk_used, memory_percent, read_count, write_count, read_bytes, write_byte

   | Syntax | Examples |
   | ------ | -------- |
   | system_{metricname} | system_disk_used, system_memory_percent, system_write_count |

   **Process Metrics**
   > cpu_percent, memory_percent, cpu_user_time, cpu_system_time, cpu_iowait_time, memory_rss, memory_vms, io_read_count, io_write_count, io_read_bytes, io_write_bytes, file_descriptors, threads

   - Frontend. Represents the Java process hosting the REST APIs 

     | Syntax | Examples |
     | ------ | -------- |
     | frontend_{metricname} | frontend_cpu_percent, frontend_memory_percent, frontend_cpu_iowait_time, frontend_memory_rss, frontend_io_write_bytes, frontend_threads |

   - Workers. Represents the python worker processes. Metrics for worker(s) are always available with an aggregate  
     > Aggregates  
     > sum, avg, min, max

     | Syntax | Examples |
     | ------ | -------- |
     | {aggregate}\_workers\_{metricname} | total_workers, sum_workers_memory_percent, avg_workers_iowait_time, min_workers_io_write_bytes, max_workers_threads |

   - All (Frontend + Workers). Represents aggregate metrics for both frontend and worker processes.
  
     | Syntax | Examples |
     | ------ | -------- |
     | {aggregate}\_all\_{metricname} | sum_all_memory_percent, avg_all_iowait_time, min_all_io_write_bytes, max_all_threads |

   - Miscellaneous
      * total_processes - Total number of processes spawned for frontend & workers
      * total_workers - Total number of workers spawned
      * orphans - Total number of orphan processes

## Test Strategy & Cases
More details about our testing strategy and test cases can be found [here](TESTS.md) 

## FAQ

Q1. Is it possible to use the performance regression framework to test MMS on Python2.7?

Yes. Even though, the performance regression framework needs Python 3.7+ (as Taurus requires Python 3.7+), there are two
possible ways to achieve this
* Please create a Python 2.7 virtual env which runs MMS and a Python 3.7 virtual env which runs 
  the test framework and test cases.
* Alternatively, deploy the standalone monitoring agent on the MMS instance and run the test cases against the remote
server. Note that the standalone monitoring agent works on both Python 2/3. 



