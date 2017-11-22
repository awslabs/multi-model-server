# Use JMeter to do load testing for Deep Model Server

[JMeter](http://jmeter.apache.org/) is a widely used load testing tool. This is a simple example for how we used JMeter to load test MMS.

## Quick start
To load test MMS serving your model, you can use the included run_load_test.sh.

As an example, here's how you would test MMS running on ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com, hosting a resnet-18 model.

```shell
./run_load_test.sh -i ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com -c 100 -n 20 -f report.csv
```

```
Arguments:
-i : ip address or DNS name
-c : concurrency, number of threads
-n : number of requests send by each thread
-f : output file name

optional:
-p : port that hosting the endpoint
-o : raw result tree output filename
```

This will start `100` threads concurrently and each threads will simulate `20` requests. The aggregated report will be write to `./report/report.csv` and the JMeter logs will be write to
`./log/jmeter.log`.

## Prerequisite
To setup the JMeter, you need to get JMeter with Plugin. On Mac OS you can use `homebrew` to get JMeter with Plugin by running:
```
brew install jmeter --with-plugins
```

Then you need to find the path of `CMDRunner.jar`. You can use `brew list jmeter`. For example, on my machine I will get the following result.
```
/usr/local/Cellar/jmeter/3.3/bin/jmeter
/usr/local/Cellar/jmeter/3.3/libexec/backups/ (20 files)
/usr/local/Cellar/jmeter/3.3/libexec/bin/ (760 files)
/usr/local/Cellar/jmeter/3.3/libexec/docs/ (1888 files)
/usr/local/Cellar/jmeter/3.3/libexec/extras/ (20 files)
/usr/local/Cellar/jmeter/3.3/libexec/lib/ (169 files)
/usr/local/Cellar/jmeter/3.3/libexec/licenses/ (50 files)
/usr/local/Cellar/jmeter/3.3/libexec/printable_docs/ (70 files)
/usr/local/Cellar/jmeter/3.3/libexec/serveragent/ (13 files)
```

The `CMDRUnner.jar` is located in `/usr/local/Cellar/jmeter/3.3/libexec/lib/ext`, then you need to replace the command_runner location is the `run_load_test.sh` file line 38.

## Modifying the test plan
You can use JMeter's GUI by running command `jmeter` so that you can modify the test plan `test_mms.jmx`. For the current setting we are testing a HTTP POST request that send multi-part form-data to the server. The testing file is hard coded in `run_load_test.sh` that `inputfile="$curr_dir/test.jpg"`, you can change it to the image that you are testing.
