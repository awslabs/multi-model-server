# Use JMeter to do load testing for Deep Model Server

JMeter is a widely used load testing tool. This is a simple example for how we used
JMeter to do load testing for DMS.

## Quick start
Suppose you have hosted your DMS on a remote server, whose DNS is ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com. And also suppose you are hosting `resnet-18` service (so that you can use this test plan to test load without making changes). And you are sending the `test.jpg` image to the `predict` endpoint.

You can used this simple tool to test load by running
```shell
./run_load_test.sh -i ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com -c 100 -n 20 -f report.csv
```

And this will start 100 threads concurrently and each threads will simulate 20 requests. The aggregated report will be write to `report/report.csv` and the JMeter logs will be write to
`log/jmeter.log`.

## Prerequisite
To setup JMeter you need to get Apache JMeter with Plugin. On Mac OS you can use `homebrew` to get JMeter with Plugin by running:
```
brew install jmeter --with-plugins
```

Then you need to find the path of `CMDRunner.jar`. You can use `brew list jmeter`. For example you might get the following result.
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
