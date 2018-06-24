#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` -i ip-address -c concurrency -n number-of-requests-each-thread -f output-filename [-p port] [-o raw-result]"
  echo "Example $0 -i ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com -c 100 -n 20 -f report.csv"
  exit 0
fi

while getopts "i:f:p:c:n:o" option; do
  case $option in
    i)      dns=$OPTARG;;
    f)      filename=$OPTARG;;
    p)      port=$OPTARG;;
    c)      threads=$OPTARG;;
    n)      loops=$OPTARG;;
    o)      raw_output=$OPTARG;;
  esac
done

if [ "$dns" == "" ]; then
  echo "You need to provide the server name that hosting your MMS"
  echo "e.g. ./run_load_test.sh -i ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com"
fi

if [ "$filename" == "" ]; then
  echo "You need to set provide the name of the report file"
  echo "e.g. ./run_load_test.sh -f aggregate_report_c20.csv"
  exit 1
fi

if [ "$threads" == "" ]; then
  threads=20
fi
if [ "$loops" == "" ]; then
  loops=10
fi

# make this var point to your installed CMDRunner location
command_runner="/usr/local/Cellar/jmeter/4.0/libexec/lib/ext/CMDRunner.jar"
curr_dir=$(pwd)

if [ ! -d "$curr_dir/report" ]; then
  mkdir "$curr_dir/report"
fi
if [ ! -d "$curr_dir/log" ]; then
  mkdir "$curr_dir/log"
fi

output="$curr_dir/report/${filename}"
tmpfile="$curr_dir/output.jtl"
logfile="$curr_dir/log/jmeter.log"
inputfile="$curr_dir/test.jpg"

jmeter -n -t "$curr_dir/test_mms.jmx" -Jhostname=$dns -Jport=$port -Jthreads=$threads -Jloops=$loops -Jfilepath=$inputfile -l $tmpfile -j $logfile
java -jar $command_runner --tool Reporter --generate-csv $output --input-jtl $tmpfile --plugin-type AggregateReport
rm $tmpfile
