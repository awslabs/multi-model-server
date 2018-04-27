#!/bin/bash

#Get option to run test on GPU/CPU image
while getopts "g:" option; do
  case $option in
    g)      use_gpu=$OPTARG;;
  esac
done
echo $use_gpu


# Install Docker CLI in case it is missing
docker run --name hw hello-world
if [[ `echo $?` != 0 ]] ;
then
    echo "Something went wrong with docker CLI installation"
    # Install DOCKER CLI
    apt-get remove docker docker-engine docker.io
    apt-get update
    apt-get install -y \
            apt-transport-https \
            ca-certificates \
            curl \
            software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg |  apt-key add -
    apt-key fingerprint 0EBFCD88
    add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install  -y docker-ce=18.03.1~ce-0~ubuntu

else
    echo "Docker is happy"
fi


# Pull correct docker images
if [ "$use_gpu" == "0" ]; then
    docker pull awsdeeplearningteam/mms_cpu
elif [ "$use_gpu" == "1" ]; then
     nvidia-docker --help
     if [[ `echo $?` != 0 ]] ; then
         curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
         distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
         curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |tee /etc/apt/sources.list.d/nvidia-docker.list
         apt-get update
         apt-get install -y nvidia-docker2
         pkill -SIGHUP dockerd
    else
         echo "nvidia docker already installed"
    fi
    docker pull awsdeeplearningteam/mms_gpu
fi

# JMETER installation
echo "jmeter installation"
if [ -d "$HOME/apache-jmeter-4.0" ]; then
    echo "JMeter already installed"
else
    echo "Installing jmeter"
    cd /tmp
    wget -c http://ftp.ps.pl/pub/apache//jmeter/binaries/apache-jmeter-4.0.tgz
    wget https://www.apache.org/dist/jmeter/binaries/apache-jmeter-4.0.tgz.asc
    wget -O - https://www.apache.org/dist/jmeter/KEYS |gpg --import
    gpg --verify apache-jmeter-4.0.tgz.asc
    cd $HOME
    tar -zxf /tmp/apache-jmeter-4.0.tgz
    export JMETER_HOME=$HOME/apache-jmeter-4.0
    export PATH=$JMETER_HOME/bin:$PATH
    apt-get install unzip
    cd /tmp
    wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-Standard-1.4.0.zip
    wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-Extras-1.4.0.zip
    wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-ExtrasLibs-1.4.0.zip
    wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-WebDriver-1.4.0.zip
    wget http://jmeter-plugins.org/files/packages/jpgc-synthesis-2.1.zip
    unzip -n jpgc-synthesis-2.1.zip -d /tmp
    unzip -n JMeterPlugins-Standard-1.4.0.zip -d $HOME/apache-jmeter-4.0
    unzip -n JMeterPlugins-Extras-1.4.0.zip -d $HOME/apache-jmeter-4.0
    unzip -n JMeterPlugins-ExtrasLibs-1.4.0.zip -d $HOME/apache-jmeter-4.0
    unzip -n JMeterPlugins-WebDriver-1.4.0.zip -d $HOME/apache-jmeter-4.0
    cp ./lib/jmeter-plugins-cmn-jmeter-0.4.jar $HOME/apache-jmeter-4.0/lib/
    cp ./lib/ext/* $HOME/apache-jmeter-4.0/lib/ext
fi
# end of jmeter installation

echo "setting up conf"
#Change conf files for running resnet
cd /
if [ -d "/mxnet-model-server" ]; then
    git clone https://github.com/awslabs/mxnet-model-server.git
fi
sed -i -e 's/squeezenet=https:\/\/s3.amazonaws.com\/model-server\/models\/squeezenet_v1.1\/squeezenet_v1.1.model/resnet-18=https:\/\/s3.amazonaws.com\/model-server\/models\/resnet-18\/resnet-18.model /g'\
 /mxnet-model-server/docker/mms_app_cpu.conf
sed -i -e 's/squeezenet=https:\/\/s3.amazonaws.com\/model-server\/models\/squeezenet_v1.1\/squeezenet_v1.1.model/resnet-18=https:\/\/s3.amazonaws.com\/model-server\/models\/resnet-18\/resnet-18.model /g'\
 /mxnet-model-server/docker/mms_app_gpu.conf
sed -i -e 's/\/usr\/local\/Cellar\/jmeter\/3.3\/libexec\/lib\/ext\/CMDRunner.jar/\/home\/ubuntu\/apache-jmeter-4.0\/lib\/ext\/CMDRunner.jar /g' /mxnet-model-server/load-test/run_load_test.sh


file="$HOME/log_results"

if [ -f $file ] ; then
    rm $file
fi

#start the container test
for workers in 50 100
do
    for requests in 25 50 100
    do
        echo "*************************************** Workers $workers and reqs $requests   ****************************************************"
        if [ "$use_gpu" == "0" ]; then
            docker run  --name mms -v /mxnet-model-server:/mxnet-model-server  -itd -p 80:8080 awsdeeplearningteam/mms_cpu
            docker exec mms pip install -U -e /mxnet-model-server/.
            docker exec mms mxnet-model-server start --mms-config /mxnet-model-server/docker/mms_app_cpu.conf >& /dev/null &
            sleep 30
        elif [ "$use_gpu" == "1" ]; then
            nvidia-docker run --name mms -v /mxnet-model-server:/mxnet-model-server -itd -p 80:8080 awsdeeplearningteam/mms_gpu
            nvidia-docker exec mms pip install -U -e /mxnet-model-server/.
            nvidia-docker exec mms pip uninstall --yes mxnet-cu90mkl
            nvidia-docker exec mms pip uninstall --yes mxnet
            nvidia-docker exec mms pip install mxnet-cu90mkl
            nvidia-docker exec mms mxnet-model-server start --mms-config /mxnet-model-server/docker/mms_app_gpu.conf >& /dev/null &
            sleep 90
        fi

        cd /mxnet-model-server/load-test
        export JMETER_HOME=$HOME/apache-jmeter-4.0
        export PATH=$JMETER_HOME/bin:$PATH
        ./run_load_test.sh -i 127.0.0.1 -c $workers -n $requests -f report.csv -p 80
        echo "done report generation"
        OLDIFS= $IFS
        IFS=","
        read -r sampler_label aggregate_report_count average aggregate_report_median aggregate_report_90_line aggregate_report_95_line aggregate_report_99_line aggregate_report_min aggregate_report_max a\
ggregate_report_error aggregate_report_rate aggregate_report_bandwidth aggregate_report_stddev < <(tail -n1 ./report/report.csv)

        echo -e "{\
        Throughput_concurrency_${workers}_req_${requests} :$aggregate_report_rate,\
        Average_latency_concurrency_${workers}_req_${requests} :$average,\
        Median_latency_concurrency_${workers}_req_${requests} :$aggregate_report_median,\
        P90_latency_concurrency_${workers}_req_${requests} :$aggregate_report_90_line,\
        Error_rate_concurrency_${workers}_req_${requests} :$aggregate_report_error}\n" >> $HOME/log_results
        IFS=$OLDIFS

        docker rm -f mms
    done
done
docker rm -f hw
exit 0
