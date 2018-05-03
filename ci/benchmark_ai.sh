#!/bin/bash

##global variables
use_gpu="0"
report=report.csv
num_workers=(50 100)
num_requests=(25 50 100)

# Install Docker CLI in case it is missing
install_docker(){
    docker run --name hw hello-world >& /dev/null
    if [[ `echo $?` != 0 ]] ;then
       echo "Installing docker cli"
       # Install DOCKER CLI
       apt-get remove docker docker-engine docker.io &&
       apt-get update &&
       apt-get install -y \
            apt-transport-https \
            ca-certificates \
            curl \
            software-properties-common && 
       curl -fsSL https://download.docker.com/linux/ubuntu/gpg |  apt-key add - &&
       apt-key fingerprint 0EBFCD88 &&
       add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
       apt-get update &&
       apt-get install  -y docker-ce=18.03.1~ce-0~ubuntu
       if [[ `echo $?` != 0 ]] ; then
           echo "ERROR: Exiting as docker cli installation failed"
           exit 1
       fi
    else
        echo "Docker already installed"
    fi
}

# Install Nvidia docker
install_nvidia_docker(){
    nvidia-docker --help  >& /dev/null
    if [[ `echo $?` != 0 ]] ; then
        echo "installing nvidia docker"
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - &&
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID) &&
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |tee /etc/apt/sources.list.d/nvidia-docker.list &&
        apt-get update &&
        apt-get install -y nvidia-docker2 &&
        pkill -SIGHUP dockerd
        if [[ `echo $?` != 0 ]] ; then
            echo "ERROR: Exiting as nvidia-docker installation failed"
            exit 1
        fi
    else
        echo "nvidia docker already installed"
    fi

}

pull_docker_images(){
    # Pull correct docker images
    if [ "$use_gpu" == "0" ]; then
        docker pull awsdeeplearningteam/mms_cpu
        if [[ `echo $?` != 0 ]] ; then
            echo "ERROR: Exiting as docker pull failed for awsdeeplearningteam/mms_cpu"
            exit 1
        fi
    elif [ "$use_gpu" == "1" ]; then
        docker pull awsdeeplearningteam/mms_gpu
        if [[ `echo $?` != 0 ]] ; then
            echo "ERROR: Exiting as docker pull failed for awsdeeplearningteam/mms_cpu"
            exit 1
        fi
    fi
}

#JMETER installation
install_jmeter(){
    echo "jmeter installation"
    if [ -d "$HOME/apache-jmeter-4.0" ]; then
        echo "JMeter already installed"
    else
        echo "Installing jmeter"
        cd /tmp &&
        wget -c http://ftp.ps.pl/pub/apache//jmeter/binaries/apache-jmeter-4.0.tgz &&
        wget https://www.apache.org/dist/jmeter/binaries/apache-jmeter-4.0.tgz.asc &&
        wget -O - https://www.apache.org/dist/jmeter/KEYS |gpg --import &&
        gpg --verify apache-jmeter-4.0.tgz.asc &&
        cd $HOME &&
        tar -zxf /tmp/apache-jmeter-4.0.tgz &&
        apt-get install unzip &&
        cd /tmp &&
        wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-Standard-1.4.0.zip &&
        wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-Extras-1.4.0.zip &&
        wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-ExtrasLibs-1.4.0.zip &&
        wget http://jmeter-plugins.org/downloads/file/JMeterPlugins-WebDriver-1.4.0.zip &&
        wget http://jmeter-plugins.org/files/packages/jpgc-synthesis-2.1.zip &&
        unzip -n jpgc-synthesis-2.1.zip -d /tmp &&
        unzip -n JMeterPlugins-Standard-1.4.0.zip -d $HOME/apache-jmeter-4.0 &&
        unzip -n JMeterPlugins-Extras-1.4.0.zip -d $HOME/apache-jmeter-4.0 &&
        unzip -n JMeterPlugins-ExtrasLibs-1.4.0.zip -d $HOME/apache-jmeter-4.0 &&
        unzip -n JMeterPlugins-WebDriver-1.4.0.zip -d $HOME/apache-jmeter-4.0 &&
        cp ./lib/jmeter-plugins-cmn-jmeter-0.4.jar $HOME/apache-jmeter-4.0/lib/ &&
        cp ./lib/ext/* $HOME/apache-jmeter-4.0/lib/ext
        if [[ `echo $?` != 0 ]] ; then
            echo "ERROR: Exiting as jmeter installation failed"
            exit 1
        fi
    fi
# end of jmeter installation
}

prepare_repo(){
    echo "setting up conf"
    #Change conf files for running resnet
    cd /
    if [ ! -d "/mxnet-model-server" ]; then
        git clone https://github.com/awslabs/mxnet-model-server.git
    fi
sed -i -e 's/squeezenet=https:\/\/s3.amazonaws.com\/model-server\/models\/squeezenet_v1.1\/squeezenet_v1.1.model/resnet-18=https:\/\/s3.amazonaws.com\/model-server\/models\/resnet-18\/resnet-18.model /g'\
 /mxnet-model-server/docker/mms_app_cpu.conf &&
sed -i -e 's/squeezenet=https:\/\/s3.amazonaws.com\/model-server\/models\/squeezenet_v1.1\/squeezenet_v1.1.model/resnet-18=https:\/\/s3.amazonaws.com\/model-server\/models\/resnet-18\/resnet-18.model /g'\
 /mxnet-model-server/docker/mms_app_gpu.conf &&
sed -i -e 's/\/usr\/local\/Cellar\/jmeter\/3.3\/libexec\/lib\/ext\/CMDRunner.jar/\/home\/ubuntu\/apache-jmeter-4.0\/lib\/ext\/CMDRunner.jar /g' /mxnet-model-server/load-test/run_load_test.sh
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as preparation of repository failed"
        exit 1
    fi

}

## parse logs
parse_csv_log(){
    echo "start parsing files from $report"
    
    OLDIFS= $IFS
    IFS=","
    read -r sampler_label aggregate_report_count average aggregate_report_median aggregate_report_90_line aggregate_report_95_line aggregate_report_99_line aggregate_report_min aggregate_report_max a\
ggregate_report_error aggregate_report_rate aggregate_report_bandwidth aggregate_report_stddev < <(tail -n1 /mxnet-model-server/load-test/report/$report)

    echo -e "{\
        Throughput_concurrency_${workers}_req_${requests} :$aggregate_report_rate,\
        Average_latency_concurrency_${workers}_req_${requests} :$average,\
        Median_latency_concurrency_${workers}_req_${requests} :$aggregate_report_median,\
        P90_latency_concurrency_${workers}_req_${requests} :$aggregate_report_90_line,\
        Error_rate_concurrency_${workers}_req_${requests} :$aggregate_report_error}\n"
        IFS=$OLDIFS
}

#start the container
start_mms_cpu(){
    echo "starting mms for cpu runs"
    docker run  --name mms -v /mxnet-model-server:/mxnet-model-server  -itd -p 80:8080 awsdeeplearningteam/mms_cpu
    docker exec mms pip install -U -e /mxnet-model-server/.  >& /dev/null
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as mms installation from source failed"
        exit 1
    fi
    docker exec mms mxnet-model-server start --mms-config /mxnet-model-server/docker/mms_app_cpu.conf >& /dev/null &
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as MMS startup failed"
        remove_container_images
        exit 1
    fi
    echo "sleep until mms starts"
    sleep 30
}

start_mms_gpu(){
    echo "starting mms for gpu runs"
    nvidia-docker run --name mms -v /mxnet-model-server:/mxnet-model-server -itd -p 80:8080 awsdeeplearningteam/mms_gpu
    nvidia-docker exec mms pip install -U -e /mxnet-model-server/. >& /dev/null &&
    nvidia-docker exec mms pip uninstall --yes mxnet-cu90mkl  >& /dev/null &&
    nvidia-docker exec mms pip uninstall --yes mxnet  >& /dev/null &&
    nvidia-docker exec mms pip install mxnet-cu90mkl  >& /dev/null
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as mxnet-cu90mkl installation failed"
        remove_container_images
        exit 1
    fi
    nvidia-docker exec mms mxnet-model-server start --mms-config /mxnet-model-server/docker/mms_app_gpu.conf >& /dev/null &
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as MMS startup failed"
        remove_container_images
        exit 1
    fi
    echo "sleep until mms starts"
    sleep 90
}

start_jmeter(){
    echo "start sending jmeter requests"
    cd /mxnet-model-server/load-test
    ./run_load_test.sh -i 127.0.0.1 -c $workers -n $requests -f $report -p 80
    if [[ `echo $?` != 0 ]] ; then
        echo "ERROR: Exiting as jmeter startup failed"
        remove_container_images
        exit 1
    fi
    echo "done report generation"
}

remove_container_images(){
     service docker stop && service docker start
     echo "docker restarted"
     docker rm -f mms
     echo "container removed"
}

run_test(){
    export JMETER_HOME=$HOME/apache-jmeter-4.0
    export PATH=$JMETER_HOME/bin:$PATH
    for workers in ${num_workers[@]}
    do
        for requests in ${num_requests[@]}
        do
            echo "*************************************** Workers $workers and reqs $requests   ****************************************************"
            if [ "$use_gpu" == "0" ]; then
                start_mms_cpu
            elif [ "$use_gpu" == "1" ]; then
                start_mms_gpu
            fi
            start_jmeter
            parse_csv_log
            service docker stop && service docker start
            echo "docker restarted"
            docker rm -f mms
            echo "container removed"
        done
    done
}

main(){
    #Get option to run test on GPU/CPU image
    while getopts "g:" option; do
	echo "Parsing Args"
        case $option in
            g)      use_gpu=$OPTARG;;
        esac
    done
    echo "use_gpu is $use_gpu"
    install_docker
    install_jmeter
    if [ "$use_gpu" == "1" ]; then
        install_nvidia_docker
    fi
    prepare_repo
    pull_docker_images
    run_test
    service docker stop && service docker start
    echo "docker restarted"
    docker rm -f hw
    echo "container removed"
}

main "$@"

