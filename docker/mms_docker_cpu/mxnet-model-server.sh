#!/bin/bash

# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

MMS_DIR='/mxnet-model-server'
CONFIG_FILE=$MMS_DIR'/mms_app.conf'
GUNICORN_ARGS='gunicorn'
MXNET_ARGS='mxnet'
NGINX_ARGS='nginx'
MMS_ARG='mms'
NGINX_CONFIG_FILE='/etc/nginx/conf.d/virtual.conf'

# Start MxNet model server. This function expects one argument which is the application config file.
start_mms() 
{
    gunicorn_arguments=''
    # TODO: Check if this process is already running . If yes ignore start and throw warning.

    # This function expects one argument.
    if [[ "$#" != 1 ]]
    then
        echo "Wrong arguments given ($#) \"$@\""
        exit 100
    fi

    MMS_CONFIG_FILE=$1

    # if the MMS CONFIG FILE is an empty string or if the file doesn't exist, throw error and exit
    if [[ ( -z "${MMS_CONFIG_FILE##*( )}" ) || ( ! -f "$MMS_CONFIG_FILE" ) ]]
    then
        echo "No configuration file given... $MMS_CONFIG_FILE"
        exit 100
    fi

    while read -r line || [ -n "$line" ]
    do
        line="${line##*( )}"
	    shopt -s nocasematch
	    # Comments and empty lines should be ignored . # and $ are comment starters.
	    if [[ ( -z "$line" ) || ( "$line" =~ ^\# ) || ( "$line" =~ ^\$ ) ]]
     	then
    	    continue
	    fi
	
	    if [[ "$line" =~ ^\[[A-Za-z\ ] ]] # new header
	    then
    	    VAR=$line
	        if [[ ( "$VAR" =~ "$NGINX_ARGS" ) ]] ; then
	            rm -f $NGINX_CONFIG_FILE
	            touch $NGINX_CONFIG_FILE
	        fi
	        continue
	    fi
	
	    if [[ "$VAR" =~ "$GUNICORN_ARGS" ]]
	    then
    	    gunicorn_arguments="${gunicorn_arguments} $line "
    	elif [[ "$VAR" =~ "$MXNET_ARGS" ]]
    	then
    	    export $line
    	elif [[ "$VAR" =~ "$NGINX_ARGS" ]]
    	then
    	    # Check if nginx config file exists. Else create one.
    	    if [[ -f $NGINX_CONFIG_FILE ]]
    	    then
    	        if [[ ( "$line" =~ "server_name" ) && ( ! -z ${MMS_HOST// } ) ]]
    	        then
    	            host_name=$MMS_HOST
    	            line="server_name $host_name;"
    	        fi
    	    fi
    		echo "$line" >> $NGINX_CONFIG_FILE
        elif [[ "$VAR" =~ "$MMS_ARG" ]]
        then
            continue
    	else
    	    echo "Invalid config header seen $VAR"
    	    exit 100
    	fi

    done < "$MMS_CONFIG_FILE"
    shopt -u nocasematch

    app_script='wsgi'
    service nginx restart
    gunicorn $gunicorn_arguments --chdir /mxnet-model-server --env MXNET_MODEL_SERVER_CONFIG=$MMS_CONFIG_FILE $app_script
}

stop_mms() {
    echo "Stopping MMS"
    kill -9 $(ps -e | grep "gunicorn\|nginx" | awk '{print $1}') 2> /dev/null
}

usage_help() { 
    echo "Usage:"
    echo ""
    echo "$0 [start | stop | restart | help] [--mms-config <MMS config file>]"
    echo ""
    echo "start        : Start a new instance of MxNet model server."
    echo "stop         : Stop the current running instance of MxNet model server"
    echo "restart      : Restarts all the MMS worker instances."
    echo "help         : Usage help for $0"
    echo "--mms-config : Location pointing to the MxNet model server configuration file."
    echo "To start the MxNet model server, run"
    echo "$0 start --mms-config <path-to-config-file>"
    echo ""
    echo "To stop the running instnce of MxNet model server, run"
    echo "$0 stop"
    echo ""
    echo "To restart the running instance of MxNet model server, run"
    echo "$0 restart --mms-config <path-to-config-file>"
}

main() {
    option='none'
    mms_config_file='default'

    shopt -s nocasematch
    while true ; do
        case "$1" in
    	"start" | "stop" | "help" | "restart" ) option=$1 ; shift ;;
	    "--mms-config" )  mms_config_file=$2 ; shift 2 ;;
	    * ) break ;;
        esac
    done

    case "$option" in
        "start" ) start_mms $mms_config_file ;;
        "stop" ) stop_mms ;;
        "restart" ) stop_mms && sleep 5 && start_mms $mms_config_file ;;
        "help" ) usage_help ;;
        "none" | "*")
             echo "Invalid options given"
             usage_help
             exit 100;;
    esac

    shopt -u nocasematch
}

main "$@"
exit $?