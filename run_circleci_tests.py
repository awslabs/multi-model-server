#!/usr/bin/env python

# Make sure you have following dependencies installed on your local machine
# 1. PyYAML (pip install PyYaml)
# 2. CircleCI cli from - https://circleci.com/docs/2.0/local-cli/#installation
# 3. docker

import subprocess
import sys

import yaml
import copy
import argparse
from collections import OrderedDict
from functools import reduce

parser = argparse.ArgumentParser(description='Execute circleci jobs in a container on local machine')
parser.add_argument('workflow', type=str, help='Workflow name from config.yml')
parser.add_argument('-j', '--job', type=str, help='Job name from config.yml')
parser.add_argument('-e', '--executor', type=str, help='Executor name from config.yml')
args = parser.parse_args()

workflow = args.workflow
job = args.job
executor = args.executor

cci_config_file = '.circleci/config.yml'
processed_file = '.circleci/processed.yml'
xformed_file = '.circleci/xformed.yml'
cci_config = {}
processed_config = {}
xformed_config = {}
xformed_job_name = 'mms_xformed_job'
blacklisted_steps = ['persist_to_workspace', 'attach_workspace', 'store_artifacts']

# Read CircleCI's config
with open(cci_config_file) as fstream:
    try:
        cci_config = yaml.safe_load(fstream)
    except yaml.YAMLError as err:
        print(err)

# Create processed YAML using circleci cli's 'config process' commands
porcess_config_cmd = 'circleci config process {} > {}'.format(cci_config_file, processed_file)
print("Executing command : ",porcess_config_cmd)
subprocess.check_call(porcess_config_cmd, shell=True)

# Read the processed config
with open(processed_file) as fstream:
    try:
        processed_config = yaml.safe_load(fstream)
    except yaml.YAMLError as err:
        print(err)

# All executors available in the config file
available_executors = list(cci_config['executors'])

# All jobs available under the specified workflow
jobs_in_workflow = processed_config['workflows'][workflow]['jobs']


# Recursively iterate over jobs in the workflow to generate an ordered list of parent jobs
def get_processed_job_sequence(processed_job_name):
    result = []
    for job in jobs_in_workflow :
        if isinstance(job, str) and processed_job_name == job:
            # We have reached a root job (outter most parent), no further traversal
            break
        elif isinstance(job, dict) and processed_job_name == list(job)[0]:
            # Find all parent jpbs, recurse to find their respective ancestors
            parent_jobs = job[processed_job_name].get('requires',[])
            for pjob in parent_jobs:
                result += get_processed_job_sequence(pjob)
            break
    return result + [processed_job_name]


def get_jobs_to_exec(job):
    result = {}
    executors = [executor] if executor else available_executors

    for exec in executors:
        if job is None:
            # List of all job names(as string)
            result[exec] = map(lambda j: j if isinstance(j, str) else list(j)[0], jobs_in_workflow)
            # Filter processed job names as per the executor
            # "job_name-executor_name" is a convention set in config.yml
            result[exec] = filter(lambda j: exec in j, result[exec])
        else:
            # The list might contain duplicate parent jobs due to multiple fan-ins like config; remove the duplicates
            # "job_name-executor_name" is a convention set in config.yml
            result[exec] = OrderedDict.fromkeys(get_processed_job_sequence(job+'-'+exec))
        result[exec] = list(result[exec])

    return result

# jobs_to_exec is a dict, with executor(s) as the key and list of jobs to be executed as its value
jobs_to_exec = get_jobs_to_exec(job)


# Merge all the steps from list of jobs to execute
def get_jobs_steps(result, job_name):
    job_steps = processed_config['jobs'][job_name]['steps']
    filtered_job_steps = list(filter(lambda step: list(step)[0] not in blacklisted_steps, job_steps))
    return result + filtered_job_steps

result = {}

for exec, jobs in jobs_to_exec.items():
    merged_steps = reduce(get_jobs_steps, jobs, [])

    # Create a new job, using the first job as a reference
    # This ensures configs like executor, environment, etc are maintained as it is from the first job
    first_job = jobs[0]
    xformed_job = copy.deepcopy(processed_config['jobs'][first_job])

    # Add the merged steps to this newly introduced job
    xformed_job['steps'] = merged_steps

    # Create a duplicate config(transformed) with the newly introduced job (as the only job in config)
    xformed_config = copy.deepcopy(processed_config)
    xformed_config['jobs'] = {}
    xformed_config['jobs'][xformed_job_name] = xformed_job

    # Create a transformed yaml
    with open(xformed_file, 'w+') as fstream:
        yaml.dump(xformed_config, fstream)

    try:
        # Locally execute the newly created job
        # This job has all the steps(ordered and merged) from the mentioned job along with it's ancestors(if any)
        local_execute_command = 'circleci local execute -c {} --job {}'.format(xformed_file, xformed_job_name)
        print('Executing command : ', local_execute_command)
        result[exec] = subprocess.check_call(local_execute_command, shell=True)
    except subprocess.CalledProcessError as err:
        result[exec] = err.returncode

# Clean up, remove the processed and transformed yml files
cleanup_cmd = 'rm {} {}'.format(processed_file, xformed_file)
print('Executing command : ', cleanup_cmd)
subprocess.check_call(cleanup_cmd, shell=True)

# Print job execution details
for exec, retcode in result.items():
    colorcode, status = ('\033[0;37;42m', 'successful') if retcode == 0 else ('\033[0;37;41m', 'failed')
    print("{} Job execution {} using {} executor \x1b[0m".format(colorcode, status, exec))

# Exit as per overall status
sys_exit_code = 0 if all(retcode == 0 for exec, retcode in result.items()) else 1
sys.exit(sys_exit_code)
