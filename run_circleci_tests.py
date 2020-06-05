# Make sure you have following dependencies installed on your local machine
# 1. PyYAML (pip install PyYaml)
# 2. CircleCI cli from - https://circleci.com/docs/2.0/local-cli/#installation
# 3. docker

import subprocess
import yaml
import copy
from collections import OrderedDict
from functools import reduce

workflow = 'build_and_test'
job = 'python_tests'

circleci_config_file = '.circleci/config.yml'
processed_file = 'processed.yml'
processed_config = {}
xformed_file = 'xformed.yml'
xformed_config = {}
xformed_job_name = 'mms_xformed_job'
blacklisted_steps = ['persist_to_workspace', 'attach_workspace', 'store_artifacts']

# Create processed YAML using circleci cli's 'config process' commands
subprocess.check_call('circleci config process {} > {}'.format(circleci_config_file, processed_file), shell=True)

# Read the processed config
with open(processed_file) as fstream:
    try:
        processed_config = yaml.safe_load(fstream)
    except yaml.YAMLError as err:
        print(err)

jobs_in_workflow = processed_config['workflows'][workflow]['jobs']

# Recursively iterate over jobs in the workflow to generate an ordered list of parent jobs
def getJobsToExec(job_name):
    result = []
    for job in jobs_in_workflow :
        if(isinstance(job, str) and job_name == job):
            # We have reached a root job (outter most parent), no further traversal
            break
        elif(isinstance(job, dict) and job_name == list(job)[0]):
            # Find all parent jpbs, recurse to find their respective ancestors
            parent_jobs = job[job_name].get('requires',[])
            for pjob in parent_jobs :
                result += getJobsToExec(pjob)
            break
    return (result + [job_name])

# The list might contain duplicate parent jobs due to muliple fan-ins like config; remove the duplicates
jobs_to_exec = list(OrderedDict.fromkeys(getJobsToExec(job)))

# Merge all the steps from jobs to be executed
def getJobSteps(result, job_name):
    job_steps = processed_config['jobs'][job_name]['steps']
    filtered_job_steps = list(filter(lambda step: list(step)[0] not in blacklisted_steps, job_steps))
    return result + filtered_job_steps

merged_steps = reduce(getJobSteps, jobs_to_exec, [])

# Create a new job, using the first job as a reference
# This ensures configs like executor, environment, etc are maintained as it is from the first job
first_job = jobs_to_exec[0]
xformed_job = copy.deepcopy(processed_config['jobs'][first_job])

# Add the merged steps to this newly introduced job
xformed_job['steps'] = merged_steps

# Create a duplicate config(transformed) with the newly introduced job (as the only job in config)
xformed_config = copy.deepcopy(processed_config)
xformed_config['jobs'] = {}
xformed_config['jobs'][xformed_job_name] = xformed_job

# Create a transformed yaml
with open(xformed_file, 'w+') as fstream:
    yaml.dump(xformed_config,fstream)

# Locally execute the newly created job
# This job has all the steps(ordered and merged) from the mentioned job along with it's ancestors
subprocess.check_call('circleci local execute -c {} --job {}'.format(xformed_file, xformed_job_name), shell=True)

# Clean up, remove the processed and transformed yml files
subprocess.check_call('rm {} {}'.format(processed_file, xformed_file), shell=True)
