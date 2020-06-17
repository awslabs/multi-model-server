import psutil
import pandas as pd

SEPARATOR = '~'
# INPUTS = ['min~cpu_percent', 'max~cpu_times~system', 'sum~memory_info~rss', 'mean~io_counters~read_count', 'sum~num_fds', 'sum~num_threads']
INPUTS = ['max~cpu_times~system', 'sum~num_threads']

def input_parser(input):
    tokens = input.split(SEPARATOR)
    operator = tokens[0]
    metric_type = tokens[1]
    try:
        metric_attr = tokens[2]
    except IndexError:
        metric_attr = None

    metric_name = metric_type if metric_attr is None else metric_type + '_' + metric_attr
    return operator, metric_type, metric_attr, metric_name


info = {
    'name': [],
    'ptype': [],
    'pid': [],
    'parent_pid': []
}

def parse_process_tree(process, level=0):
    stat_dict = process.as_dict()

    info['name'].append(stat_dict['name'])
    info['ptype'].append(level)
    info['pid'].append(stat_dict['pid'])
    info['parent_pid'].append(stat_dict['ppid'])

    for input in INPUTS:
        operator, metric_type, metric_attr, metric_name = input_parser(input)
        info[metric_name] = info[metric_name] if info.get(metric_name) else []

        if metric_attr is None:
            info[metric_name].append(stat_dict.get(metric_type, 0))
        else:
            info[metric_name].append(getattr(stat_dict.get(metric_type, {}), metric_attr, 0))

    for child in process.children():
        parse_process_tree(child, level+1)


server_process = psutil.Process(8182)
parse_process_tree(server_process)

df = pd.DataFrame(info)
print("---------------------------Data Frame Object---------------------------")
print(df)
print()
result = {}
for input in INPUTS:
    operator, metric_type, metric_attr, metric_name = input_parser(input)
    operator_fn = getattr(df, operator)
    res = operator_fn()
    print("-------------- "+operator+" : "+metric_name+" --------------")
    print(res)
    print()
    result[input] = res.get(metric_name)
print("---------------------------------------------------")
print(result)