from mms import mxnet_model_server
import sys
import os

mms_arg_header = 'MMS Argument'
config_file = os.environ['MXNET_MODEL_SERVER_CONFIG']
args = []
found_mms_args = 0

print("Using %s for config file" % config_file)

try:
    f = open(config_file)
except IOError as e:
    sys.exit("File %s could not be located. Error %s" % config_file, e)
else:
    print("Successfully read config file %s.." % config_file)
    with f:
        try:
            content = f.readlines()
            content = [line.rstrip() for line in content]

            for i, line in enumerate(content):
                line = line.lstrip()
                if line.startswith('['):
                    found_mms_args = 1 if mms_arg_header.lower() in line.lower() else 0
                if line.startswith('#') or line.startswith('$'):
                    continue
                if found_mms_args is 1:
                    if line.startswith('--') and content[i + 1] != 'optional':
                        args.append(line)
                        if line == '--models':
                            args += content[i + 1].split(' ')
                        else:
                            args.append(content[i + 1])
        except Exception as e:
            sys.exit("Error reading the file %s" % e)

args.append('--gpu')
args.append(int(os.environ['gpu_id']))
os.environ['gpu_id'] = str(int(os.environ['gpu_id']) + 1)

server = mxnet_model_server.MMS(args=args)
application = server.create_app()
