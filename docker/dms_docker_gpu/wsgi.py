import os
from dms import mxnet_model_server

# Read arguments
config_file = 'dms_app.config'
args = []
gunicorn_arg_start = '# Gunicorn arguments'
with open(config_file) as f:
    content = f.readlines()
    content = [line.rstrip() for line in content]
for i, line in enumerate(content):
    if line == gunicorn_arg_start:
        break
    if line.startswith('--') and content[i + 1] != 'optional':
        args.append(line)
        if line == '--models':
            args += content[i + 1].split(' ')
        else:
            args.append(content[i + 1])

args.append('--gpu')
args.append(int(os.environ['gpu_id']))
os.environ['gpu_id'] = str(int(os.environ['gpu_id']) + 1)

server = mxnet_model_server.DMS(args=args)
application = server.create_app()