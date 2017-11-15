from dms import deep_model_server

# Read arguments
config_file = 'dms_app.conf'
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

server = deep_model_server.DMS(args=args)
application = server.create_app()