from setuptools import setup, find_packages
setup(
    py_modules=['mxnet_model_server', 'export_model', 'arg_parser', 'client_sdk_generator', 'log', 'service_manager', 'serving_frontend', 'storage'],
    name='mxnet-model-server',
    version='0.3.2',
    description='MXNet Model Server',
    url='https://github.com/deep-learning-tools/mxnet-model-server',
    keywords='MXNet Model Serving Deep Learning Inference',
    packages=['tools', 'model_service', 'request_handler', 'utils'],
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors'],
    entry_points={
        'console_scripts':['mxnet-model-server=mxnet_model_server:start_serving', 'mxnet-model-export=export_model:export']
    },
    include_package_data=True
)