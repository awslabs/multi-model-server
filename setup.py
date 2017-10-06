from setuptools import setup, find_packages
setup(
    py_modules=['mxnet_model_server', 'export_model'],
    name='mxnet-model-server',
    version='0.3',
    description='MXNet Model Server',
    url='https://github.com/deep-learning-tools/mxnet-model-server',
    keywords='MXNet Model Serving Deep Learning Inference',
    packages=['tools', 'src', 'src.model_service', 'src.request_handler', 'src.utils'],
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors'],
    entry_points={
        'console_scripts':['mxnet-model-server=mxnet_model_server:start_serving', 'mxnet-model-export=export_model:export']
    },
    include_package_data=True
)