from setuptools import setup, find_packages
setup(
    py_modules=['mxnet_model_server', 'export_model'],
    name='mxnet-model-server',
    version='0.2.1.2',
    description='MXNet Model Serving',
    url='https://github.com/yuruofeifei/mms',
    keywords='MXNet Serving',
    packages=['tools', 'mms', 'mms.model_service', 'mms.request_handler', 'mms.utils'],
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors'],
    entry_points={
        'console_scripts':['mxnet-model-server=mxnet_model_server:mms', 'mms-export=export_model:export']
    },
    include_package_data=True
)