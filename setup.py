from setuptools import setup, find_packages

pkgs = find_packages()
pkgs.append('tools')

setup(
    name='deep-model-server',
    version='0.1',
    description='Deep Model Server is a tool for deploying neural net models for inference',
    url='https://github.com/deep-learning-tools/deep-model-server',
    keywords='MXNet Model Serving Deep Learning Inference',
    packages=pkgs,
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors'],
    entry_points={
        'console_scripts':['deep-model-server=mms.mxnet_model_server:start_serving', 'deep-model-export=mms.export_model:export']
    },
    include_package_data=True,
    license='Apache License Version 2.0'
)