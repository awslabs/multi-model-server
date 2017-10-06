from setuptools import setup, find_packages
pkgs = find_packages()
pkgs.append('tools')
setup(
    name='mxnet-model-server',
    version='0.3.4',
    description='MXNet Model Server',
    url='https://github.com/deep-learning-tools/mxnet-model-server',
    keywords='MXNet Model Serving Deep Learning Inference',
    packages=pkgs,
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors'],
    entry_points={
        'console_scripts':['mxnet-model-server=mms.mxnet_model_server:start_serving', 'mxnet-model-export=mms.export_model:export']
    },
    include_package_data=True
)