#!/usr/bin/env python
import platform
from subprocess import check_call
from pip import main as pip_install
import re


CUSTOM_TF_PATHS = {
    'centos_gpu_east': 's3://zendesk-datascientists-intermediate/tensorflow/wheel/tensorflow-1.0.0-cp27-cp27mu-linux_x86_64.whl',
    'centos_gpu_west': 's3://zendesk-datascientists-intermediate/tensorflow/wheel/west/tensorflow-1.0.0-cp27-cp27mu-linux_x86_64.whl',
    'ubuntu_cpu_hedge': 'https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl'
}


def check_tf_type():
    """
    Determines tensorflow type to install using a regex on the hostname.
    """
    system = None
    hostname = platform.node()

    # Identify hedge nodes by looking for the word hedge
    if re.search('hedge', hostname) is not None:
        system = 'ubuntu_cpu_hedge'

    # Identify GPU-West nodes, they have g2-2xlarge1.numbats.us-west....
    elif re.search('numbats', hostname) is not None:
        system = 'centos_gpu_west'

    # Identify GPU-East nodes, they live in pod14
    elif re.search('pod14', hostname) is not None:
        system = 'centos_gpu_east'

    # All other OS types are generally pip compatible
    else:
        system = 'pip_compatible'
    return system


def install_requirements_pip():
    """
    Installs packages using pip from requirements_pip.txt; removes 
    tensorflow from the requirements to pass off to the special 
    install_tensorflow function.
    """
    with open('requirements.txt', 'r') as f:
        for requirement in f:
            if re.search('tensorflow==', requirement) is None:
                pip_install(['install', requirement])
            else:
                tensorflow_req = requirement
    return tensorflow_req


def install_tensorflow(tensorflow):
    """
    Checks the system type, then installs the correct tensorflow type
    for a given hostname.
    """
    tf_type = check_tf_type()
    if tf_type == 'pip_compatible':
        pip_install(['install', tensorflow])
    elif tf_type == 'ubuntu_cpu_hedge':
        pip_install(['install', CUSTOM_TF_PATHS['ubuntu_cpu_hedge']])
    else:
        check_call(['aws', 's3', 'cp', CUSTOM_TF_PATHS[tf_type], '.'])
        pip_install(['install', CUSTOM_TF_PATHS[tf_type].split('/')[-1],
                     '--ignore-installed', 'setuptools', 'numpy',
                     '--upgrade'])


def main():
    tf_requirement = install_requirements_pip()
    install_tensorflow(tf_requirement)


main()
