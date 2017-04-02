#!/bin/bash -e
# Create or replace the python virtual environment and installs all the packages required for the app.
#
# Usage:
#
# ./install.sh
#
# The requirements file listing the conda and PIP dependencies can be overriden:
#
# * REQUIREMENTS_PIP_FILE - List of dependencies/libs to be installed with pip
# * REQUIREMENTS_CONDA_FILE - List of dependencies/libs to be installed with conda
#

echo "installing for Apple Mac architecture"

# install Tensorflow
if [[ `uname` == "Darwin" ]]; then
  CURRTF=tensorflow-0.10.0-py2-none-any.whl
  curl https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl > ./tensorflow-0.10.0-py2-none-any.whl
else
  CURRTF=tensorflow-0.10.0-cp27-none-linux_x86_64.whl
  curl https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl > ./tensorflow-0.10.0-cp27-none-linux_x86_64.whl
fi


source miniconda_info.sh

conda_command="${MINICONDA_DIRECTORY}/bin/conda"
$conda_command info -a

# Create or replace python virtual environment
rm -rf $VIRTUAL_ENV_PATH
$conda_command create -y python -p $VIRTUAL_ENV_PATH --file "conda_env_spec.txt"


# Install the app and its dependencies
source run_in_environment.sh

echo Installing pip requirement file
cat requirements.txt | \
while read PKG; do
    pip install --upgrade $PKG --ignore-installed setuptools || true
done


pip install --no-deps ./$CURRTF
SCRIPT_PATH=`readlink -f "$0"`
SCRIPT_DIR=`dirname "$SCRIPT_PATH"`

rm $CURRTF

# install jupyter and ipython
conda install -y --no-deps jupyter ipython
