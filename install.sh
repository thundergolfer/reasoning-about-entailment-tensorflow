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

# Set Python version. Checks for whether we're in a Travis build
PYTHON_VERSION="${TRAVIS_PYTHON_VERSION:-3.5}" # or sub in 2.7 for 3.5

# install Tensorflow
if [[ `uname` == "Darwin" ]]; then
  echo "installing for Apple Mac architecture"
  if [[ $PYTHON_VERSION == "3.5" ]]; then
    CURRTF=tensorflow-0.10.0-py3-none-any.whl
    curl https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl > ./tensorflow-0.10.0-py3-none-any.whl

  elif [[ $PYTHON_VERSION == "2.7" ]]; then
    CURRTF=tensorflow-0.10.0-py2-none-any.whl
    curl https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl > ./tensorflow-0.10.0-py2-none-any.whl
  else
    echo "Python version must be set to '3.5' or '2.7'"
    exit 1
  fi
else
  echo "installing for Linux Architecture"
  if [[ $PYTHON_VERSION == "3.5" ]]; then
    CURRTF=tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
    curl https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl > ./tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
  elif [[ $PYTHON_VERSION == "2.7" ]]; then
    CURRTF=tensorflow-0.10.0-cp27-none-linux_x86_64.whl
    curl https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl > ./tensorflow-0.10.0-cp27-none-linux_x86_64.whl
  else
    echo "Python version must be set to '3.5' or '2.7'"
    exit 1
fi


source miniconda_info.sh

conda_command="${MINICONDA_DIRECTORY}/bin/conda"
$conda_command info -a

# Create or replace python virtual environment
rm -rf $VIRTUAL_ENV_PATH
$conda_command create -y python=$PYTHON_VERSION -p $VIRTUAL_ENV_PATH --file "conda_env_spec.txt"


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
