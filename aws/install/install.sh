#!/bin/bash

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# ==================================================================================================
# -- Install CARLA ---------------------------------------------------------------------------------
# ==================================================================================================
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y pulseaudio

# ==================================================================================================
# -- Install CARLA ---------------------------------------------------------------------------------
# ==================================================================================================
CARLA_VERSION=0.9.11

echo "Installing CARLA. This may take a while..."
curl -o ${HOME}/CARLA_${CARLA_VERSION}.tar.gz https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_${CARLA_VERSION}.tar.gz
mkdir -p ${HOME}/CARLA_${CARLA_VERSION}
tar -xzf ${HOME}/CARLA_${CARLA_VERSION}.tar.gz -C ${HOME}/CARLA_${CARLA_VERSION}

echo "Installing CARLA additional maps. This may take a while..."
curl -o ${HOME}/AdditionalMaps_${CARLA_VERSION}.tar.gz https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_${CARLA_VERSION}.tar.gz
mv ${HOME}/AdditionalMaps_${CARLA_VERSION}.tar.gz ${HOME}/CARLA_${CARLA_VERSION}/Import
cd ${HOME}/CARLA_${CARLA_VERSION} && bash ImportAssets.sh && cd ${HOME}

# ==================================================================================================
# -- Install RLlib ---------------------------------------------------------------------------------
# ==================================================================================================
echo "Preparing virtual environment..."
source activate pytorch_latest_p37
pip3 install pygame paramiko scp ray[rllib] tensorboard

# ==================================================================================================
# -- Env variables ---------------------------------------------------------------------------------
# ==================================================================================================
echo "Setting up env variables..."
echo "export CARLA_ROOT=~/CARLA_0.9.11" >> ~/custom_env.sh
echo "source activate pytorch_latest_p37" >> ~/custom_env.sh
echo 'export PYTHONPATH=""' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/dist/$(ls ${CARLA_ROOT}/PythonAPI/carla/dist | grep py3.)"' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla"' >> ~/custom_env.sh
echo 'alias python3=~/anaconda3/envs/pytorch_latest_p37/bin/python3.7' >> ~/custom_env.sh
echo 'alias python=~/anaconda3/envs/pytorch_latest_p37/bin/python3.7' >> ~/custom_env.sh

echo "source ~/custom_env.sh" >> ~/.bashrc
