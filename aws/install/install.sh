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
sudo apt-get install pulseaudio

# ==================================================================================================
# -- Install CARLA ---------------------------------------------------------------------------------
# ==================================================================================================
echo "Installing CARLA. This may take a while..."
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz -P ${HOME}
mkdir ${HOME}/CARLA_0.9.11 && tar -xzf ${HOME}/CARLA_0.9.11.tar.gz -C ${HOME}/CARLA_0.9.11

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
echo "alias conda_py=/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python3.7" >> ~/custom_env.sh
echo "alias conda_tensorboard=/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/tensorboard" >> ~/custom_env.sh
echo "export CARLA_ROOT=~/CARLA_0.9.11" >> ~/custom_env.sh
echo "source activate pytorch_latest_p37" >> ~/custom_env.sh
echo 'export PYTHONPATH=""' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/dist/$(ls ${CARLA_ROOT}/PythonAPI/carla/dist | grep py3.)"' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla"' >> ~/custom_env.sh

echo "source ~/custom_env.sh" >> ~/.bashrc
