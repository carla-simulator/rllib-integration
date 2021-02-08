#!/bin/bash

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# ==================================================================================================
# -- Install CARLA ---------------------------------------------------------------------------------
# ==================================================================================================
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz -P ${HOME}
mkdir ${HOME}/CARLA_0.9.11 && tar -xzf ${HOME}/CARLA_0.9.11.tar.gz -C ${HOME}/CARLA_0.9.11

# ==================================================================================================
# -- Install RLlib ---------------------------------------------------------------------------------
# ==================================================================================================
source activate pytorch_latest_p37
pip3 install pygame paramiko scp ray[rllib]

# ==================================================================================================
# -- Env variables ---------------------------------------------------------------------------------
# ==================================================================================================
echo "export CARLA_ROOT=~/CARLA_0.9.11" >> ~/custom_env.sh
echo "source activate pytorch_latest_p37" >> ~/custom_env.sh
echo 'export PYTHONPATH=""' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla/dist/$(ls ${CARLA_ROOT}/PythonAPI/carla/dist | grep py3.)"' >> ~/custom_env.sh
echo 'export PYTHONPATH=$PYTHONPATH:"${CARLA_ROOT}/PythonAPI/carla"' >> ~/custom_env.sh

echo "source ~/custom_env.sh" >> ~/.bashrc
