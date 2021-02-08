#!/bin/bash

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

#USAGE_STRING="Usage: $0 [-h|--help] [--config={Debug,Development,Shipping}] [--no-zip] [--clean-intermediate] [--packages=Name1,Name2,...] [--target-archive=]"

SCRIPT="dqn_train.py"
CONFIGURATION_FILE="dqn_config.yaml"
DIRECTORY="~/ray_results/carla_rllib"
NAME="dqn_example"

while getopts i:s:c:d:n:ro flag
do
    case "${flag}" in
        i) INSTANCE=${OPTARG};;
        s) SCRIPT=${OPTARG};;
        c) CONFIGURATION_FILE=${OPTARG};;
        d) DIRECTORY=${OPTARG};;
        n) NAME=${OPTARG};;
        r) RESTORE="true";;
        o) OVERWRITE="true";;
        *) error "Unexpected option ${flag}" ;;
    esac
done

EC2_USER="ubuntu"
PYTHON_BIN="~/anaconda3/envs/torch/bin/python3.8"

python aws_helper.py start --instance-id $INSTANCE
python aws_helper.py put --instance-id $INSTANCE --source ${PWD} --exclude .git keys __pycache__ map_cache

PUBLIC_IP=$(echo $(python aws_helper.py info --instance-id ${INSTANCE} --field public_ip 2>&1 > /dev/null) | awk '{print $NF}')
PEM_FILE=$(echo $(python aws_helper.py info --instance-id ${INSTANCE} --field pem_file 2>&1 > /dev/null) | awk '{print $NF}')

COMMAND="$PYTHON_BIN ${SCRIPT} ${CONFIGURATION_FILE} --directory ${DIRECTORY} --name ${NAME}"
if [ "$RESTORE" == "true" ]; then
    COMMAND="$COMMAND --restore";
fi
if [ "$OVERWRITE" == "true" ]; then
    COMMAND="$COMMAND --overwrite";
fi

ssh -tt -i $PEM_FILE $EC2_USER@$PUBLIC_IP "source ~/custom_env.sh; cd rllib-integration; $COMMAND"

#python aws_helper.py stop --instance-id $INSTANCE

