#!/bin/bash

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

DIRECTORY="~/ray_results/carla_rllib"
NAME="dqn_example"

while getopts i:d:n: flag
do
    case "${flag}" in
        i) INSTANCE=${OPTARG};;
        d) DIRECTORY=${OPTARG};;
        n) NAME=${OPTARG};;
        *) error "Unexpected option ${flag}" ;;
    esac
done

EC2_USER="ubuntu"
#TENSORBOARD_BIN="tensorboard"
TENSORBOARD_BIN="/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/tensorboard"

python aws_helper.py start --instance-id $INSTANCE

PUBLIC_IP=$(echo $(python aws_helper.py info --instance-id ${INSTANCE} --field public_ip 2>&1 > /dev/null) | awk '{print $NF}')
PEM_FILE=$(echo $(python aws_helper.py info --instance-id ${INSTANCE} --field pem_file 2>&1 > /dev/null) | awk '{print $NF}')

ssh -tt -i $PEM_FILE $EC2_USER@$PUBLIC_IP "source ~/custom_env.sh; $TENSORBOARD_BIN --logdir "$DIRECTORY/$NAME" --host 0.0.0.0"
