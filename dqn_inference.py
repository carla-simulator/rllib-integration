
#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import os
import argparse
import yaml

import torch

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from dqn_example.dqn_experiment import DQNExperiment
from dqn_example.dqn_inference_model import CustomDQNModel

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperiment

def get_gpu_or_cpu_number(device):
    """Returns the GPU number on which the tensors will be run. Returns -1 if the CPU is used"""

    if 'cuda' in device:
        if not torch.cuda.is_available():
            raise RuntimeError("Torch cuda check failed, your drivers might not be correctly installed")
        gpu = device.split(":")
        if len(gpu) > 1:
            gpu_n = int(gpu[1])
        else:
            gpu_n = 0
    else:
        gpu_n = -1  # i.e, tensor are CPU based

    return gpu_n

def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS

    return config

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file of the run (*.yaml)")
    argparser.add_argument("checkpoint",
                           help='Checkpoint file with the model information (*.pt or *.pth)')
    argparser.add_argument(
        '-d', '--device',
        metavar='D',
        default= 'cuda:0',
        help='Device on with the tensors will be run. Defaults to (cuda:0)')

    args = argparser.parse_args()
    args.config = parse_config(args)
    args.gpu_n = get_gpu_or_cpu_number(args.device) # Are we using GPU or CPU?

    try:
        # Initialize the model and load the state dictionary
        model = CustomDQNModel(gpu_n=args.gpu_n)
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        if args.gpu_n >= 0:
            model.cuda()

        # Initalize the CARLA environment
        env = CarlaEnv(args.config["env_config"])
        obs = env.reset()

        while True:
            action = model.forward(obs)
            obs, _, _, _ = env.step(action)

    except KeyboardInterrupt:
        pass

    finally:
        kill_all_servers()

if __name__ == '__main__':

    main()