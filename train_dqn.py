#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import yaml

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
import torch

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.helper.carla_helper import kill_server

from dqn_example.experiment import DQNExperiment


class CustomDQNTrainer(DQNTrainer):
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)

        model = self.get_policy().model
        torch.save({
            model.state_dict(),
        }, os.path.join(checkpoint_dir, "checkpoint.pth"))

        return checkpoint_path


def find_latest_checkpoint(args):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = args.directory + "/" + args.name
    max_f = ""
    max_g = ""
    max_checkpoint = 0
    for f in os.listdir(start):
        if args.algorithm in f:
            temp = start + "/" + f
            for g in os.listdir(temp):
                if "checkpoint_" in g:
                    episode = int(''.join([n for n in g if n.isdigit()]))
                    if episode > max_checkpoint:
                        max_checkpoint = episode
                        max_f = f
                        max_g = g
    if max_checkpoint == 0:
        print(
            "Could not find any checkpoint, make sure that you have selected the correct folder path"
        )
        raise IndexError
    start += ("/" + max_f + "/" + max_g + "/" + max_g.replace("_", "-"))
    return start


def run(args):
    try:
        checkpoint = False
        if args.restore:
            checkpoint = find_latest_checkpoint(args)

        with open(args.configuration_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config["env"] = CarlaEnv
            config["env_config"]["experiment"]["type"] = DQNExperiment

        while True:
            kill_server()
            ray.init()
            tune.run(
                CustomDQNTrainer,
                name=args.name,
                local_dir=args.directory,
                stop={"perf/ram_util_percent": 85.0},
                checkpoint_freq=1,
                checkpoint_at_end=True,
                restore=checkpoint,
                config=config
            )
            ray.shutdown()
            checkpoint = find_latest_checkpoint(args)

    finally:
        kill_server()
        ray.shutdown()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory",
                           metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/ray_rllib")
    argparser.add_argument("-n", "--name",
                           metavar="N",
                           default="dqn",
                           help="Name of the experiment (default: dqn)")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag to overwrite a specific directory (warning: all content of the folder will be lost.)")

    args = argparser.parse_args()

    directory = args.directory + "/" + args.name
    if not args.restore:
        if os.path.exists(directory):
            if args.overwrite and os.path.isdir(directory):
                shutil.rmtree(directory)
            elif len(os.listdir(directory)) != 0:
                print("The directory " + directory + " is not empty. To start a new training instance, make sure this folder is either empty or non-existing.")
                return
    else:
        if not(os.path.exists(directory)) or len(os.listdir(directory)) == 0:
            print("You can't restore from an empty or non-existing directory. To restore a training instance, make sure there is at least one checkpoint.")
    run(args)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
