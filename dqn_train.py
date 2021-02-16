#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import print_function

import sys
import argparse
import os
import shutil
import yaml

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer

import torch
import numpy as np

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from dqn_example.dqn_experiment import DQNExperiment

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperiment


class DQNCallbacks(DefaultCallbacks):

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["heading_deviation"] = []

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        heading_deviation = worker.env.experiment.last_heading_deviation
        if heading_deviation > 0:
            episode.user_data["heading_deviation"].append(heading_deviation)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        heading_deviation = episode.user_data["heading_deviation"]
        if len(heading_deviation) > 0:
            heading_deviation = np.mean(episode.user_data["heading_deviation"])
        else:
            heading_deviation = 0
        episode.custom_metrics["heading_deviation"] = heading_deviation


class CustomDQNTrainer(DQNTrainer):
    """
    Modified version of DQNTrainer with the added functionality of saving the torch model for later inference
    """
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = super().save_checkpoint(checkpoint_dir)

        model = self.get_policy().model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_state_dict.pth"))

        return checkpoint_path


def find_latest_checkpoint(args):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """

    start = args.training_directory
    max_checkpoint_int = -1
    checkpoint_path = ""

    # 1st layer: Check for the different run folders
    for f in os.listdir(start):
        if os.path.isdir(start + "/" + f):
            temp = start + "/" + f

            # 2nd layer: Check all the checkpoint folders
            for c in os.listdir(temp):
                if "checkpoint_" in c:

                    # 3rd layer: Get the most recent checkpoint
                    checkpoint_int = int(''.join([n for n in c if n.isdigit()]))
                    if checkpoint_int > max_checkpoint_int:
                        max_checkpoint_int = checkpoint_int
                        checkpoint_path = temp + "/" + c + "/" + c.replace("_", "-")

    if not checkpoint_path:
        raise FileNotFoundError("Could not find any checkpoint, make sure that you have selected the correct folder path")

    return checkpoint_path

def manage_training_directory(args):
    """
    Depending of the arguments, makes sure that the directory is correctly setup
    """
    training_directory = args.directory + "/" + args.name

    if not args.restore:
        if os.path.exists(training_directory):
            if args.overwrite and os.path.isdir(training_directory):
                print("Removing all contents inside '" + training_directory + "'")
                shutil.rmtree(training_directory)
            elif len(os.listdir(training_directory)) != 0:
                print("The directory where you are trying to train (" + training_directory + ") is not empty. "
                      "To start a new training instance, make sure this folder is either empty, non-existing "
                      "or use the '--overwrite' argument to remove all the contents inside")
                sys.exit(-1)
    else:
        if not(os.path.exists(training_directory)) or len(os.listdir(training_directory)) == 0:
            print("You can't restore from an empty or non-existing directory. "
                  "To restore a training instance, make sure there is at least one checkpoint.")
            sys.exit(-1)

    return training_directory

def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["callbacks"] = DQNCallbacks

    return config

def run(args):
    try:
        checkpoint = False
        if args.restore:
            checkpoint = find_latest_checkpoint(args)

        kill_all_servers()
        ray.init(auto="address")
        tune.run(
            CustomDQNTrainer,
            name=args.name,
            local_dir=args.directory,
            stop={"perf/ram_util_percent": 85.0},
            checkpoint_freq=1,
            checkpoint_at_end=True,
            restore=checkpoint,
            config=args.config
        )

    finally:
        kill_all_servers()
        ray.shutdown()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory",
                           metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/carla_rllib")
    argparser.add_argument("-n", "--name",
                           metavar="N",
                           default="dqn_example",
                           help="Name of the experiment (default: dqn_example)")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument("--overwrite",
                           action="store_true",
                           default=False,
                           help="Flag to overwrite a specific directory (warning: all content of the folder will be lost.)")

    args = argparser.parse_args()
    args.training_directory = manage_training_directory(args)
    args.config = parse_config(args)
    run(args)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
