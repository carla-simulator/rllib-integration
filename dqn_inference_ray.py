#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import yaml

import ray
from ray.rllib.agents.dqn import DQNTrainer

from dqn_example.dqn_experiment import DQNExperiment
from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers



def main(args):
    # Load configuration from file params.json
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "..", "params.json")

    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = DQNExperiment
        config["num_workers"] = 0
        config["explore"] = False
        del config["num_cpus_per_worker"]
        del config["num_gpus_per_worker"]

    # Restore agent
    agent = DQNTrainer(env=CarlaEnv, config=config)
    agent.restore(args.checkpoint)

    env = agent.workers.local_worker().env

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: {}".format(episode_reward))
    return episode_reward


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument(
        "checkpoint",
        type=str,
        help="Checkpoint from which to roll out.")
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        ray.init()
        main(arguments)
        ray.shutdown()
    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        kill_all_servers()
