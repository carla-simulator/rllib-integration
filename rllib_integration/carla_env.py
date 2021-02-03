#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
This is a sample carla environment. It does basic functionality.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore


class CarlaEnv(gym.Env):

    def __init__(self, config):
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.experiment_config = self.experiment.config

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment_config)

        self.world = self.core.get_core_world()
        self.map = self.world.get_map()

        self.reset()

    def reset(self):
        # Reset sensors hero and 
        self.core.reset_sensors(self.experiment_config["sensors"]) # TODO: join reset_sensors and setup_sensors
        self.core.reset_hero(self.experiment_config)
        self.hero = self.core.get_hero()
        self.core.setup_sensors(self.experiment.config["sensors"], self.hero, self.config["carla"]["sync_mode"])

        # Save hero realted information
        self.experiment.set_hero(self.hero)

        # Reset the experiment
        self.experiment.reset()

        # Tick once
        self.core.tick()
        observation, info = self.experiment.get_observation(self.core)
        observation = self.experiment.process_observation(observation)
        return observation

    def step(self, action):
        self.core.tick()
        self.experiment.update_actions(action, self.hero)
        observation, info = self.experiment.get_observation(self.core)
        observation = self.experiment.process_observation(observation)
        reward = self.experiment.compute_reward(observation, self.map)
        done = self.experiment.get_done_status()

        return observation, reward, done, info
