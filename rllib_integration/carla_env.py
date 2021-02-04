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

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):

        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once
        sensor_data = self.core.tick(None)

        observation, _ = self.experiment.get_observation(sensor_data)
        return observation

    def step(self, action):
        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data)
        reward = self.experiment.compute_reward(observation, self.core)
        done = self.experiment.get_done_status(observation, self.core)

        return observation, reward, done, info
