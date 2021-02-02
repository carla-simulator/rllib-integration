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
        self.experiment_config = self.experiment.get_experiment_config()

        self.core = CarlaCore(self.config, self.experiment_config, self.config["carla"])

        self.world = self.core.get_core_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        self.reset()

    def reset(self):
        self.core.reset_sensors(self.experiment_config["sensors"])

        self.experiment.spawn_hero(self.world, self.spawn_points, autopilot=False)

        self.core.setup_sensors(
            self.experiment.experiment_config["sensors"],
            self.experiment.get_hero(),
            self.world.get_settings().synchronous_mode,
        )

        self.experiment.initialize_reward(self.core)
        self.experiment.set_server_view(self.core)
        self.experiment.tick(self.core, self.world, action=None)
        obs, info = self.experiment.get_observation(self.core)
        obs = self.experiment.process_observation(self.core, obs)
        return obs

    def step(self, action):
        self.experiment.tick(self.core, self.world, action)
        observation, info = self.experiment.get_observation(self.core)
        observation = self.experiment.process_observation(self.core, observation)
        reward = self.experiment.compute_reward(self.core,observation, self.map)
        done = self.experiment.get_done_status()

        return observation, reward, done, info
