#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore


class CarlaEnv(gym.Env):
    """
    This is a CARLA environment, responsible of handling all the CARLA related parts of the training.
    """

    def __init__(self, config):
        """
        Initializes the environment.
        :param config: Input from the configuration file.
        """
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.set_observation_space()

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        """
        Resets the esperiment to start a new episode.
        :return: The observation
        """
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observation
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data)

        return observation

    def step(self, action):
        """
        Computes one tick of the simulation and of the environment,
        in order to return the new observation, as well as the rewards
        :param action: The control that we apply
        """

        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data)
        done = self.experiment.get_done_status(self.core)
        reward = self.experiment.compute_reward(self.core)

        return observation, reward, done, info
