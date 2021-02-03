#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla

BASE_EXPERIMENT_CONFIG = {
    "hero_blueprint": "vehicle.lincoln.mkz2017",
    "town": "Town05_Opt",
    "n_vehicles": 0,
    "n_walkers": 0,
    "sensors": {
        # "sensor_name1": {
        #     "attribute1": attribute_value1
        #     "attribute2": attribute_value2
        # }
        # "sensor_name2": {
        #     "attribute_name1": attribute_value1
        #     "attribute_name2": attribute_value2
        # }
    },
    "weather": carla.WeatherParameters.ClearNoon, #TODO: use it
}

class BaseExperiment(object):
    def __init__(self, user_config):
        self.config = BASE_EXPERIMENT_CONFIG.copy()
        self.config.update(user_config)
        self.hero = None
        self.action = carla.VehicleControl()

    def set_hero(self, hero):
        """Sets the ego vehicle"""
        self.hero = hero

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    def get_actions(self):
        """Returns the actions"""
        raise NotImplementedError

    def get_observation(self, core):
        obs = core.sensor_interface.get_data()
        return obs, {}

    def get_action_space(self):
        """Returns the action space"""
        raise NotImplementedError

    def update_actions(self, action, hero):
        """Given the action, moves the hero vehicle"""
        raise NotImplementedError

    def get_observation_space(self):
        """Returns the observation space"""
        raise NotImplementedError

    def process_observation(self, observation):
        """Main function to do all the post processing of observations (sensor data)."""
        return NotImplementedError

    def get_done_status(self):
        """Returns whether or not the experiment has to end"""
        return NotImplementedError

    def initialize_reward(self):
        """Initialization of reward function"""
        raise NotImplementedError

    def compute_reward(self):
        """Computes the reward"""
        return NotImplementedError
