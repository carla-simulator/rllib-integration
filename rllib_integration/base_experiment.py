#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from rllib_integration.helper import join_dicts

BASE_EXPERIMENT_CONFIG = {
    "hero": {
        "blueprint": "vehicle.lincoln.mkz2017",
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
        "spawn_points": [
            # "0,0,0,0,0,0",  # x,y,z,roll,pitch,yaw
        ]
    },
    "background_activity": {
        "n_vehicles": 0,
        "n_walkers": 0,
        "tm_hybrid_mode": True,
        "seed": None
    },
    "town": "Town05_Opt",
    "weather": 'ClearNoon'
}

class BaseExperiment(object):
    def __init__(self, config):
        self.config = join_dicts(BASE_EXPERIMENT_CONFIG, config)

    def reset(self):
        """
        Initializes variables at the beginning and each time the simulation is reset.
        """
        pass

    def set_observation_space(self):
        """
        Creates the observation space.
        """
        raise NotImplementedError

    def set_action_space(self):
        """
        Creates the action space.
        """
        raise NotImplementedError

    def get_action_space(self):
        """
        Returns the action space, in this case, a discrete space.
        """
        raise NotImplementedError

    def compute_action(self, action):
        """
        Given the action, returns a carla.VehicleControl() which will be applied to the hero.
        :param action: The selected action
        """
        raise NotImplementedError

    def get_observation(self, sensor_data):
        """
        Function that does all the post processing of observations (sensor data).

        :param sensor_data: A dictionary of sensors {sensor_name: sensor_data}

        :return: A tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty.
        """
        return NotImplementedError

    def get_done_status(self, observation, core):
        """
        Returns whether or not the experiment has to end.
        :param core: The CARLA core
        :returns: If the current episode is done or not
        """
        return NotImplementedError

    def compute_reward(self, observation, core):
        """
        Computes the reward, based on the specified parameters.
        :param core: The CARLA core
        """
        return NotImplementedError
