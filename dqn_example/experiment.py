#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import sys
import math

import numpy as np
from gym.spaces import Box

import carla

from rllib_integration.base_experiment import *
from rllib_integration.helper import post_process_image

from PIL import Image

SERVER_VIEW_CONFIG = {
}

SENSOR_CONFIG = {
    "SENSOR": [],
    "CAMERA_NORMALIZED": [True], # apparently doesnt work if set to false, its just for the image!
    "CAMERA_GRAYSCALE": [True],
    "FRAMESTACK": 4,
}

BIRDVIEW_CONFIG = {
    "SIZE": 300,
    "RADIUS": 15,
    "FRAMESTACK": 4
}

OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": [False],
    "COLLISION_OBSERVATION": True,
    "LOCATION_OBSERVATION": True,
    "RADAR_OBSERVATION": False,
    "IMU_OBSERVATION": False,
    "LANE_OBSERVATION": True,
    "GNSS_OBSERVATION": False,
    "BIRDVIEW_OBSERVATION": True,
}

EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": OBSERVATION_CONFIG,
    "Server_View": SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": SENSOR_CONFIG,
    "town": "Town02_Opt",
    "BIRDVIEW_CONFIG": BIRDVIEW_CONFIG,
    "n_vehicles": 0,
    "n_walkers": 0,
    "hero_blueprint": "vehicle.lincoln.mkz2017",
}


class DQNExperiment(BaseExperiment):
    def __init__(self, user_config={}):
        config = EXPERIMENT_CONFIG.copy()
        config.update(user_config)

        super().__init__(config)

    def initialize_reward(self, core):
        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        self.previous_distance = 0
        self.i = 0
        self.frame_stack = 4  # can be 1,2,3,4
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

    def get_observation(self, core):
        obs = core.sensor_interface.get_data()
        print("Data received: {}".format(obs.keys()))
        
        info = {}
        info["control"] = {
            "steer": self.action.steer,
            "throttle": self.action.throttle,
            "brake": self.action.brake,
            "reverse": self.action.reverse,
            "hand_brake": self.action.hand_brake,
        }
        return obs, info


    def process_observation(self, core, observation):
        """
        Process observations according to your experiment
        :param core:
        :param observation:
        :return:
        """
        # if self.i % 100 == 0:
        #     img = Image.fromarray(observation["birdview"], 'RGB')
        #     img.show()
        # self.i += 1

        self.set_server_view(core)
        image = post_process_image(observation['birdview'],
                                   normalized = False,
                                   grayscale = False
        )

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image

        return images

    def set_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.experiment_config["BIRDVIEW_CONFIG"]["SIZE"],
                self.experiment_config["BIRDVIEW_CONFIG"]["SIZE"],
                num_of_channels * self.experiment_config["BIRDVIEW_CONFIG"]["FRAMESTACK"],
            ),
            dtype=np.uint8,
        )
        self.observation_space = image_space

    def get_done_status(self):
        #done = self.observation["collision"] is not False or not self.check_lane_type(map)
        # self.done_idle = self.max_idle < self.t_idle
        # if self.get_speed() > 2.0:
        #     self.t_idle = 0
        # self.done_max_time = self.max_ep_time < self.t_ep
        # self.done_falling = self.hero.get_location().z < -0.5
        return self.done_idle or self.done_max_time or self.done_falling

    def inside_lane(self, map):
        self.current_w = map.get_waypoint(self.hero.get_location(), lane_type=carla.LaneType.Any)
        return self.current_w.lane_type in self.allowed_types

    def dist_to_driving_lane(self, map_):
        cur_loc = self.hero.get_location()
        cur_w = map_.get_waypoint(cur_loc)
        return math.sqrt((cur_loc.x - cur_w.transform.location.x)**2 +
                         (cur_loc.y - cur_w.transform.location.y)**2)

    def compute_reward(self, core, observation, map, world):
        """
        Reward function
        :param observation:
        :param core:
        :return:
        """
        # TODO
        return 0
