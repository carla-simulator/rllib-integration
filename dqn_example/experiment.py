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
    "FRAMESTACK": 1
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
        self.last_location = self.start_location
        self.last_velocity = self.get_speed()
        self.frame_stack = 1  # can be 1,2,3,4
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
        self.done_idle = self.max_idle < self.time_idle
        if self.get_speed() > 1.0:
            self.time_idle = 0
        self.done_falling = self.hero.get_location().z < -0.5
        return self.done_idle or self.done_falling

    def find_current_waypoint(self, map_):
        return map_.get_waypoint(self.hero.get_location(), lane_type=carla.LaneType.Any)

    def inside_lane(self, waypoint):
        return waypoint.lane_type in self.allowed_types

    def compute_reward(self, core, observation, map_):
        """
        Reward function
        :param observation:
        :param core:
        :return:
        """

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def compute_angle(u, v):
            return -math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])

        # Hero-related variables
        hero_waypoint = self.find_current_waypoint(map_)
        hero_location = self.hero.get_location()
        hero_velocity = self.get_speed()
        hero_heading = self.hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]
        wp_heading = hero_waypoint.transform.get_forward_vector()
        wp_heading = [wp_heading.x, wp_heading.y]
        hero_to_wp = unit_vector([
            hero_waypoint.transform.location.x - hero_location.x,
            hero_waypoint.transform.location.y - hero_location.y
        ])

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        delta_velocity = hero_velocity - self.last_velocity
        dot_product = np.dot(hero_heading, wp_heading)
        angle = compute_angle(hero_heading, hero_to_wp)

        # Update varibles
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Calculate reward
        reward = 0

        # Reward if going forward
        if delta_distance > 0:
            reward += 10*delta_distance

        # Reward if going faster than last step
        reward += 0.05 * delta_velocity

        # Penalize if not inside the lane
        if not self.inside_lane(hero_waypoint):
            reward += -0.5

        if dot_product < 0.0 and not(hero_waypoint.is_junction):
            reward += -0.5

        if self.done_falling:
            print("Done falling")
            reward += -3
        if self.done_idle:
            print("Done idle")
            reward += -1

        return reward