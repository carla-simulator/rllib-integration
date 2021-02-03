#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import sys
import math

import numpy as np
from gym.spaces import Box, Discrete

import carla

from rllib_integration.base_experiment import *
from rllib_integration.helper import post_process_image

from PIL import Image


# EXPERIMENT_CONFIG = {
#     "Server_View": SERVER_VIEW_CONFIG,
#     "town": "Town02_Opt",
#     "n_vehicles": 0,
#     "n_walkers": 0,
#     "hero_blueprint": "vehicle.lincoln.mkz2017",
# }

class DQNExperiment(BaseExperiment):
    def __init__(self, user_config={}):
        super().__init__(user_config)
        self.config.update(user_config)

        self.frame_stack = self.config["framestack"]
        self.max_idle = 600 # ticks
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.done_idle = False
        self.done_falling = False

        # hero variables
        self.last_location = self.hero.get_transform().location
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

    def get_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.config["sensors"]["birdview"]["size"],
                self.config["sensors"]["birdview"]["size"],
                num_of_channels * self.config["framestack"],
            ),
            dtype=np.uint8,
        )
        return image_space

    def get_action_space(self):
        """:return: None. In this experiment, it is a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            2: [0.0, 0.75, 0.0, False, False],  # Right
            3: [0.0, 0.50, 0.0, False, False],  # Right
            4: [0.0, 0.25, 0.0, False, False],  # Right
            5: [0.0, -0.75, 0.0, False, False],  # Left
            6: [0.0, -0.50, 0.0, False, False],  # Left
            7: [0.0, -0.25, 0.0, False, False],  # Left
            8: [0.3, 0.00, 0.0, False, False],  # Straight
            9: [0.3, 0.75, 0.0, False, False],  # Right
            10: [0.3, 0.50, 0.0, False, False],  # Right
            11: [0.3, 0.25, 0.0, False, False],  # Right
            12: [0.3, -0.75, 0.0, False, False],  # Left
            13: [0.3, -0.50, 0.0, False, False],  # Left
            14: [0.3, -0.25, 0.0, False, False],  # Left
            15: [0.6, 0.00, 0.0, False, False],  # Straight
            16: [0.6, 0.75, 0.0, False, False],  # Right
            17: [0.6, 0.50, 0.0, False, False],  # Right
            18: [0.6, 0.25, 0.0, False, False],  # Right
            19: [0.6, -0.75, 0.0, False, False],  # Left
            20: [0.6, -0.50, 0.0, False, False],  # Left
            21: [0.6, -0.25, 0.0, False, False],  # Left
            22: [1.0, 0.00, 0.0, False, False],  # Straight
            23: [1.0, 0.75, 0.0, False, False],  # Right
            24: [1.0, 0.50, 0.0, False, False],  # Right
            25: [1.0, 0.25, 0.0, False, False],  # Right
            26: [1.0, -0.75, 0.0, False, False],  # Left
            27: [1.0, -0.50, 0.0, False, False],  # Left
            28: [1.0, -0.25, 0.0, False, False],  # Left
        }

    def update_actions(self, action, hero):
        """Given the action, moves the hero vehicle"""

        action_control = self.get_actions()[int(action)]

        self.action.throttle = action_control[0]
        self.action.steer = action_control[1]
        self.action.brake = action_control[2]
        self.action.reverse = action_control[3]
        self.action.hand_brake = action_control[4]

        hero.apply_control(self.action)

    def process_observation(self, observation):
        """
        Process observations according to your experiment
        :param core:
        :param observation:
        :return:
        """
        image = post_process_image(observation['birdview'], normalized = False, grayscale = False)

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

    def get_speed(self):
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        vel = self.hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def compute_reward(self, observation, map_):
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
            reward += -3
        if self.done_idle:
            reward += -1

        self.time_idle += 1

        return reward
