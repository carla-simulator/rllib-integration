#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import enum
import math
import random

import carla
from gym.spaces import Discrete

from rllib_integration.helper import post_process_image


class SensorsTransformEnum(enum.Enum):
    Transform_A = 0  # (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm)
    Transform_B = 1  # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
    Transform_c = 2  # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
    Transform_D = 3  # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm)
    Transform_E = 4  # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]


class SensorsEnum(enum.Enum):
    CAMERA_RGB = 0
    CAMERA_DEPTH_RAW = 1
    CAMERA_DEPTH_GRAY = 2
    CAMERA_DEPTH_LOG = 3
    CAMERA_SEMANTIC_RAW = 4
    CAMERA_SEMANTIC_CITYSCAPE = 5
    LIDAR = 6
    CAMERA_DYNAMIC_VISION = 7
    CAMERA_DISTORTED = 8


BASE_SERVER_VIEW_CONFIG = {
    "server_view_x_offset": 00,
    "server_view_y_offset": 00,
    "server_view_height": 200,
    "server_view_pitch": -90,
}

BASE_SENSOR_CONFIG = {
    "SENSOR": [SensorsEnum.CAMERA_DEPTH_RAW],
    "SENSOR_TRANSFORM": [SensorsTransformEnum.Transform_A],
    "CAMERA_X": 84,
    "CAMERA_Y": 84,
    "CAMERA_FOV": 60,
    "CAMERA_NORMALIZED": [True],
    "CAMERA_GRAYSCALE": [True],
    "FRAMESTACK": 1,
}

BASE_BIRDVIEW_CONFIG = {
    "SIZE": 300,
    "RADIUS": 20,
    "FRAMESTACK": 4
}

BASE_OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": [False],
    "COLLISION_OBSERVATION": True,
    "LOCATION_OBSERVATION": True,
    "RADAR_OBSERVATION": False,
    "IMU_OBSERVATION": False,
    "LANE_OBSERVATION": True,
    "GNSS_OBSERVATION": False,
    "BIRDVIEW_OBSERVATION": False,
}
BASE_EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": BASE_OBSERVATION_CONFIG,
    "Server_View": BASE_SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": BASE_SENSOR_CONFIG,
    "BIRDVIEW_CONFIG": BASE_BIRDVIEW_CONFIG,
    "town": "Town02_Opt",
    "quality_level": "Low",  # options are low or Epic #ToDO. This does not do anything + change to enum
    "Disable_Rendering_Mode": False,  # If you disable, you will not get camera images
    "n_vehicles": 0,
    "n_walkers": 0,
    "end_pos_spawn_id": 45,  # 34,
    "hero_blueprint": "vehicle.lincoln.mkz2017",
    "Weather": carla.WeatherParameters.ClearNoon,
    "DISCRETE_ACTION": True,
    "Debug": False,
}

DISCRETE_ACTIONS_SMALL = {
    0: [0.0, 0.00, 0.0, False, False],  # Coast
    1: [0.0, 0.00, 1.0, False, False],  # Apply Break
    2: [0.6, 0.00, 0.0, False, False],  # Straight
    #3: [0.3, 0.00, 0.0, False, False],  # Straight
    # 4: [0.0, 0.75, 0.0, False, False],  # Right
    # 5: [0.0, 0.50, 0.0, False, False],  # Right
    # 6: [0.0, -0.50, 0.0, False, False],  # Left
    # 7: [0.0, -0.75, 0.0, False, False],  # Left
    3: [0.6, 0.75, 0.0, False, False],  # Right + Accelerate
    4: [0.6, 0.50, 0.0, False, False],  # Right + Accelerate
    5: [0.6, -0.50, 0.0, False, False],  # Left + Accelerate
    6: [0.6, -0.75, 0.0, False, False],  # Left + Accelerate
}


DISCRETE_ACTIONS = DISCRETE_ACTIONS_SMALL


class BaseExperiment(object):
    def __init__(self, user_config):
        self.experiment_config = BASE_EXPERIMENT_CONFIG.copy()
        self.experiment_config.update(user_config)
        self.observation = {}
        self.observation["camera"] = []
        self.observation_space = None
        self.action = None
        self.action_space = None

        self.hero = None
        self.spectator = None
        self.spawn_point_list = []
        self.vehicle_list = []
        self.start_location = None
        self.end_location = None
        self.current_w = None
        self.hero_model = ''.join(self.experiment_config["hero_blueprint"])
        self.set_observation_space()
        self.set_action_space()
        self.max_idle = 600 # ticks
        self.max_ep_time = 3200 # ticks
        self.t_idle = None
        self.t_ep = None

        self.done_idle = False
        self.done_max_time = False
        self.done_falling = False

    def get_experiment_config(self):

        return self.experiment_config

    def set_observation_space(self):

        """
        observation_space_option: Camera Image
        :return: observation space:
        """
        raise NotImplementedError

    def get_observation_space(self):

        """
        :return: observation space
        """
        return self.observation_space

    def set_action_space(self):

        """
        :return: None. In this experiment, it is a discrete space
        """
        self.action_space = Discrete(len(DISCRETE_ACTIONS))

    def get_action_space(self):

        """
        :return: action_space. In this experiment, it is a discrete space
        """
        return self.action_space

    def set_server_view(self,core):

        """
        Set server view to be behind the hero
        :param core:Carla Core
        :return:
        """
        # spectator following the car
        transforms = self.hero.get_transform()
        server_view_x = self.hero.get_location().x - 5 * transforms.get_forward_vector().x
        server_view_y = self.hero.get_location().y - 5 * transforms.get_forward_vector().y
        server_view_z = self.hero.get_location().z + 3
        server_view_pitch = transforms.rotation.pitch
        server_view_yaw = transforms.rotation.yaw
        server_view_roll = transforms.rotation.roll
        self.spectator = core.get_core_world().get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def get_speed(self):
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        vel = self.hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self):
        return NotImplementedError

    def process_observation(self, core, observation):

        """
        Main function to do all the post processing of observations. This is an example.
        :param core:
        :param observation:
        :return:
        """
        observation['camera'] = post_process_image(
                                            observation['camera'],
                                            normalized = self.experiment_config["SENSOR_CONFIG"]["CAMERA_NORMALIZED"][0],
                                            grayscale = self.experiment_config["SENSOR_CONFIG"]["CAMERA_GRAYSCALE"][0]
            )

        return observation

    def get_observation(self, core):

        info = {}
        for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
            if len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"]) != len(self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]):
                raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
            if self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"][i]:
                self.observation['camera'].append(core.get_camera_data())
        if self.experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            self.observation["collision"] = core.get_collision_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["LOCATION_OBSERVATION"]:
            self.observation["location"] = self.hero.get_transform()
        if self.experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            self.observation["lane_invasion"] = core.get_lane_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["GNSS_OBSERVATION"]:
            self.observation["gnss"] = core.get_gnss_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["IMU_OBSERVATION"]:
            self.observation["imu"] = core.get_imu_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["RADAR_OBSERVATION"]:
            self.observation["radar"] = core.get_radar_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["BIRDVIEW_OBSERVATION"]:
            self.observation["birdview"] = core.get_birdview_data()

        info["control"] = {
            "steer": self.action.steer,
            "throttle": self.action.throttle,
            "brake": self.action.brake,
            "reverse": self.action.reverse,
            "hand_brake": self.action.hand_brake,
        }

        return self.observation, info

    def update_measurements(self, core):

        for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
            if len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"]) != len(self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]):
                raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
            if self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"][i]:
                core.update_camera()
        if self.experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            core.update_collision()
        if self.experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            core.update_lane_invasion()

    def update_actions(self, action, hero):
        if action is None:
            self.action = carla.VehicleControl()
        else:
            action = DISCRETE_ACTIONS[int(action)]
            self.action.throttle = action[0]
            self.action.steer = action[1]
            self.action.brake = action[2]
            self.action.reverse = action[3]
            self.action.hand_brake = action[4]
            hero.apply_control(self.action)

    def compute_reward(self, core, observation):

        """
        :param core:
        :param observation:
        :return:
        """

        print("This is a base experiment. Make sure you make you own reward computing function")
        return NotImplementedError

    def initialize_reward(self, core):

        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        print("This is a base experiment. Make sure you make you own reward initialization function")
        raise NotImplementedError


    # ==============================================================================
    # -- Hero -----------------------------------------------------------
    # ==============================================================================
    def spawn_hero(self, world, transform, autopilot=False):

        """
        This function spawns the hero vehicle. It makes sure that if a hero exists, it destroys the hero and respawn
        :param core:
        :param transform: Hero location
        :param autopilot: Autopilot Status
        :return:
        """

        self.spawn_points = world.get_map().get_spawn_points()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        self.end_location = self.spawn_points[self.experiment_config["end_pos_spawn_id"]]

        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        i = 0
        random.shuffle(self.spawn_points, random.random)
        while True:
            next_spawn_point = self.spawn_points[i % len(self.spawn_points)]
            self.hero = world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if self.hero is not None:
                break
            else:
                print("Could not spawn Hero, changing spawn point")
                i+=1

        world.tick()
        print("Hero spawned!")
        self.start_location = self.spawn_points[i].location
        self.past_action = carla.VehicleControl(0.0, 0.00, 0.0, False, False)
        self.t_idle = 0
        self.t_ep = 0

    def get_hero(self):

        """
        Get hero vehicle
        :return:
        """
        return self.hero

    # ==============================================================================
    # -- Tick -----------------------------------------------------------
    # ==============================================================================

    def experiment_tick(self, core, world, action):

        """
        This is the "tick" logic.
        :param core:
        :param action:
        :return:
        """

        world.tick()
        self.t_idle += 1
        self.t_ep += 1
        self.update_measurements(core)
        self.update_actions(action, self.hero)
