#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import time
import sys
import cv2
import numpy as np
from rllib_integration.helper.list_procs import search_procs_by_name
import signal

import collections


def get_parent_dir(directory):
    return os.path.dirname(directory)


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: normalized image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def kill_server():
    """
    Kill all PIDs that start with Carla. Do this if you running a single server
    :return:
    """
    for pid, name in search_procs_by_name("Carla").items():
        os.kill(pid, signal.SIGKILL)

def spawn_vehicle_at(transform, vehicle_blueprint, world, autopilot=True, max_time=0.1):
    """
    Try to spawn a vehicle and give the vehicle time to be spawned and seen by the world before you say it is spawned

    :param transform: Location and Orientation of vehicle
    :param vehicle_blueprint: Vehicle Blueprint (We assign a random color)
    :param world: World
    :param autopilot: If True, AutoPilot is Enabled. If False, autopilot is disabled
    :param max_time: Maximum time in s to wait before you say that vehicle can not be spawned at current location
    :return: True if vehicle was added to world and False otherwise
    """

    # If the vehicle can not be spawned, it is OK
    previous_number_of_vehicles = len(world.get_actors().filter("*vehicle*"))

    # Get a random color
    color = random.choice(vehicle_blueprint.get_attribute("color").recommended_values)
    vehicle_blueprint.set_attribute("color", color)

    vehicle = world.try_spawn_actor(vehicle_blueprint, transform)

    wait_tick = 0.002  # Wait of 2ms to recheck if a vehicle is spawned
    if vehicle is not None:
        vehicle.set_autopilot(autopilot)
        world.tick()  # Tick the world so it creates the vehicle
        while previous_number_of_vehicles >= len(
            world.get_actors().filter("*vehicle*")
        ):
            time.sleep(wait_tick)  # Wait 2ms and check again
            max_time = max_time - wait_tick
            if max_time <= 0:  # Check for expiration time
                return False
        return vehicle
    return False

