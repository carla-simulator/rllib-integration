#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import copy
import queue
import math

import carla

import numpy as np


# ==================================================================================================
# -- base sensor -----------------------------------------------------------------------------------
# ==================================================================================================

# BirdviewManager should inherit from this class
class BaseSensor(object):
    def __init__(self, name, attributes, interface, parent):
        self.name = name
        self.attributes = attributes
        self.interface = interface
        self.parent = parent

        self.interface.register(self.name, self)

    def is_event_sensor(self):
        return False

    def parse(self):
        raise NotImplementedError

    def callback(self, data):
        if not self.is_event_sensor():
            self.interface._data_buffers.put((self.name, self.parse(data)))
        else:
           self.interface._event_data_buffers.put((self.name, self.parse(data)))

    def destroy(self):
        raise NotImplementedError


class CarlaSensor(BaseSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

        world = self.parent.get_world()

        type_ = self.attributes.pop("type", "")
        transform = self.attributes.pop("transform", "0,0,0,0,0,0")
        transform = [float(x) for x in transform.split(",")]
        assert len(transform) == 6

        blueprint = world.get_blueprint_library().find(type_)
        blueprint.set_attribute("role_name", name)
        for key, value in attributes.items():
            blueprint.set_attribute(str(key), str(value))

        transform = carla.Transform(
            carla.Location(transform[0], transform[1], transform[2]),
            carla.Rotation(transform[3], transform[4], transform[5])
        )
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=self.parent)

        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


# ==================================================================================================
# -- cameras -----------------------------------------------------------------------------------
# ==================================================================================================
class BaseCamera(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class CameraRGB(BaseCamera):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraDepth(BaseCamera):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraSemanticSegmentation(BaseCamera):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraDVS(CarlaSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # TODO
        return []


# ==================================================================================================
# -- lidar -----------------------------------------------------------------------------------
# ==================================================================================================
class Lidar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    # TODO(joel): Check this!
    def parse(self, sensor_data):
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points


class SemanticLidar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # TODO
        return []


# ==================================================================================================
# -- others -----------------------------------------------------------------------------------
# ==================================================================================================
class Radar(CarlaSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    # TODO(joel): This is different to what we have (copied from leaderboard)
    def parse(self, sensor_data):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        return points


class Gnss(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        return [sensor_data.latitude, sensor_data.longitude]


class Imu(CarlaSensor):
    LIMITS = (-99.9, 99.9)

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        accelerometer = (
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], sensor_data.accelerometer.x)),
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], sensor_data.accelerometer.y)),
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], sensor_data.accelerometer.z)))
        gyroscope = (
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], math.degrees(sensor_data.gyroscope.x))),
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], math.degrees(sensor_data.gyroscope.y))),
            max(Imu.LIMITS[0], min(Imu.LIMITS[1], math.degrees(sensor_data.gyroscope.z))))
        compass = math.degrees(sensor_data.compass)
        return [accelerometer, gyroscope, compass]


class LaneInvasion(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        for crossed_lane_marking in sensor_data.crossed_lane_markings:
            if crossed_lane_marking.type == carla.LaneMarkingType.Curb or \
                crossed_lane_marking.type == carla.LaneMarkingType.Grass:
                return True
        return False


class Collision(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        impulse = sensor_data.normal_impulse
        return math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
