#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Here are defined all the CARLA sensors
"""

import copy
import math
import numpy as np

import carla

# ==================================================================================================
# -- BaseSensor -----------------------------------------------------------------------------------
# ==================================================================================================

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

    def update_sensor(self, data, frame):
        if not self.is_event_sensor():
            self.interface._data_buffers.put((self.name, frame, self.parse(data)))
        else:
           self.interface._event_data_buffers.put((self.name, frame, self.parse(data)))

    def callback(self, data):
        self.update_sensor(data, data.frame)

    def destroy(self):
        raise NotImplementedError


class CarlaSensor(BaseSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

        world = self.parent.get_world()

        type_ = self.attributes.pop("type", "")
        transform = self.attributes.pop("transform", "0,0,0,0,0,0")
        if isinstance(transform, str):
            transform = [float(x) for x in transform.split(",")]
        assert len(transform) == 6

        blueprint = world.get_blueprint_library().find(type_)
        blueprint.set_attribute("role_name", name)
        for key, value in attributes.items():
            blueprint.set_attribute(str(key), str(value))

        transform = carla.Transform(
            carla.Location(transform[0], transform[1], transform[2]),
            carla.Rotation(transform[4], transform[5], transform[3])
        )
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=self.parent)

        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


class PseudoSensor(BaseSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def callback(self, data, frame):
        self.update_sensor(data, frame)


# ==================================================================================================
# -- Cameras -----------------------------------------------------------------------------------
# ==================================================================================================
class BaseCamera(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the Image into an numpy array"""
        # sensor_data: [fov, height, width, raw_data]
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

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        """Parses the DVSEvents into an RGB image"""
        # sensor_data: [x, y, t, polarity]
        dvs_events = np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))

        dvs_img = np.zeros((sensor_data.height, sensor_data.width, 3), dtype=np.uint8)
        dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255 # Blue is positive, red is negative

        return dvs_img

# ==================================================================================================
# -- LIDAR -----------------------------------------------------------------------------------
# ==================================================================================================
class Lidar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the LidarMeasurememt into an numpy array"""
        # sensor_data: [x, y, z, intensity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points


class SemanticLidar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the SemanticLidarMeasurememt into an numpy array"""
        # sensor_data: [x, y, z, cos(angle), actor index, semantic tag]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        return points


# ==================================================================================================
# -- Others -----------------------------------------------------------------------------------
# ==================================================================================================
class Radar(CarlaSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the RadarMeasurement into an numpy array"""
        # sensor_data: [depth, azimuth, altitute, velocity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        return points


class Gnss(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the GnssMeasurement into an numpy array"""
        # sensor_data: [latitude, longitude, altitude]
        return np.array([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude], dtype=np.float64)


class Imu(CarlaSensor):

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the IMUMeasurement into an numpy array"""
        # sensor_data: [accelerometer, gyroscope, compass]
        return np.array([sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z,
                          sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z,
                          sensor_data.compass,
                        ], dtype=np.float64)


class LaneInvasion(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        """Parses the IMUMeasurement into a list"""
        # sensor_data: [transform, lane marking]
        return [sensor_data.transform, sensor_data.crossed_lane_markings]


class Collision(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        self._last_event_frame = 0
        super().__init__(name, attributes, interface, parent)

    def callback(self, data):
        # The collision sensor can have multiple callbacks per tick. Get only the first one
        if self._last_event_frame != data.frame:
            self._last_event_frame = data.frame
            self.update_sensor(data, data.frame)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        """Parses the ObstacleDetectionEvent into a list"""
        # sensor_data: [other actor, distance]
        impulse = sensor_data.normal_impulse
        impulse_value = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        return [sensor_data.other_actor, impulse_value]

class Obstacle(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        """Parses the ObstacleDetectionEvent into a list"""
        # sensor_data: [other actor, distance]
        return [sensor_data.other_actor, sensor_data.distance]
