#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import math
import weakref
import queue

import carla
import numpy as np


class CollisionSensor(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self.intensity = False
        self._parent = parent_actor
        self.synchronous_mode = synchronous_mode
        self.world = self._parent.get_world()
        self.bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(
            self.bp, carla.Transform(), attach_to=self._parent
        )

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: CollisionSensor._on_collision(weak_self, event)
            )
        else:
            self.collision_queue = None
            self.collision_queue = queue.Queue()
            self.sensor.listen(self.collision_queue.put)

    def read_collision_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.intensity
        else:
            try:
                CollisionSensor._on_collision(
                    weak_self, self.collision_queue.get(False)
                )
            except:
                pass

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
            self.intensity = False

    def get_collision_data(self):
        if self.intensity is not False:
            intensity = self.intensity
            self.intensity = False
            return intensity
        else:
            return False

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        self.intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self._parent = parent_actor
        self.lane_markings = []
        self.synchronous_mode = synchronous_mode
        self.world = self._parent.get_world()
        self.bp = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = self.world.spawn_actor(
            self.bp, carla.Transform(), attach_to=self._parent
        )

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
            )
        else:
            self.lane_queue = None
            self.lane_queue = queue.Queue()
            self.sensor.listen(self.lane_queue.put)

    def read_lane_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.get_lane_data()
        else:
            try:
                LaneInvasionSensor._on_invasion(
                    weak_self, self.lane_queue.get(False)
                )
            except:
                pass

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        for x in lane_types:
            self.lane_markings.append(str(x))
        return

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    def get_lane_data(self):
        for x in self.lane_markings:
            #if x in ['Solid','SolidSolid','Curb','Grass', 'NONE', 'Broken']:
            if x in ['Curb','Grass']:
                return True
        else:
            return False


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.world = self._parent.get_world()
        self.synchronous_mode = synchronous_mode
        self.bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = self.world.spawn_actor(self.bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))
        else:
            self.gnss_queue = None
            self.gnss_queue = queue.Queue()
            self.sensor.listen(self.gnss_queue.put)

    def read_gnss_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.get_gnss_data()
        else:
            try:
                GnssSensor._on_gnss_event(weak_self, self.gnss_queue.get(False))
            except:
                pass

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    def get_gnss_data(self):
        return [self.lat, self.lon]


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        self.synchronous_mode = synchronous_mode
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))
        else:
            self.imu_queue = None
            self.imu_queue = queue.Queue()
            self.sensor.listen(self.imu_queue.put)

    def read_imu_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.get_imu_data()
        else:
            try:
                IMUSensor._IMU_callback(weak_self, self.imu_queue.get(False))
            except:
                pass

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    def get_imu_data(self):
        return [self.accelerometer, self.gyroscope, self.compass]


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor, synchronous_mode=True):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.synchronous_mode = synchronous_mode
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.points = None
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        if not self.synchronous_mode:
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
        else:
            self.radar_queue = None
            self.radar_queue = queue.Queue()
            self.sensor.listen(self.radar_queue.put)

    def read_radar_queue(self):
        weak_self = weakref.ref(self)
        if not self.synchronous_mode:
            return self.get_radar_data()
        else:
            try:
                RadarSensor._Radar_callback(weak_self, self.radar_queue.get(False))
            except:
                pass

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        self.points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        self.points = np.reshape(self.points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

    def destroy_sensor(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    def get_radar_data(self):
        return self.points
