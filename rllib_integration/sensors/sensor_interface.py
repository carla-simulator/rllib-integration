#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import queue

class SensorInterface(object):
    """
    Class used to handle all the sensor data management
    """

    def __init__(self):
        self._sensors = {}  # {name: Sensor object}
        self._data_buffers = queue.Queue()
        self._queue_timeout = 10

        self._event_sensors = {}
        self._event_data_buffers = queue.Queue()

    @property
    def sensors(self):
        sensors = self._sensors.copy()
        sensors.update(self._event_sensors)
        return sensors

    def destroy(self):
        for sensor in self.sensors.values():
            sensor.destroy()
        self._data_buffers = queue.Queue()
        self._event_data_buffers = queue.Queue()

    def register(self, name, sensor):
        """Adds a specific sensor to the class"""
        if sensor.is_event_sensor():
            self._event_sensors[name] = sensor
        else:
            self._sensors[name] = sensor

    def get_data(self):
        """Returns the data of all the registered sensors as a dictionary {sensor_name: sensor_data}"""
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors.keys()):
                sensor_data = self._data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = (sensor_data[1], sensor_data[2])

        except queue.Empty:
            raise RuntimeError("A sensor took too long to send their data")

        for event_sensor in self._event_sensors:
            try:
                sensor_data = self._event_data_buffers.get_nowait()
                data_dict[sensor_data[0]] = (sensor_data[1], sensor_data[2])
            except queue.Empty:
                pass
        # print("Data buffer: {}".format(len(self._data_buffers.queue)))
        # print("Event data buffer: {}".format(len(self._event_data_buffers.queue)))
        return data_dict
