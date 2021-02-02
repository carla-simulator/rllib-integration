#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import weakref
import queue

import carla
from carla import ColorConverter as cc

import numpy as np
import pygame

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, image_size_x, image_size_y, image_fov, gamma_correction=2.2):
        self.sensors = None
        self.surface = None
        self._parent = parent_actor
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.image_fov = image_fov
        self.gamma_correction = gamma_correction
        self.dim = [self.image_size_x, self.image_size_y]
        self.image_calibration = None
        self.recording = False
        self.render = False
        self.display = None
        self.camera_data = None

        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors_list = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)"],
            ["sensor.camera.depth", cc.LogarithmicDepth, "Camera Depth (Logarithmic Gray Scale)"],
            ["sensor.camera.semantic_segmentation", cc.Raw, "Camera Semantic Segmentation (Raw)"],
            ["sensor.camera.semantic_segmentation", cc.CityScapesPalette, "Camera Semantic Segmentation (CityScapes Palette)"],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor'],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '5.0',
                'lens_circle_falloff': '5.0',
                'chromatic_aberration_intensity': '0.8',
                'chromatic_aberration_offset': '0'}]
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors_list:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(self.image_size_x))
                bp.set_attribute("image_size_y", str(self.image_size_y))
                bp.set_attribute("fov", str(self.image_fov))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(self.gamma_correction))

                calibration = np.identity(3)
                calibration[0, 2] = self.image_size_x / 2.0
                calibration[1, 2] = self.image_size_y / 2.0
                calibration[0, 0] = calibration[1, 1] = self.image_size_x / (
                    2.0 * np.tan(self.image_fov * np.pi / 360.0)
                )
                self.image_calibration = calibration

            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)
            item.append(bp)

            bp.set_attribute("sensor_tick", "0.00001")

        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(
            self.index, force_respawn=True, synchronous_mode=True
        )

    def destroy_sensor(self):
        if self.sensors is not None:
            self.sensors.destroy()
            self.surface = None
            self.sensors = None

    def set_sensor(self, index, transform_index, force_respawn=False, synchronous_mode=True):
        self.synchronous_mode = synchronous_mode
        index = index % len(self.sensors_list)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors_list[index][0] != self.sensors_list[self.index][0])
            )
        )
        if needs_respawn:
            if self.sensors is not None:
                self.sensors.destroy()
                self.surface = None
            self.sensors = self._parent.get_world().spawn_actor(
                self.sensors_list[index][-1],
                self._camera_transforms[transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            if not self.synchronous_mode:
                # We need to pass the lambda a weak reference to self to avoid
                # circular reference.
                weak_self = weakref.ref(self)
                self.sensors.listen(
                    lambda image: CameraManager._parse_image(weak_self, image)
                )
                self.last_image = None
            else:
                # Make sync queue for sensor data.
                self.camera_queue = queue.Queue()
                self.sensors.listen(self.camera_queue.put)
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording

    def set_recording(self, record_state):
        self.recording = record_state

    def set_rendering(self, render_state):
        self.render = render_state

    def read_image_queue(self):
        weak_self = weakref.ref(self)
        if (not self.synchronous_mode) and (self.last_image is not None):
            CameraManager._parse_image(weak_self, self.last_image)
        if self.synchronous_mode:
            try:
                CameraManager._parse_image(weak_self, self.camera_queue.get(True))
            except:
                print("We couldn't read Image")
                # Ignore empty Que
                pass

    def get_camera_data(self):

        if self.camera_data is None:
            return None

        return self.camera_data.astype(np.float32)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self.last_image = image
        if self.sensors_list[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
            if self.render:
                self.surface = pygame.surfarray.make_surface(lidar_img)
                if self.surface is not None:
                    if self.display is None:
                        self.display = pygame.display.set_mode(
                            (self.image_size_x, self.image_size_y),
                            pygame.HWSURFACE | pygame.DOUBLEBUF,
                        )
                    # ToDO save the output of the Lidar Image instead of real time visualization
                    self.display.blit(self.surface, (0, 0))
                    pygame.display.flip()
        elif self.sensors_list[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
            if self.render:
                self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
                if self.surface is not None:
                    if self.display is None:
                        self.display = pygame.display.set_mode(
                            (self.image_size_x, self.image_size_y),
                            pygame.HWSURFACE | pygame.DOUBLEBUF,
                        )
                    self.display.blit(self.surface, (0, 0))
                    pygame.display.flip()
        else:
            image.convert(self.sensors_list[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_data = array
            if self.render:
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                if self.surface is not None:
                    if self.display is None:
                        self.display = pygame.display.set_mode(
                            (self.image_size_x, self.image_size_y),
                            pygame.HWSURFACE | pygame.DOUBLEBUF,
                        )
                    self.display.blit(self.surface, (0, 0))
                    pygame.display.flip()
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame_number)
