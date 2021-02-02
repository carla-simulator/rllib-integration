#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import signal
import subprocess
import time
import psutil
import logging

from rllib_integration.sensors.camera_manager import CameraManager
from rllib_integration.sensors.bird_view_manager import BirdviewManager
from rllib_integration.sensors.carla_debug import get_actor_display_name
from rllib_integration.sensors.sensors_manager import *


BASE_CORE_CONFIG = {
    "RAY_DELAY": 1,  # Delay between 0 & RAY_DELAY before starting server so not all servers are launched simultaneously
    "retries_on_error": 30,
    "timeout": 60.0,
    "host": "localhost",
    "map_buffer": 1.2,  # To find the minimum and maximum coordinates of the map
}


def is_used(port):
    return port in [conn.laddr.port for conn in psutil.net_connections()]


class CarlaCore:
    def __init__(self, environment_config, experiment_config, core_config={}):
        """
        Initialize the server, clients, hero and sensors
        :param environment_config: Environment Configuration
        :param experiment_config: Experiment Configuration
        """
        self.core_config = BASE_CORE_CONFIG.copy()
        self.core_config.update(core_config)
        self.environment_config = environment_config
        self.experiment_config = experiment_config

        self.init_server(self.core_config["RAY_DELAY"])

        self.client, self.world, self.town_map, self.actors = self.__connect_client(
            self.core_config["host"],
            self.server_port,
            self.core_config["timeout"],
            self.core_config["retries_on_error"],
            self.experiment_config["Disable_Rendering_Mode"],
            True,
            self.experiment_config["Weather"],
            self.experiment_config["town"]
        )

        self.set_map_dimensions()
        self.camera_manager = None
        self.collision_sensor = None
        self.radar_sensor = None
        self.imu_sensor = None
        self.gnss_sensor = None
        self.lane_sensor = None
        self.birdview_sensor = None

    # ==============================================================================
    # -- ServerSetup -----------------------------------------------------------
    # ==============================================================================
    def init_server(self, ray_delay=0):
        """
        Start a server on a random port
        :param ray_delay: Delay so not all servers start simultaneously causing race condition
        :return:
        """
        # Generate a random port to connect to. You need one port for each server-client
        if self.environment_config["DEBUG_MODE"]:
            self.server_port = 2000
        else:
            self.server_port = random.randint(15000, 32000)
        # Create a new server process and start the client.
        if self.environment_config["RAY"] is True:
            # Ray tends to start all processes simultaneously. This causes problems
            # => random delay to start individual servers
            delay_sleep = random.uniform(0, ray_delay)
            time.sleep(delay_sleep)

        if self.environment_config["DEBUG_MODE"] is True:
            # Big Screen for Debugging
            for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
                self.experiment_config["SENSOR_CONFIG"]["CAMERA_X"][i] = 900
                self.experiment_config["SENSOR_CONFIG"]["CAMERA_Y"][i] = 1200
            self.experiment_config["quality_level"] = "High"

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port+1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port+1)

        # Run the server process
        server_command = [
            "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
            "-windowed",
            "-ResX={}".format(self.core_config["resolution_x"]),
            "-ResY={}".format(self.core_config["resolution_y"]),
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.core_config["quality_level"]),
            "--no-rendering",
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    @staticmethod
    def kill_all_servers():
        """
        Kill all PIDs that start with Carla. Do this if you running a single server
        :return:
        """
        processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
        for process in processes:
            os.kill(process.pid, signal.SIGKILL)

    # ==============================================================================
    # -- ClientSetup -----------------------------------------------------------
    # ==============================================================================
    def __connect_client(self, host, port, timeout, num_retries, disable_rendering_mode, sync_mode, weather, town):
        """
        Connect the client

        :param host: The host servers
        :param port: The server port to connect to
        :param timeout: The server takes time to get going, so wait a "timeout" and re-connect
        :param num_retries: Number of times to try before giving up
        :param disable_rendering_mode: True to disable rendering
        :param sync_mode: True for RL
        :param weather: The weather to start the world
        :param town: current town
        :return:
        """

        for i in range(num_retries):
            try:
                carla_client = carla.Client(host, port)
                carla_client.set_timeout(timeout)
                carla_client.load_world(
                    map_name=town,
                    map_layers=carla.MapLayer.NONE if self.core_config["subleveling"] else carla.MapLayer.All
                )

                world = carla_client.get_world()

                town_map = world.get_map()
                actors = world.get_actors()
                world.set_weather(weather)
                world.wait_for_tick()

                settings = world.get_settings()
                settings.no_rendering_mode = disable_rendering_mode
                settings.synchronous_mode = sync_mode
                settings.fixed_delta_seconds = self.core_config["timestep"]

                world.apply_settings(settings)

                print("Server setup is complete")

                return carla_client, world, town_map, actors

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, num_retries))
                time.sleep(3)
        # if (i + 1) == num_retries:
        raise Exception("Can not connect to server. Try increasing timeouts or num_retries")

    # ==============================================================================
    # -- MapDigestionsSetup -----------------------------------------------------------
    # ==============================================================================

    def set_map_dimensions(self):

        """
        From the spawn points, we get min and max and add some buffer so we can normalize the location of agents (0..1)
        This allows you to get the location of the vehicle between 0 and 1

        :input
        self.core_config["map_buffer"]. Because we use spawn points, we add a buffer as vehicle can drive off the road

        :output:
        self.coord_normalization["map_normalization"] = Using larger of (X,Y) axis to normalize x,y
        self.coord_normalization["map_min_x"] = minimum x coordinate
        self.coord_normalization["map_min_y"] = minimum y coordinate
        :return: None
        """

        map_buffer = self.core_config["map_buffer"]
        spawn_points = list(self.world.get_map().get_spawn_points())

        min_x = min_y = 1000000
        max_x = max_y = -1000000

        for spawn_point in spawn_points:
            min_x = min(min_x, spawn_point.location.x)
            max_x = max(max_x, spawn_point.location.x)

            min_y = min(min_y, spawn_point.location.y)
            max_y = max(max_y, spawn_point.location.y)

        center_x = (max_x+min_x)/2
        center_y = (max_y+min_y)/2

        x_buffer = (max_x - center_x) * map_buffer
        y_buffer = (max_y - center_y) * map_buffer

        min_x = center_x - x_buffer
        max_x = center_x + x_buffer

        min_y = center_y - y_buffer
        max_y = center_y + y_buffer

        self.coord_normalization = {"map_normalization": max(max_x - min_x, max_y - min_y),
                                    "map_min_x": min_x,
                                    "map_min_y": min_y}

    def normalize_coordinates(self, input_x, input_y):

        """
        :param input_x: X location of your actor
        :param input_y: Y location of your actor
        :return: The normalized location of your actor
        """
        output_x = (input_x - self.coord_normalization["map_min_x"]) / self.coord_normalization["map_normalization"]
        output_y = (input_y - self.coord_normalization["map_min_y"]) / self.coord_normalization["map_normalization"]

        # ToDO Possible bug (Clipped the observation and still didn't stop the observations from being under
        output_x = float(np.clip(output_x, 0, 1))
        output_y = float(np.clip(output_y, 0, 1))

        return output_x, output_y

    # ==============================================================================
    # -- SensorSetup -----------------------------------------------------------
    # ==============================================================================

    def setup_sensors(self, experiment_config, hero, synchronous_mode=True):
        """
        This function sets up hero vehicle sensors

        :param experiment_config: Sensor configuration for you sensors
        :param hero: Hero vehicle
        :param synchronous_mode: set to True for RL
        :return:
        """
        for i in range(0,len(experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"])):
            if experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"][i]:
                self.camera_manager = CameraManager(
                    hero,
                    experiment_config["SENSOR_CONFIG"]["CAMERA_X"],
                    experiment_config["SENSOR_CONFIG"]["CAMERA_Y"],
                    experiment_config["SENSOR_CONFIG"]["CAMERA_FOV"],
                )
                sensor = experiment_config["SENSOR_CONFIG"]["SENSOR"][i].value
                transform_index = experiment_config["SENSOR_CONFIG"]["SENSOR_TRANSFORM"][i].value
                self.camera_manager.set_sensor(sensor, transform_index, synchronous_mode=synchronous_mode)

        if experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            self.collision_sensor = CollisionSensor(
                hero, synchronous_mode=False
            )
        if experiment_config["OBSERVATION_CONFIG"]["RADAR_OBSERVATION"]:
            self.radar_sensor = RadarSensor(
                hero, synchronous_mode=synchronous_mode
            )
        if experiment_config["OBSERVATION_CONFIG"]["IMU_OBSERVATION"]:
            self.imu_sensor = IMUSensor(
                hero, synchronous_mode=synchronous_mode
            )
        if experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            self.lane_sensor = LaneInvasionSensor(
                hero, synchronous_mode=synchronous_mode
            )
        if experiment_config["OBSERVATION_CONFIG"]["GNSS_OBSERVATION"]:
            self.gnss_sensor = GnssSensor(
                hero, synchronous_mode=synchronous_mode
            )
        if experiment_config["OBSERVATION_CONFIG"]["BIRDVIEW_OBSERVATION"]:
            size = experiment_config["BIRDVIEW_CONFIG"]["SIZE"]
            radius = experiment_config["BIRDVIEW_CONFIG"]["RADIUS"]
            self.birdview_sensor = BirdviewManager(
                self.world, size, radius, hero, synchronous_mode=synchronous_mode
            )

    def reset_sensors(self, experiment_config):
        """
        Destroys sensors that were setup in this class
        :param experiment_config: sensors configured in the experiment
        :return:
        """
        if experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]:
            if self.camera_manager is not None:
                self.camera_manager.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            if self.collision_sensor is not None:
                self.collision_sensor.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["RADAR_OBSERVATION"]:
            if self.radar_sensor is not None:
                self.radar_sensor.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["IMU_OBSERVATION"]:
            if self.imu_sensor is not None:
                self.imu_sensor.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            if self.lane_sensor is not None:
                self.lane_sensor.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["GNSS_OBSERVATION"]:
            if self.gnss_sensor is not None:
                self.gnss_sensor.destroy_sensor()
        if experiment_config["OBSERVATION_CONFIG"]["BIRDVIEW_OBSERVATION"]:
            if self.birdview_sensor is not None:
                self.birdview_sensor.destroy_sensor()

    # ==============================================================================
    # -- CameraSensor -----------------------------------------------------------
    # ==============================================================================

    def record_camera(self, record_state):
        self.camera_manager.set_recording(record_state)

    def render_camera_lidar(self, render_state):
        self.camera_manager.set_rendering(render_state)

    def update_camera(self):
        self.camera_manager.read_image_queue()

    def get_camera_data(self):
        return self.camera_manager.get_camera_data()

    # ==============================================================================
    # -- CollisionSensor -----------------------------------------------------------
    # ==============================================================================

    def update_collision(self):
        self.collision_sensor.read_collision_queue()

    def get_collision_data(self):
        return self.collision_sensor.get_collision_data()

    # ==============================================================================
    # -- LaneInvasionSensor -----------------------------------------------------------
    # ==============================================================================

    def update_lane_invasion(self):
        self.lane_sensor.read_lane_queue()

    def get_lane_data(self):
        return self.lane_sensor.get_lane_data()

    # ==============================================================================
    # -- GNSSSensor -----------------------------------------------------------
    # ==============================================================================

    def update_gnss(self):
        self.gnss_sensor.read_gnss_queue()

    def get_gnss_data(self):
        return self.gnss_sensor.get_gnss_data()

    # ==============================================================================
    # -- IMUSensor -----------------------------------------------------------
    # ==============================================================================

    def update_imu_invasion(self):
        self.imu_sensor.read_imu_queue()

    def get_imu_data(self):
        return self.imu_sensor.get_imu_data()

    # ==============================================================================
    # -- RadarSensor -----------------------------------------------------------
    # ==============================================================================

    def update_radar_invasion(self):
        self.radar_sensor.read_radar_queue()

    def get_radar_data(self):
        return self.radar_sensor.get_radar_data()

    # ==============================================================================
    # -- BirdViewSensor -----------------------------------------------------------
    # ==============================================================================

    def update_birdview(self):
        self.birdview_sensor.read_birdview_queue()

    def get_birdview_data(self):
        return self.birdview_sensor.get_birdview_data()

    # ==============================================================================
    # -- OtherForNow -----------------------------------------------------------
    # ==============================================================================

    def get_core_world(self):
        return self.world

    def get_core_client(self):
        return self.client

    def get_nearby_vehicles(self, world, hero_actor, max_distance=200):
        vehicles = world.get_actors().filter("vehicle.*")
        surrounding_vehicles = []
        surrounding_vehicle_actors = []
        _info_text = []
        if len(vehicles) > 1:
            _info_text += ["Nearby vehicles:"]
            for x in vehicles:
                if x.id != hero_actor:
                    loc1 = hero_actor.get_location()
                    loc2 = x.get_location()
                    distance = math.sqrt(
                        (loc1.x - loc2.x) ** 2
                        + (loc1.y - loc2.y) ** 2
                        + (loc1.z - loc2.z) ** 2
                    )
                    vehicle = {}
                    if distance < max_distance:
                        vehicle["vehicle_type"] = get_actor_display_name(x, truncate=22)
                        vehicle["vehicle_location"] = x.get_location()
                        vehicle["vehicle_velocity"] = x.get_velocity()
                        vehicle["vehicle_distance"] = distance
                        surrounding_vehicles.append(vehicle)
                        surrounding_vehicle_actors.append(x)

    def spawn_npcs(self, n_vehicles, n_walkers, hybrid=False, seed=None):
        """
        Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters

        :param n_vehicles: Number of vehicles
        :param n_walkers: Number of walkers
        :param hybrid: Activates hybrid physics mode
        :param seed: Activates deterministic mode
        :return: None
        """

        tm_port = self.server_port//10 + self.server_port%10
        while is_used(tm_port):
            print("Is using the TM port: " + str(tm_port))
            tm_port+=1
        traffic_manager = self.client.get_trafficmanager(tm_port)
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)
        traffic_manager.set_synchronous_mode(True)

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if n_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, n_vehicles, number_of_spawn_points)
            n_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        walkers_list = []
        batch = []
        vehicles_list = []
        all_id = []
        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(n_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        self.world.tick()
