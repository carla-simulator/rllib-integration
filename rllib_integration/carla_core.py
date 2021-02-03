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
from rllib_integration.sensors.sensors_manager import *

from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory


BASE_CORE_CONFIG = {
    "host": "localhost",
    "timeout": 10.0,
    "sync_mode": True,
    "timestep": 0.05,
    "retries_on_error": 10,
    "resolution_x": 600,
    "resolution_y": 600,
    "quality_level": "Low",
    "enable_map_assets": True,  # enable / disable all town assets except for the road
    "enable_rendering": True,  # enable / disable camera images
    "ray": True,  # Same as above
    "ray_delay": 1,  # Delay between 0 & RAY_DELAY before starting server so not all servers are launched simultaneously
    "debug": False  # TODO: use it
}


def is_used(port):
    return port in [conn.laddr.port for conn in psutil.net_connections()]


class CarlaCore:
    def __init__(self, core_config={}):
        """
        Initialize the server, clients, hero and sensors
        :param environment_config: Environment Configuration
        :param experiment_config: Experiment Configuration
        """
        self.config = BASE_CORE_CONFIG.copy()
        self.config.update(core_config) # TODO: remove ray and debug from 'carla'

        self.init_server()
        self.connect_client()

        self.sensor_interface = SensorInterface()
        self.hero = None

    def setup_experiment(self, experiment_config):

        # Spawn traffic
        self.spawn_npcs(
            experiment_config["n_vehicles"],
            experiment_config["n_walkers"],
            hybrid = True
        )

        if self.config["enable_map_assets"]:
            map_layer = carla.MapLayer.All
        else:
            map_layer = carla.MapLayer.NONE

        self.world = self.client.load_world(
            map_name = experiment_config["town"],
            reset_settings = False,
            map_layers = map_layer)

        self.world.set_weather(experiment_config["weather"])
        self.town_map = self.world.get_map()
        # TODO: move weather and load town to here from connect_client
        self.actors = self.world.get_actors()
    # ==============================================================================
    # -- Tick -----------------------------------------------------------
    # ==============================================================================

    def tick(self):
        self.world.tick()
        self.set_server_view()

    def set_server_view(self):
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
        self.spectator = self.get_core_world().get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    # ==============================================================================
    # -- ServerSetup -----------------------------------------------------------
    # ==============================================================================
    def init_server(self):
        """
        Start a server on a random port
        :param ray_delay: Delay so not all servers start simultaneously causing race condition
        :return:
        """
        # Generate a random port to connect to. You need one port for each server-client
        # if self.environment_config["debug"]:
        #     self.server_port = 2000
        # else:
        self.server_port = random.randint(15000, 32000)
        # Create a new server process and start the client.
        if self.config["ray"] is True:
            # Ray tends to start all processes simultaneously. This causes problems
            # => random delay to start individual servers
            delay_sleep = random.uniform(0, self.config["ray_delay"])
            time.sleep(delay_sleep)

        # if self.environment_config["debug"] is True:
        #     # Big Screen for Debugging
        #     for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
        #         self.experiment_config["SENSOR_CONFIG"]["CAMERA_X"][i] = 900
        #         self.experiment_config["SENSOR_CONFIG"]["CAMERA_Y"][i] = 1200
        #     self.experiment_config["quality_level"] = "High"

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
            "-ResX={}".format(self.config["resolution_x"]),
            "-ResY={}".format(self.config["resolution_y"]),
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"]),
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
    def connect_client(self):
        """
        Connect the client

        :param host: The host servers
        :param port: The server port to connect to
        :param timeout: The server takes time to get going, so wait a "timeout" and re-connect
        :param num_retries: Number of times to try before giving up
        :param disable_rendering_mode: True to disable rendering
        :param sync_mode: True for RL
        :return:
        """

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = self.config["sync_mode"]
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                print("Server setup is complete")
                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    # ==============================================================================
    # -- SensorSetup -----------------------------------------------------------
    # ==============================================================================

    def setup_sensors(self,sensor_config, hero, sync_mode):
        """
        This function sets up hero vehicle sensors

        :param experiment_config: Sensor configuration for you sensors
        :param hero: Hero vehicle
        :param synchronous_mode: set to True for RL
        :return:
        """
        for name, attributes in sensor_config.items():
            sensor = SensorFactory.spawn(name, attributes, self.sensor_interface, hero)
        return self.sensor_interface.sensors

    def reset_sensors(self, sensor_config):
        """
        Destroys sensors that were setup in this class
        :param experiment_config: sensors configured in the experiment
        :return:
        """
        for sensor in self.sensor_interface.sensors.values():
            sensor.destroy()

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
                        vehicle["vehicle_type"] = x.type_id
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

    # ==============================================================================
    # -- Hero -----------------------------------------------------------
    # ==============================================================================
    def reset_hero(self, experiment_config):

        """
        This function spawns the hero vehicle. It makes sure that if a hero exists, it destroys the hero and respawn
        :param core:
        :param transform: Hero location
        :return:
        """
        spawn_points = self.town_map.get_spawn_points()

        self.hero_blueprints = self.world.get_blueprint_library().find(experiment_config['hero_blueprint'])
        self.hero_blueprints.set_attribute("role_name", "hero")

        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        random.shuffle(spawn_points, random.random)
        for i in range(0,len(spawn_points)):
            next_spawn_point = spawn_points[i % len(spawn_points)]
            self.hero = self.world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if self.hero is not None:
                break
            else:
                print("Could not spawn hero, changing spawn point")

        if self.hero is None:
            print("We ran out of spawn points")
            return

        self.world.tick()
        print("Hero spawned!")
        self.start_location = spawn_points[i].location
        self.past_action = carla.VehicleControl(0.0, 0.00, 0.0, False, False)

    def get_hero(self):

        """
        Get hero vehicle
        :return:
        """
        return self.hero
