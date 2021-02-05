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

import carla

from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory
from rllib_integration.helper import join_dicts

BASE_CORE_CONFIG = {
    "host": "localhost",  # Client host
    "timeout": 10.0,  # Timeout of the client
    "timestep": 0.05,  # Time step of the simulation
    "retries_on_error": 10,  # Number of tries to connect to the client
    "resolution_x": 600,  # Width of the server spectator camera
    "resolution_y": 600,  # Height of the server spectator camera
    "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
    "enable_map_assets": False,  # enable / disable all town assets except for the road
    "enable_rendering": True,  # enable / disable camera images
    "show_display": False  # Whether or not the server will be displayed
}


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class CarlaCore:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """
    def __init__(self, config={}):
        """Initialize the server and client"""
        self.client = None
        self.world = None
        self.map = None
        self.hero = None
        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.sensor_interface = SensorInterface()

        self.init_server()
        self.connect_client()

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port+1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"])
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        # Spawn the background activity
        self.spawn_npcs(
            experiment_config["background_activity"]["n_vehicles"],
            experiment_config["background_activity"]["n_walkers"],
            experiment_config["background_activity"]["tm_hybrid_mode"]
        )

        # Load the map
        if self.config["enable_map_assets"]:
            map_layer = carla.MapLayer.All
        else:
            map_layer = carla.MapLayer.NONE

        self.world = self.client.load_world(
            map_name = experiment_config["town"],
            reset_settings = False,
            map_layers = map_layer)

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, experiment_config["weather"])
        self.world.set_weather(weather)

    def reset_hero(self, hero_config):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface.destroy()

        self.world.tick()

        # Part 2: Spawn the ego vehicle
        user_spawn_points = hero_config["spawn_points"]
        if user_spawn_points:
            spawn_points = []
            for transform in user_spawn_points:

                transform = [float(x) for x in transform.split(",")]
                if len(transform) == 3:
                    location = carla.Location(
                        transform[0], transform[1], transform[2]
                    )
                    waypoint = self.map.get_waypoint(location)
                    waypoint = waypoint.previous(random.uniform(0, 10))[0]
                    transform = carla.Transform(
                        location, waypoint.transform.rotation
                    )
                else:
                    assert len(transform) == 6
                    transform = carla.Transform(
                        carla.Location(transform[0], transform[1], transform[2]),
                        carla.Rotation(transform[4], transform[5], transform[3])
                    )
                spawn_points.append(transform)
        else:
            spawn_points = self.map.get_spawn_points()

        self.hero_blueprints = self.world.get_blueprint_library().find(hero_config['blueprint'])
        self.hero_blueprints.set_attribute("role_name", "hero")

        # If already spawned, destroy it
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        random.shuffle(spawn_points, random.random)
        for i in range(0,len(spawn_points)):
            next_spawn_point = spawn_points[i % len(spawn_points)]
            self.hero = self.world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if self.hero is not None:
                print("Hero spawned!")
                break
            else:
                print("Could not spawn hero, changing spawn point")

        if self.hero is None:
            print("We ran out of spawn points")
            return

        self.world.tick()

        # Part 3: Spawn the new sensors
        for name, attributes in hero_config["sensors"].items():
            sensor = SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)

        # Not needed anymore. This tick will happen when calling CarlaCore.tick()
        # self.world.tick()

        return self.hero

    def spawn_npcs(self, n_vehicles, n_walkers, tm_hybrid_mode=False, seed=None): #TODO: remake + seed
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""

        tm_port = self.server_port//10 + self.server_port%10
        while is_used(tm_port):
            print("Is using the TM port: " + str(tm_port))
            tm_port+=1
        traffic_manager = self.client.get_trafficmanager(tm_port)
        if tm_hybrid_mode:
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

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calcualted at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))
        return sensor_data
